#!/usr/bin/env python3
"""
把 datasets/raw/ 下的眼底数据集转换成本项目需要的格式：

    datasets/Fundus/<名字>/<图像>            (指向 raw 的符号链接，不复制数据)
    datasets/Fundus/<名字>_<划分>.json       (COCO instance segmentation)

这个布局正是 README「Datasets」一节规定、并由 data/datasets/builtin.py 注册的。

类别固定为两类（对应 configs/test_segment.yaml 的 MODEL.ROI_HEADS.NUM_CLASSES: 2）：
    id 1 -> optic_disc (视盘)    load_coco_json 会按 id 升序重映射成 0
    id 2 -> optic_cup  (视杯)                               重映射成 1

各数据集的标注编码差异很大，脚本里逐个处理（都经过实际像素验证）：

  REFUGE      值 0=视杯, 128=视盘环, 255=背景
              => disc = (gt != 255), cup = (gt == 0)
              train/val 有掩膜；test 400 张**没有**掩膜

  Drishti_GS  SoftMap 软概率图（0/64/128/191/255，多专家一致性）
              => 阈值 >=128 二值化
              另：掩膜 2045x1752 比图像 2047x1760 小，需最近邻放大对齐

  RIM-ONE r3  Expert1/Expert2/Average 三套掩膜，默认取 Average（专家共识）
              该下载**没有官方 train/test 划分**，脚本用固定种子的分层划分生成

  ORIGA       本下载只有 jpg + 青光眼分类 csv，**没有任何分割掩膜**，直接跳过。
              需要另找带 OD/OC 掩膜的 ORIGA-650（iMED 官方需申请）。

用法：
    python tools/prepare_fundus_data.py                 # 转换全部可用的
    python tools/prepare_fundus_data.py --datasets REFUGE Drishti_GS
    python tools/prepare_fundus_data.py --rimone-expert exp1
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "datasets", "raw")
OUT = os.path.join(REPO, "datasets", "Fundus")

# ⚠️ 类别 ID 的顺序必须与作者训练权重时一致，否则评测会严重偏低。
#
# detectron2 的 load_coco_json 按 category id 升序重映射成 0..N-1 的连续标签，
# 所以"id 小的"就是模型里的类 0。实测 weights/fundus_source/*.pth：
#   * 类 1 的预测置信度极高（中位 0.99）且掩膜面积中位数 ≈ 15000 px
#   * 类 0 几乎没有高分预测
# 而 REFUGE 的 GT 面积中位数是 视盘 11608 / 视杯 2580 —— 也就是说
# **模型的类 1 是视盘、类 0 是视杯**。
#
# 若反过来（disc 占 id 1），模型的视盘预测会被拿去和 GT 的视杯比：
# Dice 上限 = 2*2580/(15000+2580) ≈ 29%，实测 34.69%，正好卡在这个上限附近。
# 交换后 REFUGE 从 ~34% 升到 ~80%（见 docs/FUNDUS_DATA.md）。
CUP_ID, DISC_ID = 1, 2
CATEGORIES = [
    {"id": CUP_ID, "name": "optic_cup"},
    {"id": DISC_ID, "name": "optic_disc"},
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def is_junk(path):
    """Windows 'Zone.Identifier' alternate-data-stream sidecars."""
    return path.endswith(":Zone.Identifier") or "Zone.Identifier" in os.path.basename(path)


def listdir_clean(path, pattern="*"):
    return sorted(
        p for p in glob.glob(os.path.join(path, pattern)) if not is_junk(p)
    )


def polygon_area(poly):
    xs, ys = poly[0::2], poly[1::2]
    n = len(xs)
    acc = 0.0
    for i in range(n):
        j = (i + 1) % n
        acc += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(acc) / 2.0


def polygon_bbox(poly):
    xs, ys = poly[0::2], poly[1::2]
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs) - x0, max(ys) - y0]


def mask_to_polygons(mask, min_area=20.0, eps_ratio=0.002):
    """二值掩膜 -> COCO 多边形列表（按面积降序）。

    只保留外轮廓（视盘/视杯都是单连通区域），并用 approxPolyDP 抽稀，
    避免 SoftMap 阈值化后产生上千个点的锯齿边界。
    """
    m = (np.asarray(mask).astype(np.uint8)) * 255
    if m.sum() == 0:
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = eps_ratio * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
        if len(approx) < 3:
            continue
        polys.append([round(float(v), 2) for pt in approx for v in pt])
    polys.sort(key=lambda p: -polygon_area(p))
    return polys


def fundus_roi_box(img, thresh=7, square=True, pad=0):
    """求眼底图像的 ROI（视网膜圆形区域）裁剪框。

    眼底图四周是黑色背景，直接用灰度阈值取非黑区域的外接框即可。
    视网膜是圆形，其外接框近似正方形，所以裁成正方形再缩放不会破坏几何比例
    —— 这正是论文「cropping the ROI ... to 800x800」能成立的前提。
    """
    gray = np.asarray(img.convert("L"))
    ys, xs = np.where(gray > thresh)
    if len(xs) < 100:                       # 阈值失效（整图过暗/过亮）时退回整图
        return 0, 0, img.width, img.height
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if square:
        side = max(x1 - x0, y1 - y0) + 2 * pad
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, y0 = cx - side // 2, cy - side // 2
        x1, y1 = x0 + side, y0 + side
        # 平移到图像范围内（保持尺寸不变）
        x0 = max(0, min(x0, img.width - side))
        y0 = max(0, min(y0, img.height - side))
        x1, y1 = min(img.width, x0 + side), min(img.height, y0 + side)
    return x0, y0, x1, y1


class CocoBuilder:
    """累积一个 COCO 数据集。

    两种模式：
      * ``mode="crop"``（默认，对应论文协议）：按 ROI 裁剪并缩放到
        ``roi_size x roi_size``，**写出真实图像文件**，掩膜同步做相同变换。
      * ``mode="link"``：原图不动，只建符号链接（用于快速冒烟，省磁盘）。
    """

    def __init__(self, name, split, mode="crop", roi_size=800, jpeg_quality=95):
        self.name = name
        self.split = split
        self.mode = mode
        self.roi_size = roi_size
        self.jpeg_quality = jpeg_quality
        self.image_root = os.path.join(OUT, name)
        os.makedirs(self.image_root, exist_ok=True)
        self.images = []
        self.annotations = []
        self._img_id = 0
        self._ann_id = 0
        self.skipped = 0
        self.cropped = 0

    def _to_square(self, mask, width, height):
        """把掩膜对齐到图像尺寸（Drishti 的掩膜比图像小 2x8 像素）。"""
        if mask is None or mask.shape == (height, width):
            return mask
        return np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST
            )
        ) > 0

    def add(self, src_image, masks, extra_crop=None):
        """masks: {category_id: 二值掩膜 ndarray}，形状须与图像一致。

        ``extra_crop`` = (x0, y0, x1, y1)，在 ROI 裁剪**之前**先执行。
        RIM-ONE 用它把左右并排的立体图切成单张眼底（见 convert_rimone）。
        """
        rel = os.path.basename(src_image)
        with Image.open(src_image) as im:
            im = im.convert("RGB")
            width, height = im.size
            masks = {k: self._to_square(v, width, height) for k, v in masks.items()}

            if extra_crop is not None:
                cx0, cy0, cx1, cy1 = extra_crop
                im = im.crop((cx0, cy0, cx1, cy1))
                masks = {
                    k: (None if v is None else v[cy0:cy1, cx0:cx1])
                    for k, v in masks.items()
                }
                width, height = im.size

            if self.mode == "crop":
                x0, y0, x1, y1 = fundus_roi_box(im)
                im = im.crop((x0, y0, x1, y1)).resize(
                    (self.roi_size, self.roi_size), Image.BILINEAR
                )
                cw, ch = self.roi_size, self.roi_size
                new_masks = {}
                for k, v in masks.items():
                    if v is None:
                        new_masks[k] = None
                        continue
                    sub = v[y0:y1, x0:x1]
                    new_masks[k] = np.array(
                        Image.fromarray(sub.astype(np.uint8) * 255).resize(
                            (cw, ch), Image.NEAREST
                        )
                    ) > 0
                masks = new_masks
                width, height = cw, ch
                self.cropped += 1
                out_path = os.path.join(self.image_root, rel)
                im.save(out_path, quality=self.jpeg_quality)
            else:
                link = os.path.join(self.image_root, rel)
                if not os.path.lexists(link):
                    os.symlink(
                        os.path.relpath(os.path.abspath(src_image), self.image_root), link
                    )

        self._img_id += 1
        self.images.append(
            {"file_name": rel, "height": height, "width": width, "id": self._img_id}
        )

        for cat_id, mask in masks.items():
            if mask is None:
                continue
            # 掩膜已在 add() 里随图像一起做过 ROI 裁剪+缩放，这里尺寸必然一致
            for poly in mask_to_polygons(mask):
                self._ann_id += 1
                self.annotations.append(
                    {
                        "id": self._ann_id,
                        "image_id": self._img_id,
                        "category_id": cat_id,
                        "iscrowd": 0,
                        "ignore": 0,
                        "bbox": polygon_bbox(poly),
                        "area": polygon_area(poly),
                        "segmentation": [poly],
                    }
                )

    def write(self):
        path = os.path.join(OUT, "{}_{}.json".format(self.name, self.split))
        with open(path, "w") as fh:
            json.dump(
                {
                    "type": "instance",
                    "categories": CATEGORIES,
                    "images": self.images,
                    "annotations": self.annotations,
                },
                fh,
            )
        return path


def report(name, split, builder, path):
    print(
        "  {:<16} {:<6} {:>4} images  {:>5} annotations  -> {}".format(
            name, split, len(builder.images), len(builder.annotations),
            os.path.relpath(path, REPO),
        )
    )


# --------------------------------------------------------------------------- #
# REFUGE:  0 = cup, 128 = disc annulus, 255 = background
# --------------------------------------------------------------------------- #
def convert_refuge(mode="crop", roi_size=800):
    root = os.path.join(RAW, "REFUGE", "REFUGE")
    if not os.path.isdir(root):
        print("[skip] REFUGE not found"); return

    for split, tag in (("train", "train"), ("val", "Valid")):
        b = CocoBuilder("REFUGE", tag, mode=mode, roi_size=roi_size)
        imgs = listdir_clean(os.path.join(root, split, "Images"), "*.jpg")
        for img_path in imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(root, split, "gts", stem + ".bmp")
            if not os.path.exists(gt_path):
                b.skipped += 1
                continue
            gt = np.array(Image.open(gt_path))
            b.add(img_path, {DISC_ID: gt != 255, CUP_ID: gt == 0})
        report("REFUGE", tag, b, b.write())

    # test 只有图像，没有 GT
    b = CocoBuilder("REFUGE", "test", mode=mode, roi_size=roi_size)
    for img_path in listdir_clean(os.path.join(root, "test", "Images"), "*.jpg"):
        b.add(img_path, {})
    report("REFUGE", "test", b, b.write())
    print("      ^ 注意：REFUGE test 的 400 张图**没有官方标注**，该 json 只有图像，"
          "在其上算 Dice 没有意义（见 docs/FUNDUS_DATA.md）")


# --------------------------------------------------------------------------- #
# Drishti-GS:  SoftMap 软图，阈值可调；掩膜尺寸比图像小，需放大
# --------------------------------------------------------------------------- #
def convert_drishti(mode="crop", roi_size=800, thresh=128):
    """SoftMap 是多专家一致性软图（取值 0/64/128/191/255），需阈值二值化。

    ``thresh`` 的含义（SoftMap 值 = 认可该像素为前景的专家比例）：
      * ``>=128``（默认，过半同意）——论文未指定，这是文献里最常见的取法；
      * ``>=255``（全体同意）——更保守，掩膜更小、杯盘比更低。

    Drishti 的杯盘比中位数在 ``>=128`` 下是 0.601，明显高于其它三个域
    （REFUGE 0.238 / RIM-ONE 0.223 / ORIGA 0.351），所以这个阈值可能偏松。
    用 ``--drishti-threshold 255`` 可以对比。
    """
    base = os.path.join(RAW, "Drishti_GS")
    if not os.path.isdir(base):
        print("[skip] Drishti_GS not found"); return

    specs = [
        ("train", glob.glob(os.path.join(base, "Training-*", "Training")), "GT"),
        ("test", glob.glob(os.path.join(base, "Test-*", "Test")), "Test_GT"),
    ]
    for split, dirs, gt_name in specs:
        if not dirs:
            continue
        root = dirs[0]
        b = CocoBuilder("Drishti_GS", split, mode=mode, roi_size=roi_size)
        imgs = listdir_clean(os.path.join(root, "Images", "*"), "*.png")
        for img_path in imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            sm = os.path.join(root, gt_name, stem, "SoftMap")
            od_p = os.path.join(sm, stem + "_ODsegSoftmap.png")
            cup_p = os.path.join(sm, stem + "_cupsegSoftmap.png")
            if not (os.path.exists(od_p) and os.path.exists(cup_p)):
                b.skipped += 1
                continue
            with Image.open(img_path) as im:
                w, h = im.size
            od = np.array(Image.open(od_p).resize((w, h), Image.NEAREST)) >= thresh
            cup = np.array(Image.open(cup_p).resize((w, h), Image.NEAREST)) >= thresh
            b.add(img_path, {DISC_ID: od, CUP_ID: cup})
        report("Drishti_GS", split, b, b.write())
    print("      ^ SoftMap 阈值 >={}".format(thresh))


def _stereo_half(img_path, disc_mask):
    """RIM-ONE 的 "Stereo Images" 是**左右两张眼底并排**拼在一张图里
    （2144x1424 = 2 x 1072x1424），而标注只画在其中一半上。

    实测全部 159 张的 disc 掩膜中心都落在左半（cx/(W/2) 在 0.46~0.56），
    但这里仍按掩膜实际位置判断，避免硬编码。

    不做这一步的后果很严重：模型会正确地分割出**两只眼**的视盘/视杯，
    而 GT 只有一份，多出来的预测在 DiceEvaluator 里全部被记为假阳性——
    RIM-ONE 的 Dice 会从 ~80% 掉到 18%（实测）。
    """
    with Image.open(img_path) as im:
        width, height = im.size
    if disc_mask is None or disc_mask.sum() == 0:
        return None
    _, xs = np.where(disc_mask)
    centre = (xs.min() + xs.max()) / 2.0
    half = width // 2
    return (0, 0, half, height) if centre < half else (half, 0, width, height)


# --------------------------------------------------------------------------- #
# RIM-ONE r3:  无官方划分 -> 固定种子分层划分
# --------------------------------------------------------------------------- #
def convert_rimone(expert="avg", test_frac=0.2, seed=20260323, mode="crop",
                   roi_size=800, split_stereo=True):
    base = os.path.join(RAW, "RIM-ONE-r3", "RIM-ONE r3")
    if not os.path.isdir(base):
        print("[skip] RIM-ONE-r3 not found"); return

    mask_dir = {"avg": "Average_masks", "exp1": "Expert1_masks", "exp2": "Expert2_masks"}[expert]
    # 实际命名：N-1-L-Disc-Avg.png / N-1-L-1-Disc-exp1.png / N-1-L-1-Disc-exp2.png
    tmpl = {
        "avg": "{stem}-{part}-Avg.png",
        "exp1": "{stem}-1-{part}-exp1.png",
        "exp2": "{stem}-1-{part}-exp2.png",
    }[expert]

    # 收集 (图像, disc掩膜, cup掩膜, 分层标签)
    items = []
    for sub, label in (("Healthy", "healthy"), ("Glaucoma and suspects", "glaucoma")):
        img_dir = os.path.join(base, sub, "Stereo Images")
        m_dir = os.path.join(base, sub, mask_dir)
        for img_path in listdir_clean(img_dir, "*.jpg"):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            disc_p = os.path.join(m_dir, tmpl.format(stem=stem, part="Disc"))
            cup_p = os.path.join(m_dir, tmpl.format(stem=stem, part="Cup"))
            if not (os.path.exists(disc_p) and os.path.exists(cup_p)):
                continue
            items.append((img_path, disc_p, cup_p, label))

    if not items:
        print("  [ERROR] RIM-ONE: 没找到任何 图像+掩膜 配对，检查 --rimone-expert "
              "与 raw 目录结构"); return

    # 论文 4.2 节：「Each source dataset is randomly divided into an 8:2
    # train/test split」——整体随机，不做分层。固定种子保证可复现。
    rng = np.random.default_rng(seed)
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    perm = rng.permutation(len(order))
    n_test = int(round(len(order) * test_frac))
    test_idx = set(perm[:n_test].tolist())
    train = [items[order[i]] for i in range(len(order)) if i not in test_idx]
    test = [items[order[i]] for i in range(len(order)) if i in test_idx]

    for split, group in (("train", train), ("test", test)):
        b = CocoBuilder("RIM_ONE_r3", split, mode=mode, roi_size=roi_size)
        for img_path, disc_p, cup_p, _ in group:
            disc = np.array(Image.open(disc_p)) > 0
            cup = np.array(Image.open(cup_p)) > 0
            extra = _stereo_half(img_path, disc) if split_stereo else None
            b.add(img_path, {DISC_ID: disc, CUP_ID: cup}, extra_crop=extra)
        report("RIM_ONE_r3", split, b, b.write())
    print("      ^ 该数据集无官方划分；按论文 4.2 节做随机 {:.0f}/{:.0f} 划分 "
          "(expert={}, seed={})".format(
              (1 - test_frac) * 100, test_frac * 100, expert, seed))


# --------------------------------------------------------------------------- #
# ORIGA:  互斥标签图 0=背景 / 1=视盘环 / 2=视杯
# --------------------------------------------------------------------------- #
def convert_origa(test_frac=0.2, seed=20260323, mode="crop", roi_size=800):
    """转换 datasets/raw/ORIGA-masked（带掩膜的第三方整理版）。

    掩膜编码**与 REFUGE 同类**：是互斥标签图 `{0,1,2}`，不是二值 0/255。
    实测 60 张里 59 张满足 `fill_holes(label1) == label1 | label2`，
    即 **label 1 是视盘环（盘减杯）、label 2 是视杯**，所以：

        disc = (mask != 0)      # 环 ∪ 杯 = 完整视盘
        cup  = (mask == 2)

    杯/盘面积比中位数 0.370，符合已知的杯盘比范围。

    用哪一套图：该整理版提供三套，**只用 `Images/`（全分辨率原图）**——
      * `Images_Cropped/` 是紧贴视盘的放大裁剪（只有视盘局部），框错了；
      * `Images_Square/` 是 512x512 的整幅眼底，分辨率低于原图。
    `Masks/` 与 `Images/` 逐像素对齐（实测三套图/掩膜尺寸都严格一致）。

    划分：论文 4.2 节说「每域随机 8:2」，这里照做。
    注意 `OrigaList.csv` 另有一个官方 `Set` 列（A/B 各 325，即 50/50），
    与论文的 8:2 不同，如需复现官方划分可改这里。
    """
    base = os.path.join(RAW, "ORIGA-masked", "ORIGA")
    if not os.path.isdir(base):
        print("[skip] ORIGA-masked not found"); return

    items = []
    for img_path in listdir_clean(os.path.join(base, "Images"), "*.jpg"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        msk_path = os.path.join(base, "Masks", stem + ".png")
        if not os.path.exists(msk_path):
            continue
        items.append((img_path, msk_path))
    if not items:
        print("  [ERROR] ORIGA: 没找到 图像+掩膜 配对"); return

    rng = np.random.default_rng(seed)
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    perm = rng.permutation(len(order))
    n_test = int(round(len(order) * test_frac))
    test_idx = set(perm[:n_test].tolist())
    train = [items[order[i]] for i in range(len(order)) if i not in test_idx]
    test = [items[order[i]] for i in range(len(order)) if i in test_idx]

    for split, group in (("train", train), ("test", test)):
        b = CocoBuilder("ORIGA", split, mode=mode, roi_size=roi_size)
        for img_path, msk_path in group:
            m = np.array(Image.open(msk_path))
            b.add(img_path, {DISC_ID: m != 0, CUP_ID: m == 2})
        report("ORIGA", split, b, b.write())
    print("      ^ 掩膜为互斥标签图：disc=(m!=0), cup=(m==2)；随机 {:.0f}/{:.0f} 划分 "
          "(seed={})".format((1 - test_frac) * 100, test_frac * 100, seed))



# --------------------------------------------------------------------------- #
# RIM-ONE DL:  官方 485 张单眼方形图 + 独立的参考分割（Disc/Cup）
# --------------------------------------------------------------------------- #
RIMONE_DL_IMG = os.path.join(RAW, "RIM-ONE_DL", "RIM-ONE_DL_images")
# 参考分割需要单独下载（README: https://bit.ly/rim-one-dl-reference-segmentations）。
# 常见落盘位置，按顺序探测；也可用 --rimone-dl-seg-dir 指定。
RIMONE_DL_SEG_CANDIDATES = (
    os.path.join(RAW, "RIM-ONE_DL", "RIM-ONE_DL_reference_segmentations"),
    os.path.join(RAW, "RIM-ONE_DL_reference_segmentations"),
    os.path.join(RAW, "RIM-ONE_DL", "reference_segmentations"),
)


def _find_rimone_dl_seg_dir(explicit=None):
    if explicit:
        return explicit if os.path.isdir(explicit) else None
    for c in RIMONE_DL_SEG_CANDIDATES:
        if os.path.isdir(c):
            return c
    return None


def convert_rimone_dl(partition="randomly", seg_dir=None,
                      mode="crop", roi_size=800, out_names=("RIM_ONE_r3_train",
                                                             "RIM_ONE_r3_test")):
    """用 RIM-ONE DL 替换域 A（默认写出与 r3 相同的文件名，便于复用现有配置）。

    与 r3 的差别：
      * DL 是**已裁好的单眼方形图**（485 张，非黑占比 1.000），没有立体图问题；
      * DL 自带**官方划分**，两套变体：
          partitioned_randomly     339 训练 / 146 测试（论文说"随机 8:2"，用这套）
          partitioned_by_hospital  311 训练 / 174 测试（按医院划分，更难）
      * 参考分割是**独立下载**的：<名>-1-Disc-T.png / <名>-1-Cup-T.png（专家1）。

    掩膜命名与 r3 的 `-1-` 专家 1 一致，二值 0/255。
    """
    if not os.path.isdir(RIMONE_DL_IMG):
        print("[skip] RIM-ONE_DL 图像目录不存在"); return
    base = os.path.join(RIMONE_DL_IMG, "partitioned_{}".format(partition))
    if not os.path.isdir(base):
        print("  [ERROR] 找不到划分目录 {}（可选 randomly / by_hospital）".format(base)); return

    seg = _find_rimone_dl_seg_dir(seg_dir)
    if seg is None:
        print("  [ERROR] 找不到 RIM-ONE DL 参考分割目录。")
        print("          它需要**单独下载**：https://bit.ly/rim-one-dl-reference-segmentations")
        print("          下载后解压到以下任一位置即可被自动识别：")
        for c in RIMONE_DL_SEG_CANDIDATES:
            print("            " + os.path.relpath(c, REPO))
        print("          或用 --rimone-dl-seg-dir <路径> 指定。")
        return
    print("  参考分割目录: {}".format(os.path.relpath(seg, REPO)))

    for split, tag in (("training_set", "train"), ("test_set", "test")):
        b = CocoBuilder("RIM_ONE_r3", tag, mode=mode, roi_size=roi_size)
        n_missing = 0
        for cls in ("glaucoma", "normal"):
            for img_path in listdir_clean(os.path.join(base, split, cls), "*.png"):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                disc_p = os.path.join(seg, cls, stem + "-1-Disc-T.png")
                cup_p = os.path.join(seg, cls, stem + "-1-Cup-T.png")
                if not (os.path.exists(disc_p) and os.path.exists(cup_p)):
                    n_missing += 1
                    continue
                disc = np.array(Image.open(disc_p)) > 0
                cup = np.array(Image.open(cup_p)) > 0
                b.add(img_path, {DISC_ID: disc, CUP_ID: cup})
        report("RIM-ONE DL", tag, b, b.write())
        if n_missing:
            print("      ^ 警告：{} 张图缺掩膜，已跳过（检查分割目录结构）".format(n_missing))
    print("      ^ 已用 RIM-ONE DL（partitioned_{}）覆盖写出 {} / {}".format(
        partition, out_names[0], out_names[1]))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*",
                    default=["REFUGE", "Drishti_GS", "RIM_ONE_r3", "ORIGA"],
                    choices=["REFUGE", "Drishti_GS", "RIM_ONE_r3", "ORIGA"])
    ap.add_argument("--rimone-expert", default="avg", choices=["avg", "exp1", "exp2"])
    ap.add_argument("--rimone-test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260323)
    ap.add_argument("--mode", default="crop", choices=["crop", "link"],
                    help="crop=按论文裁 ROI 并缩放到 roi-size（默认）；link=只建符号链接")
    ap.add_argument("--roi-size", type=int, default=800)
    ap.add_argument("--origa-test-frac", type=float, default=0.2)
    ap.add_argument("--rimone-source", default="r3", choices=["r3", "dl"],
                    help="域 A 用哪一版 RIM-ONE：r3（默认，立体图切半）或 dl（官方 485 张单眼图）")
    ap.add_argument("--rimone-dl-partition", default="randomly",
                    choices=["randomly", "by_hospital"])
    ap.add_argument("--rimone-dl-seg-dir", default=None)
    ap.add_argument("--drishti-threshold", type=int, default=128,
                    help="Drishti SoftMap 二值化阈值：128=过半专家同意（默认），255=全体同意")
    ap.add_argument("--no-rimone-split-stereo", dest="rimone_split_stereo",
                    action="store_false", default=True,
                    help="关闭 RIM-ONE 立体图切半（默认开启）")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("raw      :", os.path.relpath(RAW, REPO))
    print("output   :", os.path.relpath(OUT, REPO))
    print()

    if "REFUGE" in args.datasets:
        convert_refuge(args.mode, args.roi_size)
    if "Drishti_GS" in args.datasets:
        convert_drishti(args.mode, args.roi_size, args.drishti_threshold)
    if "RIM_ONE_r3" in args.datasets:
        if args.rimone_source == "dl":
            convert_rimone_dl(args.rimone_dl_partition, args.rimone_dl_seg_dir,
                              args.mode, args.roi_size)
        else:
            convert_rimone(args.rimone_expert, args.rimone_test_frac, args.seed,
                           args.mode, args.roi_size, args.rimone_split_stereo)
    if "ORIGA" in args.datasets:
        convert_origa(args.origa_test_frac, args.seed, args.mode, args.roi_size)

    print("\n完成。可用以下命令检查注册结果：")
    print("  .venv/bin/python -c \"import data.datasets.builtin as b;"
          "from detectron2.data import DatasetCatalog as D;"
          "print([n for n in D.list() if 'REFUGE' in n or 'Drishti' in n or 'RIM' in n])\"")


if __name__ == "__main__":
    sys.exit(main())
