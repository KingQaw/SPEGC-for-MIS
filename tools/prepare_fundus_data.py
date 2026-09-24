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

DISC_ID, CUP_ID = 1, 2
CATEGORIES = [
    {"id": DISC_ID, "name": "optic_disc"},
    {"id": CUP_ID, "name": "optic_cup"},
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


class CocoBuilder:
    """累积一个 COCO 数据集，并顺带建立图像符号链接。"""

    def __init__(self, name, split):
        self.name = name
        self.split = split
        self.image_root = os.path.join(OUT, name)
        os.makedirs(self.image_root, exist_ok=True)
        self.images = []
        self.annotations = []
        self._img_id = 0
        self._ann_id = 0
        self.skipped = 0

    def add(self, src_image, masks):
        """masks: {category_id: 二值掩膜 ndarray}，形状须与图像一致。"""
        rel = os.path.basename(src_image)
        with Image.open(src_image) as im:
            width, height = im.size

        link = os.path.join(self.image_root, rel)
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(os.path.abspath(src_image), self.image_root), link)

        self._img_id += 1
        self.images.append(
            {"file_name": rel, "height": height, "width": width, "id": self._img_id}
        )

        for cat_id, mask in masks.items():
            if mask is None:
                continue
            if mask.shape != (height, width):
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (width, height), Image.NEAREST
                    )
                ) > 0
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
def convert_refuge():
    root = os.path.join(RAW, "REFUGE", "REFUGE")
    if not os.path.isdir(root):
        print("[skip] REFUGE not found"); return

    for split, tag in (("train", "train"), ("val", "Valid")):
        b = CocoBuilder("REFUGE", tag)
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
    b = CocoBuilder("REFUGE", "test")
    for img_path in listdir_clean(os.path.join(root, "test", "Images"), "*.jpg"):
        b.add(img_path, {})
    report("REFUGE", "test", b, b.write())
    print("      ^ 注意：REFUGE test 的 400 张图**没有官方标注**，该 json 只有图像，"
          "在其上算 Dice 没有意义（见 docs/FUNDUS_DATA.md）")


# --------------------------------------------------------------------------- #
# Drishti-GS:  SoftMap 软图，阈值 128；掩膜尺寸比图像小，需放大
# --------------------------------------------------------------------------- #
def convert_drishti():
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
        b = CocoBuilder("Drishti_GS", split)
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
            od = np.array(Image.open(od_p).resize((w, h), Image.NEAREST)) >= 128
            cup = np.array(Image.open(cup_p).resize((w, h), Image.NEAREST)) >= 128
            b.add(img_path, {DISC_ID: od, CUP_ID: cup})
        report("Drishti_GS", split, b, b.write())


# --------------------------------------------------------------------------- #
# RIM-ONE r3:  无官方划分 -> 固定种子分层划分
# --------------------------------------------------------------------------- #
def convert_rimone(expert="avg", test_frac=0.3, seed=20260323):
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

    # 分层划分：每个类别内部按固定种子打乱
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in ("healthy", "glaucoma"):
        group = sorted([it for it in items if it[3] == label], key=lambda x: x[0])
        idx = rng.permutation(len(group))
        n_test = int(round(len(group) * test_frac))
        test_idx = set(idx[:n_test].tolist())
        for i, it in enumerate(group):
            (test if i in test_idx else train).append(it)

    for split, group in (("train", train), ("test", test)):
        b = CocoBuilder("RIM_ONE_r3", split)
        for img_path, disc_p, cup_p, _ in group:
            disc = np.array(Image.open(disc_p)) > 0
            cup = np.array(Image.open(cup_p)) > 0
            b.add(img_path, {DISC_ID: disc, CUP_ID: cup})
        report("RIM_ONE_r3", split, b, b.write())
    print("      ^ 该数据集无官方划分；以上为 stratified {:.0f}/{:.0f} 划分 "
          "(expert={}, seed={})".format(
              (1 - test_frac) * 100, test_frac * 100, expert, seed))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*",
                    default=["REFUGE", "Drishti_GS", "RIM_ONE_r3"],
                    choices=["REFUGE", "Drishti_GS", "RIM_ONE_r3", "ORIGA"])
    ap.add_argument("--rimone-expert", default="avg", choices=["avg", "exp1", "exp2"])
    ap.add_argument("--rimone-test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260323)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("raw      :", os.path.relpath(RAW, REPO))
    print("output   :", os.path.relpath(OUT, REPO))
    print()

    if "REFUGE" in args.datasets:
        convert_refuge()
    if "Drishti_GS" in args.datasets:
        convert_drishti()
    if "RIM_ONE_r3" in args.datasets:
        convert_rimone(args.rimone_expert, args.rimone_test_frac, args.seed)
    if "ORIGA" in args.datasets:
        print("[skip] ORIGA: 本下载只有 jpg + 青光眼分类 csv，没有 OD/OC 分割掩膜，"
              "无法生成 COCO 标注。")

    print("\n完成。可用以下命令检查注册结果：")
    print("  .venv/bin/python -c \"import data.datasets.builtin as b;"
          "from detectron2.data import DatasetCatalog as D;"
          "print([n for n in D.list() if 'REFUGE' in n or 'Drishti' in n or 'RIM' in n])\"")


if __name__ == "__main__":
    sys.exit(main())
