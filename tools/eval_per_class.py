#!/usr/bin/env python3
"""
按类别计算 Dice（视杯 / 视盘 / 两类均值），补上仓库 `DiceEvaluator` 看不到的信息。

为什么需要这个工具
------------------
`evaluation/dice_metric.py:49-77` 的口径是「**对每个预测实例**取最佳匹配后求均值」：

    for pred_class, pred_mask in zip(pred_classes, pred_masks):
        best = max(dice(pred_mask, gt) for gt in 同类 GT)     # 没有同类 GT 就是 0
        self.dice_scores.append(best * 100)

于是**预测多的那个类会主导均值**。实测作者发布的
`weights/fundus_source/*.pth` 几乎只输出视盘（视杯预测的置信度全都 <0.5），
所以仓库报出来的 ~85% **实际上只是视盘 Dice**——视杯的 Dice 接近 0 却看不出来。

OD/OC 分割论文报的 DSC 通常是**视盘/视杯两类的平均**，本工具按这个口径算：

    per image per class: 取同类预测中 IoU 最高的那个的 Dice；漏检记 0
    然后对全数据集求均值，最后给出两类均值

用法
----
    .venv/bin/python tools/eval_per_class.py \
        --config configs/test_fundus_local.yaml \
        --weights weights/fundus_source/model_C.pth \
        --datasets ORIGA_test REFUGE_Valid Drishti_GS_test \
        --limit 50
"""

import argparse
import os
import statistics
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from detectron2.config import get_cfg                       # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer     # noqa: E402
from detectron2.data import DatasetCatalog                  # noqa: E402
from pycocotools import mask as mu                          # noqa: E402

from config import add_spegc_config                          # noqa: E402
from engine.trainer import BaselineTrainer                   # noqa: E402
from modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN  # noqa
from modeling.proposal_generator.rpn import PseudoLabRPN                  # noqa
from modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab        # noqa
import data.datasets.builtin  # noqa: F401,E402  (注册数据集)


def to_bin(seg, h, w):
    d = mu.decode(mu.merge(mu.frPyObjects(seg, h, w)))
    return (d[:, :, 0] if d.ndim == 3 else d).astype(bool)


def dice(a, b):
    return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum() + 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/test_fundus_local.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=50, help="每个数据集最多评估多少张")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.05, 0.5, 0.9],
                    help="逐个分数阈值报告（仓库 DiceEvaluator 用 TEST.DICE_THRES，默认 0.9）")
    ap.add_argument("--ttt", action="store_true",
                    help="推理前先做测试时适应（SPEGC 的 L_G + lambda*L_C），"
                         "复刻 engine/trainer.py 的 TTT 循环")
    ap.add_argument("--ttt-lr", type=float, default=None,
                    help="适应阶段的学习率；不指定则用 config 的 SOLVER.BASE_LR")
    ap.add_argument("--ttt-steps", type=int, default=None,
                    help="每个域最多适应多少步；默认 None = 跑满整个流")
    ap.add_argument("--json", action="store_true",
                    help="额外以 JSON 输出结果（供留一法驱动脚本汇总）")
    args = ap.parse_args()

    cfg = get_cfg()
    add_spegc_config(cfg)
    cfg.merge_from_file(os.path.join(REPO, args.config))
    cfg.merge_from_list(["MODEL.WEIGHTS", os.path.abspath(args.weights),
                         "MODEL.DEVICE", args.device])
    cfg.freeze()

    def build_and_load():
        m = BaselineTrainer.build_model(cfg)
        DetectionCheckpointer(m, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False)
        return m

    model = build_and_load()
    model.eval()

    # 类别名取自 metadata（JSON 的 categories，按 id 升序 → 类 0/类 1）
    print(f"权重: {os.path.relpath(cfg.MODEL.WEIGHTS, REPO)}")
    print(f"设备: {args.device}   每集上限: {args.limit} 张\n")
    hdr = f"{'数据集':<18}{'类名':<14}{'预测数':>8}{'分数中位':>10}" + \
          "".join(f"{'>='+str(t):>12}" for t in args.thresholds)
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for ds in args.datasets:
        names = DatasetCatalog.get(ds) and None  # touch to load (populates thing_classes)
        from detectron2.data import MetadataCatalog
        classes = list(MetadataCatalog.get(ds).thing_classes)
        if args.ttt:
            # 每个域独立适应：重置权重再在该域上跑 TTT。
            # 注意这与仓库 test() 的"连续流"不同——那里一个域适应完的状态会
            # 带到下一个域；这里刻意隔离，才能干净地量化适应本身的增益。
            if args.ttt_lr is not None:
                cfg.defrost(); cfg.SOLVER.BASE_LR = args.ttt_lr; cfg.freeze()
            model = build_and_load()
            model.train()
            opt = BaselineTrainer.build_optimizer(cfg, model)
            ttt_loader = BaselineTrainer.build_test_loader(cfg, ds)
            n_ok = n_none = 0
            for idx, inputs in enumerate(ttt_loader):
                if args.ttt_steps is not None and idx >= args.ttt_steps:
                    break
                loss, _, _, _ = model(inputs, branch="TTT")
                if loss is None:
                    n_none += 1
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
                n_ok += 1
            print(f"  [TTT] {ds}: 有效适应步 {n_ok}，图池未满跳过 {n_none}"
                  f"  (lr={args.ttt_lr if args.ttt_lr is not None else cfg.SOLVER.BASE_LR})")
            model.eval()
            del opt, ttt_loader

        loader = BaselineTrainer.build_test_loader(cfg, ds)
        gts = {r["image_id"]: r for r in DatasetCatalog.get(ds)}
        scores = {0: [], 1: []}
        acc = {t: {0: [], 1: []} for t in args.thresholds}
        for i, inputs in enumerate(loader):
            if i >= args.limit:
                break
            with torch.no_grad():
                out = model(inputs)
            inst = out[0]["instances"]
            pm = inst.pred_masks.cpu().numpy()
            pc = inst.pred_classes.cpu().numpy()
            ps = inst.scores.cpu().numpy()
            for k in range(len(pc)):
                scores.setdefault(int(pc[k]), []).append(float(ps[k]))
            rec = gts[inputs[0]["image_id"]]
            h, w = rec["height"], rec["width"]
            gt = [(a["category_id"], to_bin(a["segmentation"], h, w))
                  for a in rec.get("annotations", [])]
            for t in args.thresholds:
                keep = ps >= t
                for gc, gm in gt:
                    best = max([dice(m, gm) for m, c in zip(pm[keep], pc[keep]) if c == gc],
                               default=0.0)
                    acc[t].setdefault(gc, []).append(best * 100)
        for c in (0, 1):
            nm = classes[c] if c < len(classes) else f"类{c}"
            s = np.array(scores.get(c) or [0.0])
            row = f"{ds:<18}{nm:<14}{len(scores.get(c, [])):>8}{np.median(s):>10.3f}"
            for t in args.thresholds:
                v = acc[t].get(c) or [0.0]
                row += f"{statistics.mean(v):>12.2f}"
            print(row)
        for t in args.thresholds:
            vals = [statistics.mean(acc[t].get(c) or [0.0]) for c in (0, 1)]
            summary.setdefault(t, []).append((ds, vals[0], vals[1], sum(vals) / 2))
        print()

    print("=" * 78)
    print("两类均值（OD/OC 分割论文通常报这个口径）")
    print("=" * 78)
    print(f"{'数据集':<18}" + "".join(f"{'DSC@'+str(t):>16}" for t in args.thresholds))
    for i, ds in enumerate(args.datasets):
        row = f"{ds:<18}"
        for t in args.thresholds:
            row += f"{summary[t][i][3]:>16.2f}"
        print(row)
    print("-" * 78)
    row = f"{'各阈值平均':<18}"
    for t in args.thresholds:
        row += f"{statistics.mean(s[3] for s in summary[t]):>16.2f}"
    print(row)

    if args.json:
        import json
        payload = {
            "weights": os.path.relpath(cfg.MODEL.WEIGHTS, REPO),
            "per_dataset": {
                ds: {
                    "cup": summary[t][i][1],
                    "disc": summary[t][i][2],
                    "mean2": summary[t][i][3],
                    "threshold": t,
                }
                for t in args.thresholds
                for i, ds in enumerate(args.datasets)
                if t == args.thresholds[-1]      # 默认取最后一个阈值（与仓库 DICE_THRES 对齐）
            },
            "thresholds": args.thresholds,
        }
        print("\n--- JSON ---")
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
