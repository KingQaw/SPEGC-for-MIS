#!/usr/bin/env python3
"""
汇总 tools/run_loo_per_class.sh 产出的逐模型按类 Dice，按**测试域**聚合成
论文 Table 1 的列结构（每列 = 在其余四个源模型上的均值）。

用法：
    .venv/bin/python tools/summarize_loo_per_class.py --input .cache/loo_per_class.json
"""

import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOMAIN_MEMBERS = {
    "A": ["RIM_ONE_r3_train", "RIM_ONE_r3_test"],
    "B": ["REFUGE_train"],                      # REFUGE_test 无标注，排除
    "C": ["ORIGA_train", "ORIGA_test"],
    "D": ["REFUGE_Valid"],
    "E": ["Drishti_GS_train", "Drishti_GS_test"],
}
DOMAIN_LABEL = {"A": "RIM-ONE", "B": "REFUGE", "C": "ORIGA",
                "D": "REFUGE-Test", "E": "Drishti-GS"}
# 论文 Table 1 SPEGC 行的 DSC
PAPER = {"A": 84.90, "B": 83.34, "C": 84.57, "D": 83.54, "E": 85.51}


def load(path):
    """读取每行一个 JSON 对象（或一个 JSON 数组）的文件。"""
    raw = open(path).read().strip()
    if not raw:
        return []
    try:
        d = json.loads(raw)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        out = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(REPO, ".cache/loo_per_class.json"))
    args = ap.parse_args()

    runs = load(args.input)
    if not runs:
        print(f"读不到结果：{args.input}\n先跑 bash tools/run_loo_per_class.sh")
        return 1

    # 从权重文件名推断源域：.../model_C.pth -> C
    per_run = {}
    for r in runs:
        stem = os.path.basename(r.get("weights", "")).replace(".pth", "")
        src = stem.split("_")[-1][:1].upper()
        per_run[src] = r.get("per_dataset", {})

    print("=" * 96)
    print("按类 Dice（两类均值）逐次运行 —— 行 = 源模型，列 = 测试数据集")
    print("=" * 96)
    datasets = sorted({d for r in per_run.values() for d in r})
    print(f"{'源':<5}" + "".join(f"{d[:19]:>21}" for d in datasets))
    for k in sorted(per_run):
        print(f"{k:<5}" + "".join(
            f"{per_run[k][d]['mean2']:>21.2f}" if d in per_run[k] else f"{'-':>21}"
            for d in datasets))

    print()
    print("=" * 96)
    print("留一法汇总（两类均值 DSC；列 = 测试域，值 = 其余四个源模型的均值）")
    print("=" * 96)
    header = f"{'源模型':<10}" + "".join(f"{DOMAIN_LABEL[d]:>14}" for d in "ABCDE")
    print(header)
    print("-" * len(header))

    domain_vals = {d: [] for d in "ABCDE"}
    for src in sorted(per_run):
        row = f"model_{src:<4}"
        for dom in "ABCDE":
            vals = [per_run[src][m]["mean2"] for m in DOMAIN_MEMBERS[dom] if m in per_run[src]]
            if vals:
                v = statistics.mean(vals)
                row += f"{v:>14.2f}"
                if src not in DOMAIN_MEMBERS[dom]:
                    domain_vals[dom].append(v)
            else:
                row += f"{'-':>14}"
        print(row)

    print("-" * len(header))
    row = f"{'留一平均':<9}"
    for d in "ABCDE":
        row += f"{statistics.mean(domain_vals[d]):>14.2f}" if domain_vals[d] else f"{'n/a':>14}"
    print(row)
    row = f"{'论文':<11}"
    for d in "ABCDE":
        row += f"{PAPER[d]:>14.2f}"
    print(row)
    row = f"{'差值':<11}"
    for d in "ABCDE":
        row += (f"{statistics.mean(domain_vals[d]) - PAPER[d]:>+14.2f}"
                if domain_vals[d] else f"{'n/a':>14}")
    print(row)

    ours = [statistics.mean(domain_vals[d]) for d in "ABCDE" if domain_vals[d]]
    print()
    print(f"可算域平均: 本项目 {statistics.mean(ours):.2f}  vs  论文 "
          f"{statistics.mean(PAPER[d] for d in 'ABCDE' if domain_vals[d]):.2f}")
    print()
    print("说明：本表用的是标准「每图每类取最佳匹配、漏检记 0」的两类均值，")
    print("      与仓库 DiceEvaluator（按预测平均、被视盘主导）口径不同。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
