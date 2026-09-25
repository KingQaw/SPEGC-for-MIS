#!/usr/bin/env python3
"""
汇总留一法（leave-one-out）评测结果，复现论文 Table 1 的列结构。

读取 output/loo_<A..E>/result.txt，按**测试域**聚合：论文 Table 1 的每一列
是「在该域上测试、源模型来自其余四个域」的均值（论文称 "based on five
experimental runs"）。

用法：
    .venv/bin/python tools/summarize_leave_one_out.py
    .venv/bin/python tools/summarize_leave_one_out.py --root output
"""

import argparse
import os
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 域 -> 该域包含的数据集名（论文 4.1 节；D 用的是仓库里的 REFUGE_Valid）
DOMAIN_MEMBERS = {
    "A": ["RIM_ONE_r3_train", "RIM_ONE_r3_test"],
    "B": ["REFUGE_train"],
    "C": ["ORIGA_train", "ORIGA_test"],
    "D": ["REFUGE_Valid"],
    "E": ["Drishti_GS_train", "Drishti_GS_test"],
}
DOMAIN_LABEL = {
    "A": "RIM-ONE",
    "B": "REFUGE",
    "C": "ORIGA",
    "D": "REFUGE-Test",
    "E": "Drishti-GS",
}

# 论文 Table 1 的 SPEGC 行（DSC, E_phi^max, S_alpha），取自 CVPR 2026 正文
PAPER_SPEGC = {
    "A": (84.90, 93.81, 88.50),
    "B": (83.34, 93.01, 86.74),
    "C": (84.57, 93.71, 88.42),
    "D": (83.54, 93.21, 85.42),
    "E": (85.51, 93.54, 88.92),
}
PAPER_AVG = (84.37, 94.56, 87.60)

ROW_RE = re.compile(
    r"^\s*(\S+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*$"
)


def parse_result(path):
    """返回 {dataset_name: (dice, ea, sm)}，跳过 *_mean 汇总行。"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            m = ROW_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            if name.endswith("_mean"):
                continue
            out[name] = tuple(float(m.group(i)) for i in (2, 3, 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "output"))
    ap.add_argument("--pattern", default="loo_{}")
    args = ap.parse_args()

    # collected[domain][source_config] = [ (dice,ea,sm), ... ]  —— 取该域所有成员数据集的均值
    per_run = defaultdict(dict)          # config -> {dataset: metrics}
    for k in "ABCDE":
        path = os.path.join(args.root, args.pattern.format(k), "result.txt")
        res = parse_result(path)
        if res:
            per_run[k] = res
        else:
            print(f"[warn] 读不到 {path}", file=sys.stderr)

    if not per_run:
        print("没有任何结果，先跑 tools/run_leave_one_out.sh")
        return 1

    print("=" * 100)
    print("各次运行（行 = 源模型配置，列 = 测试数据集，Dice %）")
    print("=" * 100)
    datasets = sorted({d for r in per_run.values() for d in r})
    print(f"{'源':<4}" + "".join(f"{d[:17]:>19}" for d in datasets))
    for k in sorted(per_run):
        row = f"{k:<4}"
        for d in datasets:
            v = per_run[k].get(d)
            row += f"{v[0]:>19.2f}" if v else f"{'-':>19}"
        print(row)

    # 按域做留一平均
    print()
    print("=" * 100)
    print("留一法汇总（列 = 测试域；值 = 在其余四个源模型上的均值，仅 Dice）")
    print("=" * 100)
    header = f"{'源模型':<10}" + "".join(f"{DOMAIN_LABEL[d]:>15}" for d in "ABCDE")
    print(header)
    print("-" * len(header))

    domain_vals = defaultdict(list)
    for src in sorted(per_run):
        row = f"model_{src:<5}"
        for dom in "ABCDE":
            vals = [per_run[src][m] for m in DOMAIN_MEMBERS[dom] if m in per_run[src]]
            if vals:
                dice = sum(v[0] for v in vals) / len(vals)
                row += f"{dice:>15.2f}"
                if src not in DOMAIN_MEMBERS[dom]:      # 留一：排除源域自身
                    domain_vals[dom].append(dice)
            else:
                row += f"{'-':>15}"
        print(row)

    print("-" * len(header))
    row = f"{'留一平均':<9}"
    for dom in "ABCDE":
        vs = domain_vals[dom]
        row += f"{sum(vs)/len(vs):>15.2f}" if vs else f"{'n/a':>15}"
    print(row)
    row = f"{'论文 Table1':<9}"
    for dom in "ABCDE":
        row += f"{PAPER_SPEGC[dom][0]:>15.2f}"
    print(row)
    print("-" * len(header))
    row = f"{'差值':<10}"
    for dom in "ABCDE":
        vs = domain_vals[dom]
        row += f"{sum(vs)/len(vs) - PAPER_SPEGC[dom][0]:>+15.2f}" if vs else f"{'n/a':>15}"
    print(row)

    print()
    print("说明：")
    print("  * 域 C (ORIGA) 无掩膜，其列恒为 n/a。")
    print("  * 其余四列是「剔除 ORIGA 后」的留一平均，与论文的完整留一不完全可比。")
    print("  * REFUGE_test 无标注（恒 0）已排除在域 B 的成员之外，否则会污染均值。")
    print("  * 仓库 DiceEvaluator 与论文 DSC 口径可能不同，见 docs/FUNDUS_DATA.md。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
