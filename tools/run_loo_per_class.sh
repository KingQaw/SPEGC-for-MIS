#!/usr/bin/env bash
#
# 用**按类 Dice**（视杯/视盘/两类均值）跑留一法，汇总成论文 Table 1 的列结构。
#
# 为什么不用 tools/run_leave_one_out.sh：那个走仓库的 DiceEvaluator，
# 而它的口径是「对每个预测取最佳匹配后求均值」——预测多的类会主导均值。
# 实测作者权重几乎只输出视盘，于是仓库报的是视盘 Dice，视杯塌陷被掩盖
# （详见 tools/eval_per_class.py 的说明与 docs/FUNDUS_DATA.md 第 5 节）。
#
# 本脚本对每个源模型跑一遍，覆盖「除自己之外的全部域」，取两类均值。
#
# 用法：
#   bash tools/run_loo_per_class.sh                      # 用 weights/fundus_source/
#   WD=output/src bash tools/run_loo_per_class.sh        # 用自训权重
#   LIMIT=80 bash tools/run_loo_per_class.sh             # 每个数据集只评前 N 张（快速）
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
WD="${WD:-weights/fundus_source}"
LIMIT="${LIMIT:-0}"                     # 0 = 不限制
OUT="${OUT:-.cache/loo_per_class.json}"
mkdir -p "$(dirname "$OUT")"

# 域 -> 成员数据集（与 test_segment.yaml 的留一结构一致）
declare -A DOM
DOM[A]="RIM_ONE_r3_train RIM_ONE_r3_test"
DOM[B]="REFUGE_train"
DOM[C]="ORIGA_train ORIGA_test"
DOM[D]="REFUGE_Valid"
DOM[E]="Drishti_GS_train Drishti_GS_test"

: > "$OUT"
for K in A B C D E; do
    W="$WD/model_$K.pth"
    if [ ! -f "$W" ]; then echo "[skip] 缺 $W"; continue; fi
    # 测试列表 = 除域 K 之外的全部域
    DS=()
    for X in A B C D E; do
        [ "$X" = "$K" ] && continue
        for d in ${DOM[$X]}; do DS+=("$d"); done
    done
    echo "=== model_$K  ->  ${DS[*]}"
    ARGS=(--weights "$W" --datasets "${DS[@]}" --device "$DEVICE" --thresholds 0.9 --json)
    [ "$LIMIT" != "0" ] && ARGS+=(--limit "$LIMIT")
    "$PY" tools/eval_per_class.py "${ARGS[@]}" 2>/dev/null \
        | sed -n '/--- JSON ---/,$p' | tail -n +2 >> "$OUT"
done

echo
echo "逐模型结果已写入 $OUT"
"$PY" tools/summarize_loo_per_class.py --input "$OUT"
