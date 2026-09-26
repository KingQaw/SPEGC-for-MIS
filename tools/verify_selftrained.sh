#!/usr/bin/env bash
#
# 验证自训的 5 个源模型在**自己的域**上的两类 Dice，确认视杯在每个域都学得会
# （作者发布的权重视杯 Dice 恒为 0，见 docs/FUNDUS_DATA.md 第 5 节）。
#
# 用法：
#   bash tools/verify_selftrained.sh            # 用 output/selftrained/
#   WD=weights/fundus_source bash tools/verify_selftrained.sh   # 对照作者权重
#   LIMIT=40 GPU=0 bash tools/verify_selftrained.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/mpl}"
export PYTHONPATH="$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

WD="${WD:-output/selftrained}"
LIMIT="${LIMIT:-40}"

# 域 -> 自己在训练时留出的测试集
declare -A TEST
TEST[A]="RIM_ONE_r3_test"
TEST[B]="REFUGE_Valid"
TEST[C]="ORIGA_test"
TEST[D]="REFUGE_train"          # 域 D 用 REFUGE_Valid 训练，故在 REFUGE_train 上测
TEST[E]="Drishti_GS_test"

printf "%-10s %-18s %10s %10s %10s\n" "权重" "测试集" "视杯Dice" "视盘Dice" "两类均值"
printf -- "------------------------------------------------------------------------\n"
for K in A B C D E; do
    W="$WD/model_$K.pth"
    DS="${TEST[$K]}"
    if [ ! -f "$W" ]; then
        printf "%-10s %-18s %10s\n" "model_$K" "$DS" "(缺权重)"
        continue
    fi
    OUT="$("$PY" tools/eval_per_class.py --weights "$W" --datasets "$DS" \
            --limit "$LIMIT" --thresholds 0.9 --json 2>/dev/null)"
    echo "$OUT" | "$PY" -c "
import sys, json
txt = sys.stdin.read()
i = txt.find('--- JSON ---')
rec = json.loads(txt[i+12:].strip()) if i >= 0 else None
if not rec:
    print(f'{\"model_$K\":<10}{\"$DS\":<18}{\"(解析失败)\"}'); raise SystemExit
ds = list(rec['per_dataset'].values())[0]
print(f'{\"model_$K\":<10}{\"$DS\":<18}{ds[\"cup\"]:>10.2f}{ds[\"disc\"]:>10.2f}{ds[\"mean2\"]:>10.2f}')
"
done
