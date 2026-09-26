#!/usr/bin/env bash
#
# 在**某一个测试域**上评测所有可用的源模型，打印该域的留一列候选值。
#
# 用法：
#   bash tools/eval_domain.sh RIM_ONE_r3_train RIM_ONE_r3_test
#   DOMAIN_LABEL="A (RIM-ONE)" bash tools/eval_domain.sh RIM_ONE_r3_train RIM_ONE_r3_test
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/mpl}"
export PYTHONPATH="$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

WD="${WD:-output/selftrained}"
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || { echo "用法: bash tools/eval_domain.sh <数据集...>"; exit 1; }

echo "域: ${DOMAIN_LABEL:-${DATASETS[*]}}"
printf "%-10s" "源模型"
for d in "${DATASETS[@]}"; do printf "%22s" "$d"; done
printf "%10s\n" "域均值"
printf -- "------------------------------------------------------------------------------\n"

vals=()
for K in A B C D E; do
    W="$WD/model_$K.pth"
    [ -f "$W" ] || continue
    OUT="$("$PY" tools/eval_per_class.py --weights "$W" --datasets "${DATASETS[@]}" \
            --thresholds 0.9 --json 2>/dev/null | sed -n '/--- JSON ---/,$p' | tail -n +2)"
    printf "%-10s" "model_$K"
    echo "$OUT" | "$PY" -c "
import sys, json
rec = json.loads(sys.stdin.read())
ds = rec['per_dataset']
import os
names = list(ds.keys())
vals = [ds[n]['mean2'] for n in names]
for v in vals: print(f'{v:>22.2f}', end='')
print(f'{sum(vals)/len(vals):>10.2f}')
"
done
printf -- "------------------------------------------------------------------------------\n"
