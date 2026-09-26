#!/usr/bin/env bash
#
# 对比 Drishti-GS SoftMap 两种二值化阈值对**域 E 列**的影响。
#
# 关键点：留一法里域 E 那一列是 A/B/C/D 四个源模型的平均，这四个都不在
# Drishti 上训练，所以换测试集阈值**不需要重训**就能看出影响。
#
# 用法：
#   bash tools/eval_drishti_thresh.sh              # 用 output/selftrained/
#   WD=weights/fundus_source bash tools/eval_drishti_thresh.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/mpl}"
export PYTHONPATH="$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

WD="${WD:-output/selftrained}"
LIMIT="${LIMIT:-0}"

for K in A B C D; do
    W="$WD/model_$K.pth"
    [ -f "$W" ] || { echo "### model_$K  (缺权重)"; continue; }
    echo "### model_$K"
    ARGS=(--weights "$W" --datasets Drishti_GS_train Drishti_GS_test --thresholds 0.9 --json)
    [ "$LIMIT" != "0" ] && ARGS+=(--limit "$LIMIT")
    "$PY" tools/eval_per_class.py "${ARGS[@]}" 2>/dev/null \
        | sed -n '/--- JSON ---/,$p' | tail -n +2
done
