#!/usr/bin/env bash
#
# 用官方预训练权重评测眼底目标域的持续测试时适应（CTTA）性能。
#
# 用法：
#   bash tools/run_fundus_eval.sh                       # 默认 GPU + 模型B + 全部可用域
#   DEVICE=cpu bash tools/run_fundus_eval.sh            # 无 GPU 环境
#   MODEL=C bash tools/run_fundus_eval.sh               # 换权重
#   TTT_STEPS=4 bash tools/run_fundus_eval.sh           # 只做少量 TTT 步（快速冒烟）
#   DOMAINS='("Drishti_GS_test",)' bash tools/run_fundus_eval.sh
#
# 前置条件：
#   1) bash tools/setup_env.sh                       建好 .venv
#   2) python tools/prepare_fundus_data.py           把 datasets/raw 转成 COCO
#   3) weights/fundus_source/model_<A..E>.pth        官方权重
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/mpl}"

MODEL="${MODEL:-B}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-configs/test_fundus_local.yaml}"
# 留空 = 用 CONFIG 里自己写的 DATASETS.TEST（推荐，配置优先）。
# 只有显式设置 DOMAINS 时才覆盖，例如：
#   DOMAINS='("Drishti_GS_test",)' bash tools/run_fundus_eval.sh
DOMAINS="${DOMAINS:-}"
# None = 跑完整个目标流；设成整数则每个域只做这么多 TTT 步
TTT_STEPS="${TTT_STEPS:-None}"

WEIGHTS="${WEIGHTS:-weights/fundus_source/model_${MODEL}.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-output/test_fundus_${MODEL}_${DEVICE}}"

[ -x "$PY" ]        || { echo "ERROR: 找不到解释器 $PY，先跑 bash tools/setup_env.sh"; exit 1; }
[ -f "$WEIGHTS" ]   || { echo "ERROR: 找不到权重 $WEIGHTS"; exit 1; }

# 组装覆盖项：DATASETS.TEST 仅在 DOMAINS 非空时追加，否则交给配置文件
OPTS=(MODEL.WEIGHTS "$WEIGHTS" MODEL.DEVICE "$DEVICE"
      TEST.MIN_BATCH_NUM "$TTT_STEPS" OUTPUT_DIR "$OUTPUT_DIR")
if [ -n "$DOMAINS" ]; then
    OPTS+=(DATASETS.TEST "$DOMAINS")
fi

echo "配置    : $CONFIG"
echo "权重    : $WEIGHTS"
echo "设备    : $DEVICE"
echo "目标域  : ${DOMAINS:-<取自配置文件>}"
echo "TTT步数 : $TTT_STEPS"
echo "输出    : $OUTPUT_DIR"
echo

# train_net.py 强制 resume=True：残留的 last_checkpoint 会覆盖 MODEL.WEIGHTS，
# 所以每次评测都用干净目录。
rm -rf "$OUTPUT_DIR"

"$PY" train_net.py \
    --eval-only --num-gpus 1 \
    --config "$CONFIG" \
    "${OPTS[@]}" \
    2>&1 | grep -vE "UserWarning|floor_divide|torch\.div|return torch|Triggered internally"

echo
echo "============ 结果 ($OUTPUT_DIR/result.txt) ============"
cat "$OUTPUT_DIR/result.txt" 2>/dev/null || echo "(未生成 result.txt)"
