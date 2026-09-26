#!/usr/bin/env bash
#
# 自训某个域的 fundus 源模型（用于替代作者发布的欠训权重）。
#
# 背景：作者发布的 weights/fundus_source/*.pth 只训练了 999~2999 步，
# 视杯分支从未收敛（预测置信度全 <0.5）。自训 2000 步即可让视杯 Dice
# 从 0.00 升到 87.22（见 docs/FUNDUS_DATA.md 第 5 节）。
#
# 用法：
#   bash tools/train_sources.sh A                 # 训域 A（RIM-ONE），默认 GPU 0
#   GPU=1 bash tools/train_sources.sh B           # 指定卡
#   MAX_ITER=6000 bash tools/train_sources.sh C
#   DOMAINS="A B C D E" bash tools/train_sources.sh   # 顺序训多个
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
GPU="${GPU:-0}"
MAX_ITER="${MAX_ITER:-10000}"
CONFIG="${CONFIG:-configs/train_fundus_source.yaml}"
CKPT="${CKPT:-.cache/torch/R-50.pkl}"

# 域 -> 训练/测试数据集（论文 4.1；域 D 用的是仓库里的 REFUGE_Valid）
declare -A TRAIN SET
TRAIN[A]='("RIM_ONE_r3_train",)'          ; SET[A]='("RIM_ONE_r3_test",)'
TRAIN[B]='("REFUGE_train",)'              ; SET[B]='("REFUGE_Valid",)'
TRAIN[C]='("ORIGA_train",)'               ; SET[C]='("ORIGA_test",)'
TRAIN[D]='("REFUGE_Valid",)'              ; SET[D]='("REFUGE_train",)'
TRAIN[E]='("Drishti_GS_train",)'          ; SET[E]='("Drishti_GS_test",)'

[ -f "$CKPT" ] || { echo "ERROR: 缺 ImageNet 预训练权重 $CKPT"; exit 1; }

for K in ${DOMAINS:-${1:-}}; do
    [ -n "${TRAIN[$K]:-}" ] || { echo "[skip] 未知域 $K"; continue; }
    OUT="output/src_$K"
    if [ -f "$OUT/model_final.pth" ]; then
        echo "[skip] $K 已完成（$OUT/model_final.pth 存在）"; continue
    fi
    echo "################ 训练域 $K  (GPU $GPU, MAX_ITER $MAX_ITER) ################"
    echo "  TRAIN=${TRAIN[$K]}  TEST=${SET[$K]}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/mpl}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_net.py --num-gpus 1 \
        --config "$CONFIG" \
        MODEL.WEIGHTS "$CKPT" \
        OUTPUT_DIR "$OUT" \
        DATASETS.TRAIN "${TRAIN[$K]}" \
        DATASETS.TEST "${SET[$K]}" \
        SOLVER.MAX_ITER "$MAX_ITER" \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        DATALOADER.NUM_WORKERS 2 \
        > ".cache/train_src_$K.log" 2>&1
    echo "  完成，退出码 $? （日志 .cache/train_src_$K.log）"
done
