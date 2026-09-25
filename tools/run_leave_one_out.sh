#!/usr/bin/env bash
#
# 复现论文 Table 1 的留一法（leave-one-out）设置。
#
# 论文 4.1/4.2 节：
#   Domain A = RIM-ONE, B = REFUGE, C = ORIGA, D = REFUGE-Test, E = Drishti-GS
#   "Domain X" = 在 Domain X 上训练源模型、在**其余四个域**上测试；
#   Table 1 每一列是该测试域在 4 个源模型上的均值（"based on five experimental runs"）。
#
# 所以配置 K 用 weights/fundus_source/model_K.pth，测试列表 = 除域 K 之外的全部域。
#
# 五个域的数据集本轮已全部齐备（含 ORIGA-masked），下面就是
# configs/test_segment.yaml 第 5/7/9/11/13 行的**原始列表，未做任何删减**。
#
# 用法：
#   bash tools/run_leave_one_out.sh              # 跑全部 A-E
#   ONLY=A,B  bash tools/run_leave_one_out.sh    # 只跑指定配置
#   DEVICE=cpu bash tools/run_leave_one_out.sh   # 无 GPU
#   NO_ORIGA=1 bash tools/run_leave_one_out.sh   # 若缺 ORIGA 掩膜，自动剔除
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-configs/test_fundus_local.yaml}"
ONLY="${ONLY:-A,B,C,D,E}"
TTT_STEPS="${TTT_STEPS:-None}"

# 官方测试列表（test_segment.yaml 第 5/7/9/11/13 行的原样内容）
_ORIGA='"ORIGA_train","ORIGA_test",'
[ -n "${NO_ORIGA:-}" ] && _ORIGA=''
declare -A DOMAINS
DOMAINS[A]="(\"REFUGE_train\",\"REFUGE_test\",${_ORIGA}\"REFUGE_Valid\",\"Drishti_GS_train\",\"Drishti_GS_test\")"
DOMAINS[B]="(\"RIM_ONE_r3_train\",\"RIM_ONE_r3_test\",${_ORIGA}\"REFUGE_Valid\",\"Drishti_GS_train\",\"Drishti_GS_test\")"
DOMAINS[C]='("RIM_ONE_r3_train","RIM_ONE_r3_test","REFUGE_train","REFUGE_test","REFUGE_Valid","Drishti_GS_train","Drishti_GS_test")'
DOMAINS[D]="(\"RIM_ONE_r3_train\",\"RIM_ONE_r3_test\",\"REFUGE_train\",\"REFUGE_test\",${_ORIGA}\"Drishti_GS_train\",\"Drishti_GS_test\")"
DOMAINS[E]="(\"RIM_ONE_r3_train\",\"RIM_ONE_r3_test\",\"REFUGE_train\",\"REFUGE_test\",${_ORIGA}\"REFUGE_Valid\")"
[ -n "${NO_ORIGA:-}" ] && echo "NO_ORIGA=1 -> 已从各配置测试列表剔除 ORIGA_*"

echo "留一法评测   device=$DEVICE  config=$CONFIG  TTT_STEPS=$TTT_STEPS"
echo "配置: $ONLY"
echo

IFS=',' read -ra WANTED <<< "$ONLY"
for K in "${WANTED[@]}"; do
    K="$(echo "$K" | tr -d ' ')"
    [ -n "${DOMAINS[$K]:-}" ] || { echo "[skip] 未知配置 $K"; continue; }
    OUT="output/loo_$K"
    echo "################ 配置 $K  (源域 = $K, model_$K.pth) ################"
    MODEL="$K" CONFIG="$CONFIG" DEVICE="$DEVICE" TTT_STEPS="$TTT_STEPS" \
        DOMAINS="${DOMAINS[$K]}" OUTPUT_DIR="$OUT" \
        bash tools/run_fundus_eval.sh 2>&1 | tail -20
    echo
done

echo "全部完成。用以下命令汇总："
echo "  .venv/bin/python tools/summarize_leave_one_out.py"
