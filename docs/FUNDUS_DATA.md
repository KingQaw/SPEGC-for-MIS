# 眼底数据准备与 CTTA 评测

承接 [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)。本文记录**用真实眼底数据 +
官方预训练权重跑通评测**的全过程：数据盘点、格式转换、踩到的编码陷阱、数据缺口，
以及结果解读。

---

## 1. 数据盘点（`datasets/raw/`）

| 数据集 | 图像 | 标注 | 划分 | 原始格式 |
|---|---|---|---|---|
| **REFUGE** | 1200 jpg | train 400 + val 400 bmp | train/val/test 已给 | 单通道 BMP，3 值 |
| **Drishti_GS** | 101 png | 101 组 SoftMap | Training 50 / Test 51 | 软概率图 PNG |
| **RIM-ONE r3** | 159 jpg | 159×2×3 png | **无官方划分** | 二值 PNG（+MAT/TXT） |
| **ORIGA** | 650 jpg | **无分割掩膜** | train 454 / test 196 | 仅青光眼分类 csv |

来源见 `datasets/raw/README.md`（Kaggle / 百度 AI Studio / 官方站）。

### 类别约定

眼底任务是**视盘（OD）+ 视杯（OC）**两类的实例分割，对应
`configs/test_segment.yaml` 的 `MODEL.ROI_HEADS.NUM_CLASSES: 2`——与
`weights/fundus_source/*.pth` 完全一致（实测 `cls_score.weight` 形状 `(3, 1024)`
= 2 类 + 1 背景）。转换后写成一个 `disc` 一个 `cup` 两个实例（视杯嵌套在视盘内，
实例分割允许重叠）。

---

## 2. 各数据集标注编码差异（**最大的坑**）

三个数据集的掩膜编码**互不相同**，而且 REFUGE 的编码与直觉相反：

| 数据集 | 编码 | 正确解读 |
|---|---|---|
| **REFUGE** | `0` / `128` / `255` | **`0`=视杯，`128`=视盘环（盘减杯），`255`=背景**<br>→ `disc = (gt != 255)`，`cup = (gt == 0)` |
| **Drishti_GS** | SoftMap `0/64/128/191/255` | 多专家一致性软图 → 阈值 `>=128` 二值化，OD 与 cup 各一张 |
| **RIM-ONE r3** | `0` / `255` | 二值 → `>0` 即可；有 Expert1/Expert2/**Average** 三套，默认取 Average（专家共识） |

> ⚠️ REFUGE 这一步如果不验证就按"常规假设（0=背景、128=盘、255=杯）"写，
> 会把背景当成视杯，**指标静默算错**。
>
> 验证方式（脚本里用的判据）：对 `128` 区域做形态学填洞
> `binary_fill_holes`，结果应与 `(gt != 255)` 完全相等，且 `0` 区域完全落在
> 洞内。train/val 各抽 8 张实测全部成立。

### 其它格式陷阱

- **Drishti_GS 掩膜尺寸比图像小**：掩膜 `2045×1752`，图像 `2047×1760`。
  转换时用最近邻放大对齐，否则逐像素比对会错位。
- **RIM-ONE 无官方 train/test 划分**：自带的 MATLAB `Scripts/` 只有评测函数
  （`sevaluate.m`、`printTestResult.m`），没有划分定义。论文 4.2 节说明用的是
  **「每个源数据集随机 8:2 划分」**，脚本据此生成（固定种子 `20260323`）。
- **Windows `Zone.Identifier` 侧写文件**：`datasets/raw` 下每个文件都带一个
  `xxx:Zone.Identifier` 同伴文件（ADS 标记），统计文件数时会翻倍，转换脚本
  统一过滤。

---

## 2.5 论文的基准设定（据 CVPR 2026 正文 4.1/4.2 节）

### 五个眼底域 —— `REFUGE_Valid` 就是论文的 "Domain D"

论文原文：

> We employed five public datasets ... These include: **Domain A (RIM-ONE),
> Domain B (REFUGE), Domain C (ORIGA), Domain D (REFUGE-Test) and
> Domain E (Drishti-GS)**.

注意 **REFUGE 被拆成了两个域**（B = REFUGE，D = REFUGE-Test），这就是为什么仓库里
`REFUGE_Valid` 和 `REFUGE_test` 是两个独立条目。

论文 Table 1 的设定是「**Domain A 表示在 Domain A 上训练、在 Domain B–E 上测试**」。
用它反推 `configs/test_segment.yaml` 的五行配置，得到**完全自洽**的结果——每个
配置 K 恰好只缺域 K：

| 配置 | 测试列表缺失的数据集 | 对应域 | 是否自洽 |
|---|---|---|---|
| A | `RIM_ONE_r3_train/test` | Domain A (RIM-ONE) | ✓ |
| B | `REFUGE_train/test` | Domain B (REFUGE) | ✓ |
| C | `ORIGA_train/test` | Domain C (ORIGA) | ✓ |
| **D** | **`REFUGE_Valid`** | **Domain D (REFUGE-Test)** | ✓ |
| E | `Drishti_GS_train/test` | Domain E (Drishti-GS) | ✓ |

验证脚本（5 个配置全部成立）：

```python
missing = set(all_names) - set(cfg[k])   # 恰好等于 dom[k]
```

**所以仓库里的 `REFUGE_Valid` 扮演的是论文的 "REFUGE-Test" 域**，而
`REFUGE_test`（无标注那 400 张）属于域 B 的数据流。

### 息肉的四个域

> Domain A (BKAI-IGH-NEOPolyp), Domain B (CVC-ClinicDB/CVC-612),
> Domain C (ETIS), Domain D (Kvasir)

### 数据预处理（**与当前实现有差异**）

> During the data preprocessing stage, we followed the established protocol in
> [5,36], first **cropping the Region of Interest (ROI) of each image to
> 800×800 pixels** and subsequently applying **min-max normalization**.

- 息肉任务则是：**统一 resize 到 800×800**，用 **ImageNet 统计量归一化**。
- ⚠️ 当前 `prepare_fundus_data.py` **没有做 ROI 裁剪**，也没有改归一化方式——
  图像以原始尺寸（2047×1760 / 2124×2056 / 2144×1424）交给 detectron2 的
  `ResizeShortestEdge` 缩放到短边 800。这会改变视盘在输入中的相对尺度，
  是**与论文数值对齐前必须补上的一步**。
- 归一化方面，仓库配置没有覆盖 `MODEL.PIXEL_MEAN/STD`，用的仍是 detectron2
  默认的 ImageNet 统计量，与论文说的 min-max 也不一致。

### 训练与 CTTA 超参（与仓库默认值核对）

| 项目 | 论文 | 仓库 | 一致 |
|---|---|---|---|
| 源模型划分 | 每域随机 8:2 | 需在 `DATASETS.*` 体现 | 需确认 |
| 源模型优化器 | SGD, momentum 0.9, lr 0.001, bs 8 | — | — |
| 骨干 | ResNet-50 (ImageNet 预训练) | `RESNETS.DEPTH: 50` | ✓ |
| CTTA 学习率（眼底） | **0.005** | `test_segment.yaml` `BASE_LR: 0.005` | ✓ |
| CTTA 学习率（息肉） | 0.01 | — | — |
| 每样本适应步数 | 1 次迭代，无标签 | `MIN_BATCH_NUM`（None = 全流） | ✓ |
| λ | 0.2 | `SPEGC_LAMBDA: 0.2` | ✓ |
| P（低不确定性采样率） | 0.5 | `SPEGC_P: 0.5` | ✓ |
| 特征池大小 | 3（伪批 4） | `TTT_POOL_SIZE: 3` | ✓ |
| Z | 48 | `SPEGC_Z: 48` | ✓ |
| t（MC Dropout） | 4 | `SPEGC_T: 4` | ✓ |
| M（提示数） | 8 | `SPEGC_M: 8` | ✓ |
| 硬件 | 单卡 RTX 3090 | 本机 2×RTX 2070 | — |

---

## 3. 数据缺口

| 缺口 | 影响 |
|---|---|
| **ORIGA 没有分割掩膜** | 该下载只有 `ImageName,glaucoma` 分类 csv。ORIGA 是 **Domain C**，因此除了配置 C 之外的所有配置都会在测试列表里用到它。需要另找带 OD/OC 掩膜的 ORIGA-650（iMED 官方站需申请）。 |
| **REFUGE_test 没有标注** | 400 张图只有 `Images/`，没有 `gts/`。它属于 **Domain B (REFUGE)** 的数据流，出现在配置 A/C/D/E 的列表里。脚本会生成只有图像的 json，但**在该集上算 Dice 没有意义**。 |

**结论：配置 C 是唯一不依赖 ORIGA、因而可以用当前数据完整跑通的官方配置。**

- 配置 C = 源模型在 Domain C (ORIGA) 上训练，在 **A / B / D / E** 上测试；
  ORIGA 只作为源域，而源域权重已由 `weights/fundus_source/model_C.pth` 提供，
  **不需要我们持有 ORIGA 的掩膜**。
- 其余配置（A/B/D/E）的测试列表都含 ORIGA，需先补齐该数据集。

带标注、可用于评测的域共 1060 张：`REFUGE_train`(400)、`REFUGE_Valid`(400)、
`RIM_ONE_r3_train/test`(159)、`Drishti_GS_train/test`(101)。
`REFUGE_test`(400) 只能用于无监督适应，不能用于报指标。


---

## 4. 转换与用法

```bash
# raw -> datasets/Fundus/<名字>/ (符号链接) + datasets/Fundus/<名字>_<划分>.json
.venv/bin/python tools/prepare_fundus_data.py

# 可选参数
.venv/bin/python tools/prepare_fundus_data.py --datasets REFUGE Drishti_GS
.venv/bin/python tools/prepare_fundus_data.py --rimone-expert exp1   # 换专家标注
```

- **图像用符号链接，不复制数据**：1460 个链接只占 2.5 MB（原始数据 5.4 GB）。
- 布局与 `README.md` 的约定一致，且正是 `data/datasets/builtin.py` 注册的路径。
- 输出是标准 COCO instance segmentation：多边形 `segmentation`、`bbox`、
  `iscrowd=0`、`ignore=0`、唯一 id。多边形经 `approxPolyDP` 抽稀。

### 转换正确性验证

1. **每类恰好 1 个实例**：所有数据集都是 `2 × 图像数` 条标注（如 REFUGE_train
   400 图 → 800 标注），说明掩膜没有碎片化成多余轮廓。
2. **语义校验：视杯 ⊂ 视盘**（容差 2%）——1060 张全部通过。
3. **形态学校验**：视杯/视盘面积比中位数

   | 数据集 | disc 占图比 | cup/disc |
   |---|---|---|
   | Drishti_GS_test | 3.05% | 0.602 |
   | REFUGE_Valid | 1.65% | 0.234 |
   | RIM_ONE_r3_test | 2.06% | 0.213 |

   Drishti 的 cup/disc 明显更高，与其**青光眼占比高**（101 张里 70 张青光眼，
   杯盘比天然偏大）一致；另两个以正常眼为主。

### ⚠️ 一个容易踩的坑：`thing_classes`

`builtin.py` **不允许**预设 `thing_classes`。detectron2 的 `load_coco_json` 会用
JSON 里的 `categories` 去覆盖它，并断言两者相等：

```
AssertionError: Attribute 'thing_classes' in the metadata of 'X'
cannot be set to a different value!  ['lesion'] != ['optic_disc', 'optic_cup']
```

所以 `builtin.py` 只注册 `json_file` / `image_root` / `evaluator_type`，把类别名
交给 JSON 作为唯一真源。这样眼底（disc/cup）、息肉（单类）、合成（lesion）可以
共存而互不冲突。

---

## 5. 评测

```bash
# 复现论文 Table 1 的 Domain C 列（不需要 ORIGA，见第 3 节）
MODEL=C CONFIG=configs/test_config_C.yaml OUTPUT_DIR=output/test_config_C \
    bash tools/run_fundus_eval.sh

# 其它用法
bash tools/run_fundus_eval.sh                    # 默认配置 + 模型B
DEVICE=cpu bash tools/run_fundus_eval.sh         # 无 GPU 环境
TTT_STEPS=4 bash tools/run_fundus_eval.sh        # 少量 TTT 步，快速冒烟
MODEL=A bash tools/run_fundus_eval.sh            # 换源模型权重
DOMAINS='("Drishti_GS_test",)' bash tools/run_fundus_eval.sh   # 临时覆盖目标域
```

脚本执行的就是仓库原本 `test.sh` 的路径（`--eval-only` +
`SEMISUPNET.Trainer: baseline` + `TEST.TTT: True`），只是补上 `MODEL.DEVICE`。

> ⚠️ **脚本行为**：`DOMAINS` 留空（默认）时使用**配置文件里**的 `DATASETS.TEST`；
> 只有显式设置 `DOMAINS` 才会覆盖。早期版本无条件用默认值覆盖，会把配置里的
> `REFUGE_train`/`REFUGE_test` 悄悄挤掉——已修。

### 结果 A：`model_C.pth`（源域 = ORIGA），复现 Table 1 的 Domain C 列

`configs/test_config_C.yaml`，目标域 = 域 A/B/D/E（不含 ORIGA）：

| 数据集 | Dice | Enhanced Alignment | Structural Similarity |
|---|---|---|---|
| Drishti_GS_test | **75.71%** | 87.27% | 81.63% |
| Drishti_GS_train | 72.94% | 84.78% | 79.64% |
| REFUGE_Valid *(= 论文 Domain D)* | 31.55% | 46.84% | 53.61% |
| REFUGE_train *(Domain B)* | 30.10% | 45.35% | 54.10% |
| RIM_ONE_r3_test *(Domain A)* | 18.75% | 51.80% | 51.44% |
| RIM_ONE_r3_train | 18.64% | 51.56% | 51.52% |
| **REFUGE_test**（**无标注**） | **0.0000%** | **0.0000%** | **0.0000%** |
| ~~REFUGE_mean~~ | ~~20.55%~~ | ~~30.73%~~ | ~~35.91%~~ |

**`REFUGE_test` 实测三项全 0，并把 `REFUGE_mean` 从约 31% 拖到 20.55%**——
这就是缺标注的具体后果，不是模型差。报指标时必须排除该集，或补齐标注。

### 结果 B：`model_B.pth`（源域 = REFUGE）

| 数据集 | Dice | Enhanced Alignment | Structural Similarity |
|---|---|---|---|
| Drishti_GS_test | **63.85%** | 76.98% | 73.54% |
| Drishti_GS_train | 62.94% | 76.13% | 72.88% |
| REFUGE_Valid | 51.25% | 67.95% | 68.17% |
| RIM_ONE_r3_test | **22.67%** | 59.59% | 54.40% |
| RIM_ONE_r3_train | 23.15% | 60.31% | 54.61% |

> 结果 A/B 的 `DATASETS.TEST` 不完全相同，横向比较需注意。但有一点一致且符合
> 直觉：**模型在自己的源域上表现最好**——`model_B`（源=REFUGE）在
> `REFUGE_Valid` 上是 51.25%，而 `model_C`（源=ORIGA）只有 31.55%。

TTT 确实在生效——日志里每个域前几步的 loss 从 `None`（图池未满，
`CTTA_MIN_POOL_SIZE`）变成非零并逐步变化（`model_B` 实测 3.45 → 5.69 → 6.37；
`model_C` 在 RIM-ONE 上从 10.43 收敛到 3.26），说明 SPEGC 的图聚类损失在反传、
模型在在线适应。

### RIM-ONE 为什么低这么多？——**指标定义**是主因

`evaluation/dice_metric.py:49-77` 的实现是：

```python
for pred_class, pred_mask in zip(pred_classes, pred_masks):
    best_dice_score = 0
    for gt_class, gt_mask in zip(gt_classes, gt_masks):
        if pred_class == gt_class:
            best_dice_score = max(best_dice_score, dice)
    self.dice_scores.append(best_dice_score * 100)   # ← 每个"预测"都记一笔
```

即**对每个预测实例取最佳匹配后求均值**，而不是标准的按图/按类 Dice。
后果是**假阳性会被当作 0 分拉低均值**。实测每图预测数：

| 数据集 | 预测/图 | 其中 ≥0.9 | GT/图 |
|---|---|---|---|
| Drishti_GS_test | 2.3 | 2.0 | 2.0 |
| REFUGE_Valid | 2.8 | 1.2 | 2.0 |
| RIM_ONE_r3_test | **4.5** | **3.8** | 2.0 |

RIM-ONE 上模型每图多产出约 1.8 个高置信度假阳性（真实域差异：立体眼底图、
人群不同），在别的指标下影响有限，但在这个定义下被显著放大。

> **所以这三个数字不能直接和论文里的 Dice 对比**，除非确认论文用的是同一套
> 评测代码的口径。跨数据集横向比较时也要记住这一点。

### 未完成 / 可继续的方向

1. **补 ROI 裁剪（优先级最高）**：论文对眼底图像先裁 ROI 再缩到 **800×800**，
   当前实现是原图交给 `ResizeShortestEdge`。这一步不补，数值无法与 Table 1 对齐。
2. **ORIGA**：申请带掩膜的 ORIGA-650，补齐配置 A/B/D/E（当前只有 C 能跑）。
3. **RIM-ONE 划分**：论文说「随机 8:2」，当前脚本是分层 70/30（seed 固定）。
   改成 8:2 才能对齐；另可试 `--rimone-expert exp1/exp2` 看标注选择的影响。
4. **REFUGE_test 标注**：无标注，实测三项全 0 且污染 `REFUGE_mean`。
   要么补齐标注，要么在 `DATASETS.TEST` 里移除。
5. **模型 A/B/D/E**：`MODEL=X bash tools/run_fundus_eval.sh`。A/B/D/E 的测试列表
   含 ORIGA，补齐数据前跑不出完整结果（可临时用 `DOMAINS=` 指定子集）。
6. **`test.sh` 原样复现**：需要先补齐 ORIGA 与 REFUGE_test，再改
   `configs/test_segment.yaml`（注意 `test.sh` 靠**行号** 5/7/9/11/13 做 sed，
   改文件会打乱映射）。当前用 `configs/test_config_C.yaml` 绕开了这个限制。

### 论文出处

> Xiaogang Du, Jiawei Zhang, Tongfei Liu, Tao Lei, Yingbo Wang.
> *SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph
> Clustering for Medical Image Segmentation.* CVPR 2026, pp. 8481-8491.
> [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Du_SPEGC_Continual_Test-Time_Adaptation_via_Semantic-Prompt-Enhanced_Graph_Clustering_for_Medical_CVPR_2026_paper.html)
> · [arXiv:2603.11492](https://arxiv.org/abs/2603.11492)
>
> 本文第 2.5 节的域定义、预处理与超参均引自该文 4.1/4.2 节。
