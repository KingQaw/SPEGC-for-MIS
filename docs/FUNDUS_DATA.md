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
  （`sevaluate.m`、`printTestResult.m`），没有划分定义。脚本用**固定种子的分层
  划分**（按 Healthy/Glaucoma 分层，test 占 30%，seed=`20260323`），得到
  train 111 / test 48。要复现论文的精确划分需向作者索取。
- **Windows `Zone.Identifier` 侧写文件**：`datasets/raw` 下每个文件都带一个
  `xxx:Zone.Identifier` 同伴文件（ADS 标记），统计文件数时会翻倍，转换脚本
  统一过滤。

---

## 3. 数据缺口

| 缺口 | 影响 |
|---|---|
| **ORIGA 没有分割掩膜** | 该下载只有 `ImageName,glaucoma` 分类 csv。`test_segment.yaml` 的 **A / B / D / E 四个配置都用到 ORIGA**，因此这四个配置无法原样复现。需要另找带 OD/OC 掩膜的 ORIGA-650（iMED 官方站需申请）。 |
| **REFUGE_test 没有标注** | 400 张测试图只有 `Images/`，没有 `gts/`。配置 A / C / D / E 用到它。脚本会生成只有图像的 json，但**在该集上算 Dice 没有意义**。 |

**结论：五个官方配置没有一个能用当前数据原样跑通。**
可用的域是：`REFUGE_train`、`REFUGE_Valid`、`RIM_ONE_r3_train/test`、
`Drishti_GS_train/test`（共 1060 张带标注图像）。

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
bash tools/run_fundus_eval.sh                    # GPU + 模型B + 全部可用域
DEVICE=cpu bash tools/run_fundus_eval.sh         # 无 GPU
TTT_STEPS=4 bash tools/run_fundus_eval.sh        # 少量 TTT 步，快速冒烟
MODEL=C bash tools/run_fundus_eval.sh            # 换权重 A..E
```

脚本内部执行的就是仓库原本的 `test.sh` 路径（`--eval-only` +
`SEMISUPNET.Trainer: baseline` + `TEST.TTT: True`），只是把 `DATASETS.TEST`
换成实际可用的域，并补上 `MODEL.DEVICE`。

### 结果（`model_B.pth`，GPU，TTT 跑满整个目标流）

| 数据集 | Dice | Enhanced Alignment | Structural Similarity |
|---|---|---|---|
| Drishti_GS_test | **63.85%** | 76.98% | 73.54% |
| Drishti_GS_train | 62.94% | 76.13% | 72.88% |
| REFUGE_Valid | 51.25% | 67.95% | 68.17% |
| RIM_ONE_r3_test | **22.67%** | 59.59% | 54.40% |
| RIM_ONE_r3_train | 23.15% | 60.31% | 54.61% |

TTT 确实在生效——日志里每个域前几步的 loss 从 `None`（图池未满，
`CTTA_MIN_POOL_SIZE`）变成非零并逐步变化（实测 3.45 → 5.69 → 6.37），
说明 SPEGC 的图聚类损失在反传、模型在在线适应。

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

1. **ORIGA**：申请带掩膜的 ORIGA-650，补齐配置 A/B/D/E。
2. **RIM-ONE 划分**：当前是自建分层划分，与论文可能不同；且可试
   `--rimone-expert exp1/exp2` 看标注选择对结果的影响。
3. **REFUGE_test**：无标注，若只为测 CTTA 的适应过程可用，但不要报 Dice。
4. **模型 A–E 全跑**：`weights/fundus_source/` 有 5 个权重，
   `MODEL=A..E bash tools/run_fundus_eval.sh` 即可，对应论文的 5 种目标域顺序。
5. **`test.sh` 原样复现**：需要先补齐 ORIGA 与 REFUGE_test 标注，再改
   `configs/test_segment.yaml`（注意 `test.sh` 靠**行号** 5/7/9/11/13 做 sed，
   改文件会打乱映射）。
