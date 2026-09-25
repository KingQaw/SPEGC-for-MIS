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

### 类别约定 —— **类别 ID 的顺序是个大坑**

眼底任务是**视盘（OD）+ 视杯（OC）**两类的实例分割，对应
`configs/test_segment.yaml` 的 `MODEL.ROI_HEADS.NUM_CLASSES: 2`——与
`weights/fundus_source/*.pth` 完全一致（实测 `cls_score.weight` 形状 `(3, 1024)`
= 2 类 + 1 背景）。

detectron2 的 `load_coco_json` 会**按 category id 升序**重映射为 0..N-1，
所以「JSON 里 id 小的」就是模型里的**类 0**。这个顺序必须和作者训练权重时一致：

| JSON categories | 模型类 0 | 模型类 1 |
|---|---|---|
| `[{id:1,"optic_disc"}, {id:2,"optic_cup"}]`（**错**） | 视盘 | 视杯 |
| `[{id:1,"optic_cup"}, {id:2,"optic_disc"}]`（**对**） | **视杯** | **视盘** |

**怎么发现顺序反了的**：三个独立证据指向同一结论——

1. **置信度**：模型的类 1 预测置信度中位 **0.99**，而类 0 几乎没有高分预测
   （REFUGE 上类 0 的 max 只有 0.33，永远过不了 `DICE_THRES: 0.9`）。
   大而好分的视盘没理由比视杯更不确定。
2. **面积**：REFUGE 的 GT 面积中位数是 视盘 11608 / 视杯 2580，
   而模型高分预测的面积中位数是 **15000** —— 明显是视盘，不是视杯。
3. **指标上限**：若把模型的视盘预测拿去和 GT 的视杯比，
   Dice 上限 = `2×2580/(15000+2580) ≈ 29%`，实测 **34.69%**，正好卡在这个上限附近。

交换后语义也自洽了：类 0（视杯）面积中位数 < 类 1（视盘），比值
REFUGE 0.234 / RIM-ONE 0.242 / Drishti 0.570，都符合已知的杯盘比。

**影响**（`model_C.pth`，同一套权重、同一份数据，只改 JSON 里两个 id 的顺序）：

| 数据集 | 顺序错 | **顺序对** |
|---|---|---|
| RIM_ONE_r3_test | 40.81% | **88.93%** |
| RIM_ONE_r3_train | 41.33% | **89.35%** |
| REFUGE_Valid | 33.30% | **81.00%** |
| REFUGE_train | 30.53% | **81.45%** |
| Drishti_GS_test | 74.52% | **89.35%** |
| Drishti_GS_train | 73.22% | **88.92%** |

> 这个坑不会报错、不会崩溃，只是**指标静默腰斩**。任何复现这类工作的人如果不
> 去核对类别顺序，很容易误以为是自己训练/环境有问题。

转换后每个实例一个类别（视杯嵌套在视盘内，实例分割允许重叠）。


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

- **RIM-ONE 的 "Stereo Images" 是左右两只眼并排**（第二个大坑）：
  单张文件是 `2144×1424`，其实等于 **2 × 1072×1424**，左半和右半各是一张完整的
  眼底图；而**标注只画在其中一半上**。实测全部 159 张的标注都在左半
  （disc 掩膜中心 `cx/(W/2) ∈ [0.46, 0.56]`）。

  不做切分就直接用整图的后果：模型会**正确地分割出两只眼**的视盘/视杯，
  而 GT 只有一份，多出来的预测在 `DiceEvaluator` 里全部记为假阳性——
  RIM-ONE 的 Dice 从 **88.93% 掉到 18.75%**。

  脚本按掩膜实际位置判断该取哪一半（不硬编码"左"），并先切半再做 ROI 裁剪。
- **Drishti_GS 掩膜尺寸比图像小**：掩膜 `2045×1752`，图像 `2047×1760`。
  转换时用最近邻放大对齐，否则逐像素比对会错位。
- **RIM-ONE 无官方 train/test 划分**：自带的 MATLAB `Scripts/` 只有评测函数
  （`sevaluate.m`、`printTestResult.m`），没有划分定义。论文 4.2 节说明用的是
  **「每个源数据集随机 8:2 划分」**，脚本据此生成（整体随机、不分层，
  固定种子 `20260323`）→ train 127 / test 32。
- **Windows `Zone.Identifier` 侧写文件**：`datasets/raw` 下每个文件都带一个
  `xxx:Zone.Identifier` 同伴文件（ADS 标记），统计文件数时会翻倍，转换脚本
  统一过滤。

### ROI 裁剪的实测结论

论文要求「cropping the ROI of each image to 800×800」。实测**这四个数据集本身
就已经被提供方裁到视网膜了**，所以 ROI 外接框≈整图，真正起作用的是**缩放到
800×800**：

| 数据集 | 原尺寸 | 非黑占比 | ROI 框 |
|---|---|---|---|
| REFUGE train | 2124×2056 | 0.79（≈π/4，内切圆） | ≈整图 |
| REFUGE val | 1634×1634 | 0.72 | 1565×1565 |
| Drishti_GS | 2045×1752 | 0.86 | ≈整图 |
| RIM-ONE（切半后） | 1072×1424 | 0.91–0.99 | ≈整半 |

实现上仍保留 ROI 检测（`fundus_roi_box`：灰度阈值取非黑外接框，圆形的外接框
近似正方，所以缩放到 800×800 不会破坏几何比例），对未被裁好的数据能自动生效。


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

### ORIGA-650 获取途径调研（2026-09 核实）

**官方渠道已关闭。** iMED 官方数据集页（<https://imed.nimte.ac.cn/download.html>）
上明确写着：

> **！Origa-650（no longer available to the public, 不再提供下载！）**

页面更新日期 `2020-05-27`。2017 年发布公告里给的下载地址
`http://imed.nimte.ac.cn/resources.html` 现在返回「栏目不存在」。

**官方数据集本身是带 OD/OC 掩膜的。** 2017 年发布公告原文：

> iMED-Origa650数据集是由丰富临床经验的专业医生标注的 650 张眼底图，
> 其中 168 张青光眼患者的眼底图，482 张正常人群的眼底图

用途就是「青光眼自动诊断和**视杯视盘分割**」。所以
`datasets/raw/ORIGA`（Kaggle `ferencjuhsz/...`）只有分类 csv，是个**残缺子集**，
不是官方完整版。

**可尝试的途径（按推荐顺序）**：

1. **直接联系论文作者**（最实际）。SPEGC 把 ORIGA 当作 Domain C 使用，
   作者手上必然有带掩膜的版本，甚至可能就是他们的划分。论文给出的邮箱：
   `duxiaogang@sust.edu.cn`（杜晓刚，一作）、`leitao@sust.edu.cn`（雷涛，通讯）、
   `hello.jiawei@outlook.com`。顺便还能确认类别 ID 顺序与 8:2 划分。
2. **联系 iMED 实验室**。官方页留的联系人：岳星宇 `yuexingyu@nimte.ac.cn`。
   公开下载虽已关闭，学术用途的个人申请可能仍受理。
3. **Wayback Machine** 存档的旧下载页（本环境 DNS 受限打不开，可自行尝试）：
   `https://web.archive.org/web/2020/http://imed.nimte.ac.cn/resources.html`
4. **第三方镜像**（有掩膜，但来源非官方，需自行核对真实性）：
   - CSDN 整理帖 <https://blog.csdn.net/weixin_43518285/article/details/148282613>
     给出的目录含 `Masks/`、`Masks_Cropped/`、`Masks_Square/`、
     `Semi-automatic-annotations/`、`Origalist.csv`（含 `ExpCDR`/`Glaucoma` 字段）。
     ⚠️ 其中的 `Images_Square` / `Masks_Square` 很可能**已经是论文所说
     "cropping the ROI" 之后的方形图**，用之前先核对尺寸。
   - IEEE DataPort 有 "Glaucoma Screening dataset" 条目。
   - Kaggle 上有多个 ORIGA 变体（有的 520 张、有的 650 张），需逐个确认是否含
     `Masks/`。

**拿到数据后要核对的三件事**（都是本项目踩过的坑）：

1. **类别 ID 顺序**：官方 `Masks` 里哪张是 cup、哪张是 disc，对应到 JSON 的
   `category_id` 顺序必须与 `weights/fundus_source/model_C.pth` 一致
   （见第 1 节；类 0 = 视杯、类 1 = 视盘）。
2. **划分**：论文是每域随机 8:2，需确认是否有官方划分文件。
3. **图像是否已裁 ROI**：`_Square` 版可能已裁好，别重复裁剪。



---

## 4. 转换与用法

```bash
# raw -> datasets/Fundus/<名字>/{图像} + datasets/Fundus/<名字>_<划分>.json
# 默认按论文协议：RIM-ONE 切半 -> ROI 裁剪 -> 缩放 800x800
.venv/bin/python tools/prepare_fundus_data.py

# 可选参数
.venv/bin/python tools/prepare_fundus_data.py --datasets REFUGE Drishti_GS
.venv/bin/python tools/prepare_fundus_data.py --rimone-expert exp1   # 换专家标注
.venv/bin/python tools/prepare_fundus_data.py --roi-size 512         # 换输出尺寸
.venv/bin/python tools/prepare_fundus_data.py --mode link            # 只建符号链接，不裁剪
.venv/bin/python tools/prepare_fundus_data.py --no-rimone-split-stereo
```

- **默认模式写出的是 800×800 的真实 JPEG**（1359 张，约 211 MB）；
  `--mode link` 则保留原图、只建符号链接（1460 个链接约 2.5 MB），
  用于不想改数据时的快速冒烟。
- 布局与 `README.md` 的约定一致，且正是 `data/datasets/builtin.py` 注册的路径。
- 输出是标准 COCO instance segmentation：多边形 `segmentation`、`bbox`、
  `iscrowd=0`、`ignore=0`、唯一 id。多边形经 `approxPolyDP` 抽稀。
- ⚠️ 因为已经是 800×800，detectron2 的 `MIN_SIZE_TEST: 800` 不会再缩放它们。

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

### 结果 A：留一法全量复现 Table 1（**主要结果**）

论文 Table 1 的每一列是「在该域上测试、源模型来自其余四个域」的均值
（"based on five experimental runs"）。用 `tools/run_leave_one_out.sh` 跑齐
5 个源模型（配置 K 用 `model_K.pth`，测试列表 = 除域 K 外的全部域）：

```bash
bash tools/run_leave_one_out.sh                       # 跑 A–E
.venv/bin/python tools/summarize_leave_one_out.py     # 汇总
```

**逐次运行**（行 = 源模型，列 = 测试数据集，Dice %）：

| 源 | Drishti_test | Drishti_train | REFUGE_Valid | REFUGE_train | RIM_test | RIM_train |
|---|---|---|---|---|---|---|
| A | 85.37 | 83.91 | 81.93 | 86.95 | — | — |
| B | 88.00 | 88.72 | 79.41 | — | 82.90 | 81.27 |
| C | 89.35 | 88.92 | 81.00 | 81.45 | 88.93 | 89.35 |
| D | 83.17 | 83.44 | — | 87.85 | 86.49 | 83.52 |
| E | — | — | 63.16 | 75.06 | 84.96 | 83.51 |

**留一平均 vs 论文 Table 1（DSC）**：

| 测试域 | 本项目 | 论文 | 差值 |
|---|---|---|---|
| A (RIM-ONE) | **85.11** | 84.90 | **+0.21** |
| B (REFUGE) | **82.83** | 83.34 | **−0.51** |
| C (ORIGA) | n/a（无掩膜） | 84.57 | — |
| D (REFUGE-Test) | **76.38** | 83.54 | **−7.16** |
| E (Drishti-GS) | **86.36** | 85.51 | **+0.85** |

**四列中有三列误差在 ±0.85 以内**，可以认为协议已经对齐。

域 D 偏低 7.16 分，来源是单一离群：`model_E`（源域 = Drishti）在
REFUGE 上只有 63.16，而 A/B/C 都在 79–82。这与形态学差异吻合——
Drishti 的杯盘比中位数 0.57，REFUGE 只有 0.23，在 Drishti 上训练的模型
倾向于在 REFUGE 上把视杯预测得过大。`model_E` 在 `REFUGE_train` 上同样最低
（75.06），两条证据一致。

**与论文的三个口径差异**（解读时务必注意）：

1. **缺 ORIGA**：域 C 整列无法计算；其余四列的测试列表是从官方列表里
   **剔除 ORIGA** 后的版本，不是论文的完整留一。
2. **指标口径**：这里用的是仓库 `DiceEvaluator`（每个**预测**取最佳匹配后求
   均值），不是标准按图/按类 DSC，详见本节末。
3. **模型来源**：直接使用作者发布的 5 个源模型权重，**没有自己重训源模型**。
   论文的源模型是「每域随机 8:2」划分训练（SGD, lr 0.001, bs 8）。

### 结果 B：单模型 `model_C.pth` 视图

`configs/test_config_C.yaml`，目标域 = 域 A/B/D/E（不含 ORIGA）。
数据经 ROI 裁剪→800×800、RIM-ONE 已切半、类别 ID 已按作者顺序修正：

| 数据集 | Dice | Enhanced Alignment | Structural Similarity |
|---|---|---|---|
| Drishti_GS_test *(Domain E)* | **89.35%** | 97.40% | 92.52% |
| Drishti_GS_train | 88.92% | 97.83% | 92.12% |
| RIM_ONE_r3_test *(Domain A)* | **88.93%** | 97.25% | 91.43% |
| RIM_ONE_r3_train | 89.35% | 97.45% | 91.75% |
| REFUGE_train *(Domain B)* | 81.45% | 92.69% | 87.50% |
| REFUGE_Valid *(Domain D)* | 81.00% | 95.09% | 85.50% |
| **REFUGE_test**（**无标注**） | **0.0000%** | **0.0000%** | **0.0000%** |
| ~~REFUGE_mean~~ | ~~54.15%~~ | ~~62.59%~~ | ~~57.67%~~ |

`REFUGE_test` 无标注，三项全 0，把 `REFUGE_mean` 从 81.22% 拖到 54.15%
（`(81.45+81.00)/2 = 81.22`）。报指标时必须排除该集或补齐标注。
汇总脚本已把它从域 B 的成员里排除。

### 结果 C：旧数据（已废弃）

> ⚠️ 以下是**修正类别顺序之前**跑的，仅保留作为对比，不要引用。


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

### 指标口径：`DiceEvaluator` 不是标准 DSC

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
后果是**假阳性会被当作 0 分拉低均值**。

> ⚠️ **一个已修正的错误结论**：我最初把 RIM-ONE 的低分（18.75%）归因于这个
> 指标口径 + 假阳性。**这是错的**。真正原因是两条：
> (1) RIM-ONE 立体图没切半，模型分割了两只眼而 GT 只标了一只；
> (2) **类别 ID 顺序反了**，模型的视盘预测被拿去和 GT 的视杯比。
> 修完这两条后 RIM-ONE 到 88.93%。指标口径仍然是个需要注意的差异，但
> **不是**当初低分的主因。

我实测对比过两种口径（同一批预测、`model_C`、修正类别顺序**之前**的数据）：

| 数据集 | 每图预测 | 每图 GT | 命中率 | 仓库口径 | 标准 DSC |
|---|---|---|---|---|---|
| Drishti_GS_test | 1.1 | 2.0 | 0.51 | 74.52% | 37.86% |
| REFUGE_Valid | 0.9 | 2.0 | 0.46 | 34.69% | 15.90% |
| RIM_ONE_r3_test | 1.2 | 2.0 | 0.61 | 40.81% | 24.87% |

可见标准 DSC 反而更低（因为它把漏检的类按 0 计入）。
**论文的 DSC 究竟是哪种口径、是否对漏检做特殊处理，需要向作者确认**，
否则数字不能严格对齐。


### 已完成的对齐工作（本轮）

| 项 | 状态 | 效果 |
|---|---|---|
| RIM-ONE 立体图切半 | ✅ | Dice 18.75% → 40.81% |
| ROI 裁剪 + 缩放 800×800 | ✅ | 与论文协议一致 |
| RIM-ONE 随机 8:2 划分 | ✅ | 127 / 32，与论文 4.2 节一致 |
| **类别 ID 顺序修正** | ✅ | **Dice 全面 → 81–89%，进入论文区间** |

### 仍未完成 / 可继续的方向

1. **ORIGA**：申请带掩膜的 ORIGA-650，补齐配置 A/B/D/E（当前只有 C 能跑）。
2. **REFUGE_test 标注**：无标注，实测三项全 0 且污染 `REFUGE_mean`
   （81.22% → 54.15%）。要么补齐标注，要么在 `DATASETS.TEST` 里移除。
3. **Drishti 掩膜阈值**：目前 SoftMap 取 `>=128`（专家过半同意）。
   试 `>=255`（全体同意）看对结果的影响 —— Drishti 的杯盘比 0.57 偏高，
   阈值可能偏松。
4. **RIM-ONE 专家标注**：默认用 `Average_masks`；可试
   `--rimone-expert exp1/exp2` 对比。
5. **模型 A/B/D/E**：`MODEL=X bash tools/run_fundus_eval.sh`。A/B/D/E 的测试列表
   含 ORIGA，补齐数据前跑不出完整结果（可临时用 `DOMAINS=` 指定子集）。
6. **指标口径**：论文的 DSC 与仓库 `DiceEvaluator` 口径不同（详见第 5 节末）。
   若要严格对齐 Table 1，需要确认论文用的是哪种；论文每列是**留一平均**
   （4 个源模型），我们目前只跑了单个源模型。
7. **预处理归一化**：论文说 min-max，仓库配置用的是 detectron2 默认 ImageNet
   统计量。当前沿用仓库设置（毕竟权重是用它训的）。
8. **`test.sh` 原样复现**：需要先补齐 ORIGA 与 REFUGE_test，再改
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
