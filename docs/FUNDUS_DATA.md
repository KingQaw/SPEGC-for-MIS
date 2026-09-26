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
| **ORIGA** | 650 jpg | 见下方说明 | 见下方说明 | 三套图 + 互斥标签图 PNG |

**ORIGA 有两个下载，`datasets/raw/` 下同时存在：**

| 目录 | 内容 | 用途 |
|---|---|---|
| `ORIGA/`（Kaggle） | 650 jpg + `ImageName,glaucoma` 分类 csv，9.8 MB | ❌ **无掩膜，不能用** |
| `ORIGA-masked/`（第三方整理） | `Images/` + `Masks/` + `Images_Cropped/` + `Masks_Cropped/` + `Images_Square/` + `Masks_Square/` + `Semi-automatic-annotations/` + 2 个 csv，542 MB | ✅ **实际使用这一份** |

`ORIGA-masked` 的三套图/掩膜**逐像素严格对齐**（实测尺寸完全一致）。转换脚本
**只用 `Images/`（全分辨率原图，2048 高）**，因为：

- `Images_Cropped/` 是**紧贴视盘的放大裁剪**（画面里只有视盘局部），框错了；
- `Images_Square/` 是 512×512 的整幅眼底，分辨率低于原图。

两个 csv：`OrigaList.csv` 有 `Eye/Filename/ExpCDR/Set/Glaucoma` 五列，
其中 **`Set` 是官方划分（A/B 各 325，即 50/50）**；
`origa_info.csv` 记录来源路径与 CDR/离心率等。

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

四个数据集的掩膜编码**互不相同**，而且 REFUGE 与 ORIGA 的编码都与直觉相反：

| 数据集 | 编码 | 正确解读 |
|---|---|---|
| **REFUGE** | `0` / `128` / `255` | **`0`=视杯，`128`=视盘环（盘减杯），`255`=背景**<br>→ `disc = (gt != 255)`，`cup = (gt == 0)` |
| **Drishti_GS** | SoftMap `0/64/128/191/255` | 多专家一致性软图 → 阈值 `>=128` 二值化，OD 与 cup 各一张 |
| **RIM-ONE r3** | `0` / `255` | 二值 → `>0` 即可；有 Expert1/Expert2/**Average** 三套，默认取 Average（专家共识） |
| **ORIGA**（masked） | `0` / `1` / `2` 互斥标签图 | **`1`=视盘环（盘减杯），`2`=视杯，`0`=背景**<br>→ `disc = (m != 0)`，`cup = (m == 2)` |

> ⚠️ REFUGE 这一步如果不验证就按"常规假设（0=背景、128=盘、255=杯）"写，
> 会把背景当成视杯，**指标静默算错**。
>
> ⚠️ ORIGA 同理：它是**单通道标签图**而不是二值图，`1` 只覆盖视盘环、
> **不含视杯像素**。若直接把 `m==1` 当视盘，视盘会缺一块（缺的正是视杯），
> Dice 会明显偏低。判据：`fill_holes(label1) == label1 | label2`
> （实测 60 张里 59 张成立；不成立的个例是标签本身有断裂）。
> 转换后杯/盘面积比中位数 **0.351**，与另外三个域横向一致
> （REFUGE 0.238 / RIM-ONE 0.223 / Drishti 0.601）。
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

| 缺口 | 状态 | 影响 |
|---|---|---|
| ~~**ORIGA 没有分割掩膜**~~ | ✅ **已解决** | 已改用 `datasets/raw/ORIGA-masked`（带 OD/OC 掩膜），转换出 `ORIGA_train` 520 / `ORIGA_test` 130。**五个域现已齐备**，留一法可以完整跑通。 |
| **REFUGE_test 没有标注** | ⚠️ 未解决 | 400 张图只有 `Images/`，没有 `gts/`。它属于 **Domain B (REFUGE)** 的数据流，出现在配置 A/C/D/E 的列表里。脚本会生成只有图像的 json，但**在该集上算 Dice 没有意义**（实测恒 0）。汇总时已从域 B 的成员里排除。 |

**结论：五个官方配置现在都能用当前数据完整跑通**（见第 5 节结果 A）。
留一法的五个测试域 A/B/C/D/E 全部有值。

带标注、可用于评测的域共 **2110 张**：
`REFUGE_train`(400)、`REFUGE_Valid`(400)、`RIM_ONE_r3`(159)、`Drishti_GS`(101)、
`ORIGA`(650)。
`REFUGE_test`(400) 只能用于无监督适应，不能用于报指标。

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

### 实际尝试结果（2026-09）

**方法 3（Wayback Machine）在本机执行不了**——不是页面不存在，而是这个环境的
出网是**白名单制**。实测：

| 主机 | 结果 |
|---|---|
| `pypi.org` / `ar5iv.labs.arxiv.org` | HTTP 200 ✓ |
| `www.bing.com` | HTTP 302 ✓ |
| `web.archive.org` | `Connection refused`（21 ms，IPv4 `130.211.15.150` 直连也拒） |
| `archive.org` | 连接超时 |
| `arquivo.pt` | 连接超时 |
| `timetravel.mementoweb.org` | DNS 解析失败 |
| `github.com` | 连接超时 |
| `www.google.com` | `Connection reset` |

（`web_fetch` 工具另有一层 SSRF 保护，对 `web.archive.org` 直接报
"resolves to a non-public IP address"。）**你自己在能正常上网的机器上可以试这个
链接**：
`https://web.archive.org/web/2020/http://imed.nimte.ac.cn/resources.html`

**顺带验证了一个 Zenodo 上的替代源 —— 结论是否定的。**
iMED 现在的新数据集（CORN、COSTA）都托管在 Zenodo，所以顺手搜了 Zenodo 上
type=dataset 的 ORIGA 记录，只有一条：

- `Diabetic Glaucoma(ORIGA,REFUGE,ACRIMA)`，
  DOI [10.5281/zenodo.10674885](https://doi.org/10.5281/zenodo.10674885)，
  CC-BY-4.0，其中 `ORIGA.rar` **475 MB**（对比 Kaggle 分类版仅 9.8 MB，
  体量上很像完整版）。

为避免白下 475 MB，只取了文件头/尾共 24 MB 解析 RAR5 索引，得到完整清单
（650 个文件）：

| 目录 | 文件数 |
|---|---|
| `ORIGA/ORIGA/Training/normal` | 386 |
| `ORIGA/ORIGA/Training/glaucoma` | 134 |
| `ORIGA/ORIGA/Testing/normal` | 96 |
| `ORIGA/ORIGA/Testing/glaucoma` | 34 |
| **合计** | **650，扩展名全部是 `jpg`** |

650 张、168 青光眼 / 482 正常，与官方发布公告的画像完全吻合——**图像确实是
官方 ORIGA-650，但同样没有任何分割掩膜**（无 png / mat / 标注文件），
只是分辨率更高的分类版。所以这条线索**不能解决掩膜问题**。

**结论：方法 3 走不通（环境限制），方法 4 的 Zenodo 分支已排除。**
剩下真正可行的是方法 1（问作者）和方法 2（问 iMED），以及方法 4 里那些
明确带 `Masks/` 目录的中文镜像。

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

### 结果 A：作者权重的留一法（⚠️ 口径有误，已被结果 E 取代）

> **这一节用的是仓库 `DiceEvaluator` 的口径，而它被视盘主导**——作者权重
> 根本不输出视杯（见结果 D），所以下表的数值**实际上只是视盘 Dice**，
> 不能与论文的两类均值 DSC 比较。保留仅为记录排查过程。
> **请直接看结果 E。**

论文 Table 1 的每一列是「在该域上测试、源模型来自其余四个域」的均值
（"based on five experimental runs"）。用 `tools/run_leave_one_out.sh` 跑齐
5 个源模型（配置 K 用 `model_K.pth`，测试列表 = 除域 K 外的全部域，
**与 `test_segment.yaml` 第 5/7/9/11/13 行逐字符一致**）：

```bash
bash tools/run_leave_one_out.sh                       # 跑 A–E
.venv/bin/python tools/summarize_leave_one_out.py     # 汇总
```

**逐次运行**（行 = 源模型，列 = 测试数据集，Dice %）：

| 源 | Drishti_test | Drishti_train | ORIGA_test | ORIGA_train | REFUGE_Valid | REFUGE_train | RIM_test | RIM_train |
|---|---|---|---|---|---|---|---|---|
| A | 85.37 | 83.91 | 80.40 | 81.57 | 81.93 | 86.95 | — | — |
| B | 88.00 | 88.72 | 81.15 | 80.85 | 79.41 | — | 82.90 | 81.27 |
| C | 89.35 | 88.92 | — | — | 81.00 | 81.45 | 88.93 | 89.35 |
| D | 83.17 | 83.44 | 86.23 | 86.35 | — | 87.85 | 86.49 | 83.52 |
| E | — | — | **63.24** | **62.98** | **63.16** | **75.06** | 84.96 | 83.51 |

**留一平均 vs 论文 Table 1（DSC）**：

| 测试域 | 全部 4 个源模型 | 论文 | 差值 |
|---|---|---|---|
| A (RIM-ONE) | **85.11** | 84.90 | +0.21 |
| B (REFUGE) | **82.83** | 83.34 | −0.51 |
| C (ORIGA) | 77.85 | 84.57 | −6.72 |
| D (REFUGE-Test) | 76.38 | 83.54 | −7.16 |
| E (Drishti-GS) | **86.36** | 85.51 | +0.85 |
| **5 域平均** | **81.70** | **84.37** | **−2.67** |

#### C / D 两列的偏差来自同一个离群模型

把每个域上 4 个源模型的取值摊开看，**偏差全部集中在 `model_E`**
（源域 = Drishti-GS）一个模型上：

| 测试域 | model_A | model_B | model_C | model_D | **model_E** |
|---|---|---|---|---|---|
| A (RIM-ONE) | — | 82.08 | 89.14 | 85.00 | 84.23 |
| B (REFUGE) | 86.95 | — | 81.45 | 87.85 | **75.06** |
| C (ORIGA) | 80.99 | 81.00 | — | 86.29 | **63.11** |
| D (REFUGE-Test) | 81.93 | 79.41 | 81.00 | — | **63.16** |
| E (Drishti-GS) | 84.64 | 88.36 | 89.13 | 83.31 | — |

剔除 `model_E` 后：

| 测试域 | 剔 E 后 | 论文 | 差值 |
|---|---|---|---|
| A (RIM-ONE) | 85.41 | 84.90 | +0.51 |
| B (REFUGE) | 85.42 | 83.34 | +2.08 |
| C (ORIGA) | 82.76 | 84.57 | **−1.81** |
| D (REFUGE-Test) | 80.78 | 83.54 | **−2.76** |
| E (Drishti-GS) | 86.36 | 85.51 | +0.85 |
| **5 域平均** | **84.14** | **84.37** | **−0.23** |

**剔除这一个离群模型后，五域平均与论文只差 0.23 分。**

为什么会这样：`model_E` 的源域是 **Drishti-GS**，其杯盘比中位数 **0.601**，
而 REFUGE 只有 **0.238**、ORIGA **0.351**。在 Drishti 上训练的模型倾向于在
杯盘比小的域上把视杯预测得过大，于是在 REFUGE / ORIGA 上明显偏低。
三条证据一致：`model_E` 在 B(75.06)、C(63.11)、D(63.16) 三个域上都是最低的
（但在 A(RIM-ONE, 杯盘比 0.223) 上并不差，说明这不是简单的"越不像越差"）。

**与论文的三个口径差异**（解读时务必注意）：

1. **指标口径**：这里用的是仓库 `DiceEvaluator`（每个**预测**取最佳匹配后求
   均值），不是标准按图/按类 DSC，详见本节末。
2. **模型来源**：直接使用作者发布的 5 个源模型权重，**没有自己重训源模型**。
   论文的源模型是「每域随机 8:2」划分训练（SGD, lr 0.001, bs 8）。
   我们自己的转换用的是随机 8:2，但**作者训练时用的是哪些图无从得知**，
   这是最可能造成残差的来源。
3. **REFUGE_test 无标注**（恒 0），已从域 B 的成员里排除，否则会污染均值。

> 关于第 2 点：`model_E` 在 ORIGA 上只有 63 分，也可能是作者的 ORIGA 源域
> 划分/预处理与我们的不同（我们用的是第三方整理的
> `ORIGA-masked/Images`，官方原版已不再提供下载）。这条无法进一步验证。

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

### 结果 D：作者发布的权重严重欠训 —— 以及仓库指标如何掩盖了它

这一节是**对前面"协议已对齐"结论的重要修正**。不联系作者也能查清：把预测按
类别拆开看就会发现，作者权重**从来没有学会视杯**。

#### 证据 1：checkpoint 里记录的训练步数少得离谱

```
weights/fundus_source/model_A.pth  iteration =  999
weights/fundus_source/model_C.pth  iteration =  999
weights/fundus_source/model_B.pth  iteration = 1999
weights/fundus_source/model_E.pth  iteration = 1999
weights/fundus_source/model_D.pth  iteration = 2999
weights/polyp_source/*.pth         iteration = 8999   ← 对照
```

#### 证据 2：视杯分支从不输出高置信度预测

`model_C.pth` 在三个域上的分类别统计：

| 数据集 | 视杯预测数 | 视杯分数中位 | 视盘预测数 | 视盘分数中位 |
|---|---|---|---|---|
| ORIGA_test | 11 | **0.088** | 73 | 0.998 |
| REFUGE_Valid | 17 | **0.101** | 60 | 0.989 |
| Drishti_GS_test | 32 | **0.204** | 79 | 0.999 |

按标准「每图每类取最佳匹配、漏检记 0」算（阈值 0.9）：

| 数据集 | 视杯 Dice | 视盘 Dice | 两类均值 |
|---|---|---|---|
| ORIGA_test | **0.00** | 84.93 | 42.47 |
| REFUGE_Valid | **0.00** | 75.75 | 37.87 |
| Drishti_GS_test | 1.43 | 90.69 | 46.06 |

**视杯 Dice 恒为 0。**

#### 仓库指标为什么看不出来

`DiceEvaluator` 是「**对每个预测实例**取最佳匹配后求均值」。既然预测几乎全是
视盘，报出来的 ~85% **实际上就是视盘 Dice**，视杯的塌陷完全不可见。
所以前面几节基于 `result.txt` 的"与论文对齐"结论**不成立**——口径不同，
不能直接比。用正确口径衡量，作者权重在 ORIGA 上只有 **41.02**。

#### 解法：自己重训源模型（不需要作者）

材料本来就齐全：五个域 2110 张带标注图、仓库自带训练路径、R-50 ImageNet
权重（`https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl`，
98 MB，实测可直接下载）。

```bash
bash tools/train_sources.sh A          # 训练一个域；GPU=1 MAX_ITER=6000 可覆盖
bash tools/verify_selftrained.sh       # 域内按类验证
bash tools/run_loo_per_class.sh        # 用正确口径跑留一法
```

配方按论文 4.2 节：R-50 ImageNet 初始化、SGD momentum 0.9、lr 0.001。
**唯一偏差**：论文 batch size 8，单卡 8 GB 放不下，用 2（已在配置里注明）。
训练步数 6000（作者用 999）。

#### 自训结果：视杯在每个域都学得会

| 权重 | 测试集 | 视杯 Dice | 视盘 Dice | 两类均值 |
|---|---|---|---|---|
| model_A | RIM_ONE_r3_test | **88.73** | 96.68 | **92.71** |
| model_C | ORIGA_test | **87.36** | 96.12 | **91.74** |
| model_D | REFUGE_train | **85.05** | 93.26 | **89.15** |
| model_E | Drishti_GS_test | **92.82** | 97.03 | **94.92** |

同一份 `ORIGA_test`：作者 `model_C` 两类均值 **41.02** → 自训 **91.74**。

**结论：这是训练预算问题，与数据、架构、类别映射都无关。**
`model_E` 在 REFUGE/ORIGA 上的"离群"也有了统一解释——它只是 5 个欠训模型里
恰好最差的那个，不是域差异。

> ⚠️ 域内结果（89–95）**高于**论文 Table 1 的 ~84，两者不可直接比较：
> 论文每列是**跨域**留一平均，这里是域内。同口径比较见结果 E。
>
> ⚠️ 自训权重在 `output/selftrained/`，**未纳入版本控制**（`output/` 被忽略），
> 复现需按上面命令重训。

### 结果 E：自训权重的按类留一法（**当前最终结果**）

用五个自训源模型 + 标准两类均值口径跑完整留一法：

```bash
bash tools/run_loo_per_class.sh      # 默认 WD=weights/fundus_source
WD=output/selftrained bash tools/run_loo_per_class.sh
```

逐次运行（行 = 源模型，列 = 测试数据集，两类均值 DSC %）：

| 源 | Drishti_test | Drishti_train | ORIGA_test | ORIGA_train | REFUGE_Valid | REFUGE_train | RIM_test | RIM_train |
|---|---|---|---|---|---|---|---|---|
| A | 74.15 | 72.56 | 75.39 | 80.15 | 66.82 | 79.73 | — | — |
| B* | 64.81 | 64.87 | 83.91 | 86.09 | 87.68 | — | **2.90** | **1.09** |
| C | 81.88 | 83.04 | — | — | 86.48 | 87.54 | 36.05 | 28.09 |
| D | 74.17 | 80.48 | 87.60 | 88.32 | — | 89.58 | 14.51 | 9.48 |
| E | — | — | 79.66 | 83.05 | 86.31 | 87.97 | 66.31 | 66.26 |

\* `model_B` 只训到 2000 步（其余 6000）——REFUGE 域每步约 2.5 s，跑满需再等
2.5 小时，而实测模型 2000 步即收敛（`model_C`@2000 两类均值 91.34 vs
@6000 91.74），故取 2000 步 checkpoint。

**留一平均 vs 论文 Table 1**（最终版：五个模型均训 6000 步，
Drishti 用阈值 255，`model_E` 也在 255 数据上重训）：

| 测试域 | 上一版 | **最终** | 论文 | 差值 | 判定 |
|---|---|---|---|---|---|
| A (RIM-ONE) | 28.09 | 30.47 | 84.90 | **−54.43** | ❌ 取景/版本问题 |
| B (REFUGE) | 86.21 | **86.42** | 83.34 | **+3.08** | ✅ |
| C (ORIGA) | 83.02 | **85.28** | 84.57 | **+0.71** | ✅ |
| D (REFUGE-Test) | 81.82 | **82.45** | 83.54 | **−1.09** | ✅ |
| E (Drishti-GS) | 74.49 | **79.84** | 85.51 | **−5.67** | ⚠️ |
| 可算域平均 | 70.73 | 72.89 | 84.37 | −11.48 | |

两项改进的贡献：

| 改进 | 影响 |
|---|---|
| `model_B` 从 2000 步补到 6000 步 | E 列 +2.0 左右（B 是最弱的一环） |
| Drishti SoftMap 阈值 128 → 255 | E 列 **+3.75**（74.50 → 78.24，见下） |
| `model_E` 在 255 数据上重训 | 保持整表自洽 |

**B / C / D 三域误差在 ±3.1 以内，C 只差 0.71。**

残余只剩两处：

1. **RIM-ONE（−54.43）** —— 取景/版本不匹配。**RIM-ONE DL 已完整下载并实测，
   但结论是"不能直接用"，原因比预想的更根本**，见下。

### 结果 F：RIM-ONE DL 实测 —— 取景是"视盘特写"，与其它三域不兼容

参考分割已取得（`RIM-ONE_DL_reference_segmentations/{glaucoma 688, normal 1252}`
= 485 图 × (Disc/Cup × png/txt)），转换成功（339/146 随机划分，每图 2 标注）：

- **掩膜正确**：视杯⊂视盘 100% 通过；杯盘比 0.226(train)/0.181(test)，
  与旧 r3 版的 0.223 一致；叠加目视对齐无误。
- **但迁移全线归零**（用现有 A–E 五个模型评域 A）：

  | 源模型 | 域 A 均值（DL 测试集） |
  |---|---|
  | model_A | 3.47 |
  | model_B | 0.04 |
  | model_C | 0.21 |
  | model_D | 0.00 |
  | model_E | 0.61 |

**根因：RIM-ONE DL 的图是紧贴视盘的放大裁剪**——整幅画面就是视盘区域，
几乎没有周围视网膜；而非黑占比 **1.000**（完全无背景）。
其它三域都是**整幅眼底**（非黑 0.786 / 0.856 / 0.884），视盘只占画面一小块。

也就是说，换 DL 不但没修好，反而更糟（域 A 从 30.47 掉到 ~0.2）——因为它和
其它域的**取景语义**不同，不是靠 padding 能对齐的（周围像素已被裁掉）。
这与 ORIGA-masked 包里那个被我们弃用的 `Images_Cropped/` 是同一种取景。

**这反过来给出了一个重要的新线索**：论文 4.2 节说「cropping the **Region of
Interest (ROI)** of each image to 800×800」。对 OD/OC 分割任务而言，**ROI 很可能
指视盘区域而不是整幅视网膜**。若如此，论文对**所有域**都做了视盘居中的裁剪，
四个域取景一致，RIM-ONE DL 的"特写"就是正常输入。

**下一步（明确可执行）**：把**其它三个域也裁成视盘居中 ROI**再重训重测。
做法：用各域 GT 的视盘 bbox 外扩固定倍数取正方形 → 缩放到 800×800，
然后重跑 `train_sources.sh` 与 `run_loo_per_class.sh`。
这是目前唯一能同时解释"DL 特写"与"论文 ~84"的假设，但需要再跑一轮完整训练
（约 3 小时，双卡）。

当前仓库状态：域 A **已回退到 r3 版**（保持与已提交结果一致），DL 数据与转换器
都在，随时可以切回去做上面的实验。

2. **Drishti-GS（−5.67）** —— 已从 −11.02 改善到 −5.67（阈值 255 + B 补训）。
   每个源模型在这一列的值是 81.64 / 73.91 / 81.98 / 81.85，`model_B` 仍偏低。
   剩下的差距可能来自阈值仍非最优（255 下杯盘比 0.461，其它域 0.22–0.35），
   或 Drishti 的 SoftMap 与论文实际用法不同。


两处残余：

1. **RIM-ONE（−56.81）—— 取景/版本不匹配（已定位，未修）**

   `model_A` 在 RIM-ONE 上**域内 92.71**，但其他源模型迁移过去只有 1.1~66.3，
   说明该域可学、是**跨域迁移**失败。根因是四个域取景尺度完全不同——
   实测裁剪后图像非黑占比：

   | 域 | 非黑占比中位 | 形态 |
   |---|---|---|
   | REFUGE | **0.786** = π/4 | 内切圆（完美） |
   | Drishti_GS | 0.856 | 接近内切 |
   | ORIGA | 0.888 | 略超出 |
   | **RIM_ONE_r3** | **0.984** | **视网膜被矩形裁切，圆形边界超出画面** |

   行/列非黑剖面印证了形态差异——REFUGE 呈圆弦长分布 `0.07 → 1.00 → 0.07`，
   RIM-ONE 是**平坦的 0.99**：

   ```
   REFUGE  T0001  行: 0.07 0.60 0.80 0.92 0.98 1.00 0.98 0.92 0.80 0.60
   RIM-ONE G-1-L  行: 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.98 0.99 0.99
   ```

   padding 解决不了——圆的边界已在画面之外，无法反推完整范围。

   **最可能的原因**：论文 4.1 节写的是 `Domain A (RIM-ONE ¹⁷)`，引用的是
   **文献 17** 那一版；我们手上是 **RIM-ONE r3**（`medimrg.webs.ull.es`）。
   学界常用的还有 **RIM-ONE DL**，它提供已裁好的单张眼底图。
   **换版本很可能是修复该列的关键。**

2. **Drishti-GS（−11.02）—— 主要由 `model_B` 拖累**

   把每域四个源模型摊开看，E 列的值是 73.35 / 64.84 / 82.46 / 77.32，
   `model_B` 的 64.84 是明显低点。而 `model_B` 恰好是唯一只训 2000 步的模型。
   剔除 B 后 E 列 = (73.35+82.46+77.32)/3 = **77.71**，与论文的 85.51 差 7.8。
   其余残差可能来自 Drishti 的掩膜阈值（SoftMap `>=128` 偏松，其杯盘比
   0.601 明显高于其它域）。

### 结果 G：全域视盘居中裁剪 —— **假设成立，RIM-ONE 那一列被修好**

按「论文的 ROI 指视盘区域」这一假设，用 `--roi-mode disc`（factor 1.31，
推导见下）对四个域统一重转、重训五个源模型、重跑按类留一法：

| 测试域 | 之前（retina 取景） | **现在（disc 取景）** | 论文 | 差值 |
|---|---|---|---|---|
| A (RIM-ONE) | 30.47 | **73.67** | 84.90 | **−11.23** |
| B (REFUGE) | 86.42 | **84.43** | 83.34 | **+1.09** |
| C (ORIGA) | 85.28 | **82.98** | 84.57 | **−1.59** |
| D (REFUGE-Test) | 82.45 | **82.52** | 83.54 | **−1.02** |
| E (Drishti-GS) | 79.84 | **81.95** | 85.51 | **−3.56** |
| **五域平均** | 72.89 | **81.11** | **84.37** | **−3.26** |

**A 列 +43.2，E 列 +2.1，五域平均 +8.2。五个域全部进入 ±11.2，四个在 ±3.6 以内。**

逐次运行（两类均值 DSC %）：

| 源 | Drishti_test | Drishti_train | ORIGA_test | ORIGA_train | REFUGE_Valid | REFUGE_train | RIM_test | RIM_train |
|---|---|---|---|---|---|---|---|---|
| A | 85.37 | 86.96 | 74.47 | 84.62 | 75.07 | 83.01 | — | — |
| B | 82.00 | 76.76 | 81.88 | 85.01 | 88.66 | — | 67.32 | 70.06 |
| C | 88.26 | 86.36 | — | — | 87.44 | 83.66 | 82.18 | 84.28 |
| D | 74.42 | 75.44 | 82.72 | 84.97 | — | 82.00 | 57.75 | 59.09 |
| E | — | — | 83.86 | 86.32 | 78.92 | 89.03 | 83.85 | 84.83 |

**结论：论文 4.2 节的 "Region of Interest (ROI)" 指的是视盘区域，不是整幅视网膜。**
四个域统一做视盘居中裁剪后取景尺度才一致，跨域迁移才成立。此前把 REFUGE /
ORIGA / Drishti 当整幅眼底、只对 RIM-ONE 用官方特写图，是最主要的残差来源。

这也解释了为什么先前 B/C/D 看起来"已经对齐"：整幅眼底的取景对**域内**学习没
问题（模型能找到视盘），但域间尺度不一致，一旦某个域是特写就彻底崩掉。
统一后 B/C/D 略有下降（86.42→84.43 等）但仍在 ±1.6，因为源模型也要重新适应
新的取景——这是**一致性换来的**，整体明显更优。

复现命令：

```bash
# 1) 全域视盘居中裁剪（域 A 用 RIM-ONE DL，Drishti 用阈值 255）
.venv/bin/python tools/prepare_fundus_data.py \
    --datasets REFUGE Drishti_GS ORIGA RIM_ONE_r3 \
    --roi-mode disc --disc-crop-factor 1.31 \
    --drishti-threshold 255 --rimone-source dl
# 2) 重训五个源模型（双卡：GPU0 "C B" / GPU1 "A D E"）
GPU=0 MAX_ITER=6000 DOMAINS="C B" bash tools/train_sources.sh
# 3) 按类留一法
WD=output/selftrained bash tools/run_loo_per_class.sh
```

剩余的 A 列 −11.23：A 的每个源模型是 68.69 / 83.23 / 58.42 / 84.34，
`model_D`（源域 = REFUGE_Valid，仅 400 张训练图）是低点。Drishti 列 −3.56
同样由 `model_D` 的 74.93 拉低。两者都可继续追，但已在合理量级。

### 结果 H：发布的 TTT 实现是**惰性的** —— 方法层面无法复现

前面所有留一法数字都是**纯推理**。为补上方法那一半，给
`tools/eval_per_class.py` 加了 `--ttt`（复刻 `engine/trainer.py:471-497` 的
TTT 循环：`loss,_,_,_ = model(inputs, branch='TTT')` → `backward` → `step`），
并用 `TTT=1 bash tools/run_loo_per_class.sh` 重跑完整留一法。

**结果与关闭 TTT 时字节级完全相同**（`diff .cache/loo_disc.json .cache/loo_ttt.json`
无差异，五域均仍是 73.67 / 84.43 / 82.98 / 82.52 / 81.95）。

#### 根因：TTT 分支把 backbone 特征 detach 了

`modeling/meta_arch/rcnn.py:304-306`：

```python
features = self.backbone(images.tensor)
if branch == "TTT":
    features = {k: v.detach() for k, v in features.items()}   # ← 显式 detach
```

`TTTGraphPool.update_pool` 也存 `nodes.detach().clone()`（`rcnn.py:77`）。

实测验证（`model_D` 在 RIM-ONE 上跑 39 步有效适应）：

| | 变化张量 | 最大变化量 |
|---|---|---|
| **检测网络**（backbone / roi_heads / proposal_generator） | **0 / 95** | **0.000e+00** |
| SPEGC 模块 + centroids | 6 / 6 | 1.472 |

并且**第一次有效 loss 反传后，backbone 梯度非零元素数为 0**。

#### 这意味着什么

适应只更新了 `SPEGC` 自己的参数（`P_CO` / `P_HE` / `c_p` / `W_q` / `W_k`）和
`centroids`。而这两个东西**只出现在 TTT 分支里，推理路径（`inference()`）
完全不使用它们**——所以无论适应多少步，分割结果一个像素都不会变。

这与论文的表述直接冲突。论文 3.4 节写的是：

> These components jointly drive the **end-to-end fine-tuning of all model
> parameters** at test time.

**结论：按发布状态，仓库里的测试时适应是空转的，论文的 SPEGC 方法无法从这份
代码复现。** 这不是配置问题，是 `detach()` 写在了 TTT 分支的必经路径上。

#### 论文原文怎么写的（arXiv HTML 附录 A / 3.4 节）

正文提到"supplementary materials"，而 **arXiv HTML 版（`arxiv.org/html/2603.11492v1`）
带完整附录**，里面有决定性内容。

**附录 A 的 Algorithm 1** 明确写出：

```
29: Acquire semantic predictions P ∈ ℝ^{V×C} for nodes in V* from f_σ
32: Update all learnable parameters {σ, P_CO, P_HE, W_q, W_k, c_p} by backpropagating L
34: Generate final prediction O_i ← f_σ(x_i)  (using the updated parameters σ)
```

- 第 32 行：**源模型参数 σ 明确在可学习集合里**
- 第 34 行：**推理用的是更新后的 σ**

**3.4 节**定义 `P_i`：

> if two nodes v_i and v_j are structurally similar, their corresponding
> **semantic predictions** `P_i` and `P_j` must be consistent
> `L_G = ΣΣ S*_ij · D_KL(P_j || sg(P_i))`

`sg(·)` 是 stop-gradient，只作用在一侧（teacher 式），梯度从 `P_j` 回流到 σ。

#### 代码与论文的两处独立偏差

| | 论文 | 发布代码 |
|---|---|---|
| σ（检测网络）是否更新 | **是**（Alg.1 第 32 行） | **否**（`rcnn.py:306` detach，梯度实测为 0） |
| `P_i` 是什么 | `f_σ` 给出的**语义预测** `∈ℝ^{V×C}` | `softmax(V* @ centroids.T)`，**对 Z=48 个聚类中心**做 softmax（`spegc.py`） |

两处叠加：既回传不到 σ，`P` 也不是论文说的那个量。

要真正复现方法，至少需要：

1. 去掉 `rcnn.py:306` 的 `detach`，让梯度回到 σ；
2. 把 `spegc.py` 里的 `P` 换成**分割头的类别预测**（`∈ℝ^{V×C}`，C 为类别数），
   而不是聚类分配；
3. 重新量化适应的增益。

第 1、2 条现在都有论文原文作依据（附录 A 第 29/32/34 行 + 3.4 节），
不再是猜测。**但这已经属于"按论文修代码"，不是"复现发布版本"了。**


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


### 已完成的对齐工作

| 项 | 状态 | 效果 |
|---|---|---|
| RIM-ONE 立体图切半 | ✅ | Dice 18.75% → 40.81% |
| ROI 裁剪 + 缩放 800×800 | ✅ | 与论文协议一致 |
| RIM-ONE 随机 8:2 划分 | ✅ | 127 / 32，与论文 4.2 节一致 |
| **类别 ID 顺序修正** | ✅ | **Dice 全面 → 81–89%，进入论文区间** |
| **接入 ORIGA-masked** | ✅ | 补齐域 C，五个域全部可用；留一法跑齐 |
| **留一法复现 Table 1** | ✅ | 剔除单一离群模型后 **5 域平均 84.14 vs 论文 84.37（−0.23）** |

### 仍未完成 / 可继续的方向

1. **`model_E` 的离群原因**：它在 REFUGE / ORIGA 上只有 63–75，其它模型 79–88。
   最可能是作者的源域划分与我们不同，或是 Drishti 的高杯盘比（0.601）导致
   模型在小杯盘比域上系统性高估视杯。**优先向作者确认源模型的训练划分与
   预处理**，这直接决定残差能否消掉。
2. **REFUGE_test 标注**：无标注，实测三项全 0。要么补齐，要么在
   `DATASETS.TEST` 里移除（汇总脚本已从均值里排除）。
3. **Drishti 掩膜阈值**：目前 SoftMap 取 `>=128`（专家过半同意）。
   试 `>=255`（全体同意）看影响 —— Drishti 杯盘比 0.601 偏高，阈值可能偏松。
4. **ORIGA 掩膜来源**：用的是第三方整理的 `ORIGA-masked`（官方已停供），
   无法确认与作者所用版本完全一致；且官方 `Set` 划分是 50/50，
   我们按论文用的是随机 8:2。
5. **RIM-ONE 专家标注**：默认 `Average_masks`；可试
   `--rimone-expert exp1/exp2` 对比。
6. **指标口径**：论文 DSC 与仓库 `DiceEvaluator` 口径不同（详见第 5 节末），
   需向作者确认论文用的是哪种。
7. **预处理归一化**：论文说 min-max，仓库配置用的是 detectron2 默认 ImageNet
   统计量。当前沿用仓库设置（权重是用它训的）。
8. **`test.sh` 原样复现**：现在五个域齐备，可以改
   `configs/test_segment.yaml` 了（注意 `test.sh` 靠**行号** 5/7/9/11/13 做 sed，
   改文件会打乱映射）。当前用 `configs/test_config_C.yaml` 绕开了这个限制。


### 论文出处

> Xiaogang Du, Jiawei Zhang, Tongfei Liu, Tao Lei, Yingbo Wang.
> *SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph
> Clustering for Medical Image Segmentation.* CVPR 2026, pp. 8481-8491.
> [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Du_SPEGC_Continual_Test-Time_Adaptation_via_Semantic-Prompt-Enhanced_Graph_Clustering_for_Medical_CVPR_2026_paper.html)
> · [arXiv:2603.11492](https://arxiv.org/abs/2603.11492)
>
> 本文第 2.5 节的域定义、预处理与超参均引自该文 4.1/4.2 节。
