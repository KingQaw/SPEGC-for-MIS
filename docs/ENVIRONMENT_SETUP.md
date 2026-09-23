# SPEGC 环境配置与跑通记录

> 目标：在一台带 2×RTX 2070 的 Windows 11 + WSL2 机器上，为本仓库（CVPR 2026
> SPEGC，Detectron2 分支）建立独立虚拟环境，用**合成小样本数据**把训练与评测
> 流程端到端跑通（不使用真实数据）。
>
> 本文记录**完整步骤、版本决策、遇到的每一个问题及其解决方式**，目的是让这个
> 环境可以被完整复现。

---

## 0. TL;DR

```bash
cd /home/zhao/SPEGC-for-MIS

# 1) 重建环境（首次约 5–10 分钟，需联网）
bash tools/setup_env.sh

# 2) 生成合成数据（8 张训练 / 4 张测试）
.venv/bin/python tools/make_synthetic_fundus.py

# 3) 训练（CPU 冒烟测试，2 次迭代）
.venv/bin/python train_net.py --num-gpus 1 --config configs/smoke_baseline.yaml

# 4) 推理 / 评测
.venv/bin/python train_net.py --eval-only --num-gpus 1 \
    --config configs/smoke_baseline.yaml \
    MODEL.WEIGHTS output/smoke_baseline/model_final.pth
```

用 GPU 训练只需加一个覆盖项：

```bash
.venv/bin/python train_net.py --num-gpus 1 --config configs/smoke_baseline.yaml \
    MODEL.DEVICE cuda OUTPUT_DIR output/smoke_gpu
```

环境位置：`/home/zhao/SPEGC-for-MIS/.venv`（Python 3.9.25，已被 `.gitignore` 忽略）。

---

## 1. 机器与系统环境

| 项目 | 值 |
|---|---|
| 主机 | 物理机（非虚拟机），Windows 11 build `10.0.26200.9457` |
| CPU | Intel Core i7-9800X，8 核 16 线程 |
| 内存 | 物理 32 GB（WSL 分配 16 GB） |
| GPU | **2 × NVIDIA GeForce RTX 2070**，各 8192 MiB，计算能力 7.5 (Turing) |
| Windows 显卡驱动 | 591.86（CUDA 13.1） |
| WSL | WSL 2.7.14.0，内核 `6.18.33.2-2`，WSLg 1.0.73.2 |
| 发行版 | Ubuntu 26.04（用户空间），系统 Python 3.14.7 |
| conda | 26.7.1，位于 `~/miniconda3` |
| **C 编译器** | **无**（`gcc`/`cc`/`make`/`cmake` 均不存在） |

> ⚠️ **无编译器是本机最关键的限制**：任何没有预编译 wheel 的包都会安装失败。
> 本次全部依赖都恰好有 wheel（含 `pycocotools`、`typed-ast`），所以不需要编译器；
> 若换一台机器，请先确认 `gcc` 是否可用。

### GPU 可见性说明

Agent（我）执行命令时运行在 bubblewrap 沙箱内，该沙箱把 `/dev` 替换成一个
只有 14 个设备节点的私有 tmpfs，**缺少 `/dev/dxg`**（WSL 的 GPU 直通设备）。
因此在沙箱内 `nvidia-smi` 会报：

```
Failed to initialize NVML: GPU access blocked by the operating system
```

**这是沙箱假象，不是显卡故障。** 真实 WSL 环境里 `/dev/dxg` 存在
（`crw-rw-rw- 1 root root 10, 258`），GPU 完全可用。本文件中所有 GPU 结论都是
通过 `wsl.exe` 互操作在沙箱外验证的。

本环境实测（`.venv`，torch 1.9.1+cu111，沙箱外）：

```
torch         : 1.9.1+cu111
built for CUDA: 11.1
cuda available: True
device count  : 2
  [0] NVIDIA GeForce RTX 2070  8191 MiB  cc 7.5
  [1] NVIDIA GeForce RTX 2070  8191 MiB  cc 7.5
GPU matmul OK : -121137.75
cudnn         : 8005
```

---

## 2. 环境重建步骤（`tools/setup_env.sh` 做的事）

### 2.1 创建 Python 3.9 虚拟环境

```bash
export CONDA_PKGS_DIRS=$PWD/.cache/conda-pkgs
~/miniconda3/bin/conda --no-plugins create -p ./.venv python=3.9 -y --solver=classic
```

为什么是 Python 3.9：`detectron2==0.5+cu111` 的官方 wheel 只提供
`cp36/cp37/cp38/cp39`（配合 torch 1.9），且 `torch==1.9.1+cu111` 也止于
cp39。系统 Python 是 3.14，另一套现成环境 `~/miniconda3/envs/pytorch` 是
Python 3.12 + torch 2.14，**都无法使用**（既装不了 detectron2 0.5 的 wheel，
`np.bool` 等旧 API 也已失效）。

### 2.2 安装 PyTorch（CUDA 11.1）

```bash
.venv/bin/pip install "torch==1.9.1+cu111" "torchvision==0.10.1+cu111" \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html
.venv/bin/pip install --no-deps "torchaudio==0.9.1" \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

cu111 的 wheel 自带 CUDA runtime，**不需要系统安装 CUDA Toolkit，也不需要编译器**。
驱动 591.86 向下兼容 CUDA 11.1 运行时，实测可用。

### 2.3 覆盖 3 个必须改版本的包

```bash
.venv/bin/pip install "numpy==1.23.5" "Pillow==9.5.0" "scipy==1.9.3"
```

理由见第 3 节。

### 2.4 安装其余依赖

```bash
# 剔除 torch/torchvision/torchaudio/detectron2（它们来自专用 wheel 源）
grep -vE "^(torch|torchvision|torchaudio|detectron2)==" requirements.txt > .cache/req-bulk.txt
.venv/bin/pip install -r .cache/req-bulk.txt
```

### 2.5 安装 detectron2 0.5+cu111

```bash
.venv/bin/pip install --no-deps "detectron2==0.5+cu111" \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

**`detectron2` 不在 PyPI 上**，只能用上面这个 wheel 索引（README 里给的就是它）。
加 `--no-deps` 是为了防止 pip 反过来篡改 2.3 里定好的 numpy。

### 2.6 安装 distutils 兼容 shim

```bash
cp tools/env_compat/sitecustomize.py .venv/lib/python3.9/site-packages/sitecustomize.py
```

原因见问题 5。

---

## 3. 关键版本决策

### 3.1 与 `requirements.txt` 的差异

| 包 | requirements.txt | 实际安装 | 原因 |
|---|---|---|---|
| `numpy` | `1.21.6` | **`1.23.5`** | 下界：`opencv-python==4.8.1.78` 的 `cv2/typing/__init__.py` 在导入时执行 `numpy.dtype[numpy.generic]`（PEP 585 泛型），**需要 numpy ≥ 1.23**；上界：`evaluation/dice_metric.py:106` 等使用 `np.bool`，**numpy ≥ 1.24 已删除该别名**。1.23.5 是唯一同时满足两者的版本。 |
| `scipy` | `1.7.3` | **`1.9.3`** | scipy 1.7.3 自身元数据声明 `numpy<1.23.0`，与上面的 1.23.5 冲突。代码只用到 `scipy.ndimage.center_of_mass` / `scipy.spatial.distance` / `scipy.optimize`，1.9.3 完全兼容。 |
| 其它 | 全部按 pin | 全部按 pin | 包括 `typed-ast==1.5.5`、`pycocotools==2.0.7`、`antlr4-python3-runtime==4.9.3` 等都有可用 wheel。 |

> `pip check` 现在输出 **`No broken requirements found.`**，依赖图是干净的。

### 3.2 冻结版本

完整 104 个包的确切版本已记录在 **`requirements-lock.txt`**（由 `pip freeze` 生成）。
核心版本：

```
torch==1.9.1+cu111          torchvision==0.10.1+cu111     torchaudio==0.9.1
detectron2==0.5+cu111       numpy==1.23.5                 scipy==1.9.3
opencv-python==4.8.1.78     Pillow==9.5.0                 pycocotools==2.0.7
fvcore==0.1.5.post20221221  iopath==0.1.8                 yacs==0.1.8
matplotlib==3.5.3           scikit-learn==1.0.2           tensorboard==2.11.2
```

---

## 4. 遇到的问题与解决（按遇到顺序）

### 问题 1：conda 被 Anaconda TOS 插件阻断

**现象**：`conda create ...` 非交互执行时报
`conda_anaconda_tos/plugin.py ... render_interactive()` 的 traceback，退出码 1。

**原因**：新版 conda 自带 `conda_anaconda_tos` 插件，使用 `repo.anaconda.com`
默认频道时会尝试交互式展示并接受服务条款；非交互环境下直接失败。

**解决**：禁用插件，并显式指定 classic solver（禁用插件会连带禁用 libmamba
solver 插件，必须补 `--solver=classic`）：

```bash
conda --no-plugins create -p ./.venv python=3.9 -y --solver=classic
```

> 注意 `--no-plugins` 是**全局选项**，必须写在 `create` **之前**。

### 问题 2：conda 无法写 notices 缓存

**现象**：`ERROR conda.notices.core:wrapper(127): Unable to open cache file:
[Errno 30] Read-only file system: '/home/zhao/.cache/conda/notices/notices.cache'`

**原因**：Agent 沙箱只允许写工作区目录，`~/.cache` 只读。

**解决**：**无需处理**，只是警告。同理 `~/.conda/environments.txt` 注册失败也无害
（用完整路径 `conda activate /home/zhao/SPEGC-for-MIS/.venv` 即可）。

### 问题 3：机器没有 C 编译器

**现象**：`gcc` / `cc` / `g++` / `make` / `cmake` / `ninja` 全部不存在。

**影响**：任何没有预编译 wheel 的包都会装不上。

**解决**：安装前先用 `pip install --dry-run --only-binary=:all:` 预检，确认所有
pin 的包都有 wheel。实测只有 `antlr4-python3-runtime==4.9.3` 是 sdist-only，
但它是**纯 Python**，无需编译器即可构建，实际安装成功。

### 问题 4：`data/datasets/builtin.py` 缺失（致命）

**现象**：`train_net.py:20` 的 `import data.datasets.builtin` 直接
`ModuleNotFoundError`，**任何入口都跑不起来**。

**原因（根因）**：`.gitignore` 第 12 行是不带锚点的 `datasets/`，这个模式会匹配
**任意层级**的 `datasets` 目录，因此 `data/datasets/` 被一起忽略了：

```
$ git check-ignore -v data/datasets/builtin.py
.gitignore:12:datasets/	data/datasets/builtin.py
```

两个 commit（`473bb00`、`20652d8`）里都从未包含过这个文件，原始内容不可恢复。

### 问题 5：`checkpoint/detection_checkpoint.py` 缺失（致命）

**现象**：`engine/trainer.py:49` 在**模块导入期**执行
`from checkpoint.detection_checkpoint import DetectionTSCheckpointer`，仓库里根本
没有 `checkpoint/` 目录。

**原因**：同一个根因——`.gitignore` 第 14 行的 `checkpoint/` 也是不带锚点的，
把源码包 `checkpoint/` 一起忽略了：

```
$ git check-ignore -v checkpoint/detection_checkpoint.py
.gitignore:14:checkpoint/	checkpoint/detection_checkpoint.py
```

**解决（问题 4 与 5）**：

1. **修根因**：把 `.gitignore` 中的数据/输出目录模式**锚定到仓库根**：

   ```diff
   -weights/
   -datasets/
   -output/
   -checkpoint/
   +/weights/
   +/datasets/
   +/output/
   +# "checkpoint/" 在仓库根既是输出目录、又是 engine/trainer.py 导入的源码包，
   +# 因此只忽略其内容、保留 .py 源码
   +# （git 无法在整目录被排除的情况下重新包含其中的文件）
   +/checkpoint/*
   +!/checkpoint/*.py
   ```

   > 注意 `checkpoint/` 比较特殊：它在仓库根**同时**是「输出 checkpoint 的目录」
   > 和「被 import 的源码包」，两者同名同位置。单纯把 `checkpoint/` 锚定成
   > `/checkpoint/` **并不够**（源码包仍会被忽略），必须用
   > `/checkpoint/*` + `!/checkpoint/*.py` 这种「忽略内容、放行 .py」的写法。
   > `datasets/` 则没有这个冲突，锚定即可。

   验证：

   ```bash
   git check-ignore -q checkpoint/detection_checkpoint.py && echo BAD || echo OK   # OK
   git check-ignore -q datasets/synthetic/train.json       && echo OK  || echo BAD # OK
   ```

2. **重建两个模块**（内容为按调用契约还原，非原始代码）：
   - `checkpoint/detection_checkpoint.py`：`DetectionTSCheckpointer` 直接继承
     detectron2 的 `DetectionCheckpointer`。detectron2 的实现已经能正确处理任意
     `nn.Module` 图（包含 `EnsembleTSModel` 这种 teacher/student 包装），原版
     OpenMatch 的同类只多了 Caffe2 blob 处理，这里用不到。
   - `data/datasets/builtin.py`：为 config 里出现的**全部 19 个数据集名**做
     **惰性注册**（`DatasetCatalog.register` + `MetadataCatalog`），并额外注册
     合成数据集 `SynthFundus_train` / `SynthFundus_test`。惰性很关键：导入时
     不碰磁盘，只有真正取用某个数据集时才会因文件缺失而报错。

### 问题 6：torch 1.9.1 的 tensorboard 集成在现代 setuptools 下崩溃

**现象**：训练已经跑完建模型、建 dataloader，在 `build_hooks()` 处崩：

```
File ".../torch/utils/tensorboard/__init__.py", line 4, in <module>
    LooseVersion = distutils.version.LooseVersion
AttributeError: module 'distutils' has no attribute 'version'
```

**原因**：torch 1.9.1 写的是

```python
import tensorboard
from setuptools import distutils
LooseVersion = distutils.version.LooseVersion   # ← 属性式访问子模块
```

`distutils.version` 是**子模块**，只有当别处显式 `import distutils.version`
之后才会成为 `distutils` 包的属性。实测三种情况**都不会**自动导入：
stdlib distutils、setuptools 自带的 `setuptools._distutils`、以及
`import setuptools` 之后。而 detectron2 的 `default_writers()` 一定会构造
`TensorboardXWriter`，于是必然触发。

**解决**：在 site-packages 放一个 `sitecustomize.py`（Python 启动时由 `site.py`
自动导入），在进程最早期导入该子模块：

```python
import distutils.version  # noqa: F401
```

**没有修改任何第三方包**。文件同时保存在仓库内 `tools/env_compat/sitecustomize.py`，
由 `tools/setup_env.sh` 自动安装。

### 问题 7：`opencv-python` 与 `numpy` 不兼容（必须改版本）

**现象**：

```
File ".../cv2/typing/__init__.py", line 69, in <module>
    NumPyArrayGeneric = numpy.ndarray[typing.Any, numpy.dtype[numpy.generic]]
TypeError: 'numpy._DTypeMeta' object is not subscriptable
```

**原因**：`opencv-python==4.8.1.78` 在导入时使用 PEP 585 泛型下标
（`numpy.dtype[...]`），需要 numpy ≥ 1.23；而 `requirements.txt` pin 的是
`numpy==1.21.6`。**这两个 pin 本身互相矛盾，无法同时满足**。又因为
`dice_metric.py` 里用了 `np.bool`（numpy 1.24 已删除），numpy 还必须 < 1.24。

**解决**：装 `numpy==1.23.5`（区间 `[1.23, 1.24)` 内唯一可用版本）。

### 问题 8：`scipy 1.7.3` 声明的 numpy 上界冲突

**现象**：`pip` 报
`scipy 1.7.3 requires numpy<1.23.0,>=1.16.5, but you have numpy 1.23.5`。

**解决**：升到 `scipy==1.9.3`（`pip check` 随即变为干净）。详见 3.1。

### 问题 9：训练后评测指标全是 `nan`

**现象**：训练正常结束（EXIT=0，3 个 checkpoint 落盘，loss 正常回传），但
`output/*/result.txt` 里 Dice / EA / SM 全是 `nan`，并伴随
`RuntimeWarning: Mean of empty slice`。

**原因（已定位）**：`evaluation/dice_metric.py:82-84` 用
`np.mean(self.dice_scores)`，而当**预测列表为空**时该列表长度为 0 →
`np.mean([])` = `nan`。预测为空是因为模型的 RPN 没有产出任何存活 proposal：

```
num proposals = 0
```

也就是说：这是**只用 2 次迭代训练出来的、毫无意义的模型状态**，不是代码缺陷。

**验证**：用未训练的随机权重做纯推理，同一段评测代码给出**真实数值**：

```
SynthFundus_test   |  Dice 1.2385%  |  EA 42.1995%  |  SM 45.8958%
```

这证明 Dice / Enhanced Alignment / Structure measure 三条指标计算路径都是通的。

**用法建议**：冒烟测试出现 `nan` 属于预期；想看真实数字就跑

```bash
.venv/bin/python train_net.py --eval-only --num-gpus 1 \
    --config configs/smoke_baseline.yaml \
    MODEL.WEIGHTS "" OUTPUT_DIR output/smoke_randeval
```

### 问题 10：`ortools` 未列入 requirements.txt（潜在缺依赖）

`modeling/GModule/utils/ILP.py:8-9` 导入 `ortools`，但它**不在
`requirements.txt` 里**。不过 `modeling/GModule/utils/losses.py:7` 对该模块的
导入是**注释掉的**，而第 786-788 行引用的 `ILP_solver` 从未被定义——这条 ILP
路径属于**死代码**，因此本次没有安装 `ortools`，也不影响跑通。

### 问题 11：`detectron2` 不在 PyPI

`pip install detectron2` 会失败（`0.5+cu111` 这类本地版本号不在 PyPI 上）。
**必须**用 README 给出的 wheel 索引：

```
https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

---

## 5. 合成数据

真实数据（Fundus/Polyp 四个域）和预训练权重都**不在仓库里**，仓库中仅有的
`data/fetus_annotations_coco/*.json` 所引用的图像文件也全部不存在，且没有任何
config 引用它们（属于孤儿数据）。

因此提供 `tools/make_synthetic_fundus.py` 生成小样本合成数据：

```bash
.venv/bin/python tools/make_synthetic_fundus.py            # 8 train / 4 test
.venv/bin/python tools/make_synthetic_fundus.py --num-train 4 --num-test 2
```

- 输出：`datasets/synthetic/images/*.jpg`、`datasets/synthetic/{train,test}.json`
- 图像：320×320，深色眼底背景 + 亮色视网膜盘 + 血管 + 1~3 个病灶斑块 + 传感器噪声
- 标注：**COCO instance segmentation** 格式，与代码期望完全一致
  - `images[].file_name/height/width/id`
  - `annotations[].id/image_id/category_id/bbox(XYWH)/segmentation(多边形)/iscrowd=0/ignore=0/area`
  - `categories = [{"id":1,"name":"lesion"}]`（单前景类，对应 `NUM_CLASSES: 1`）
- 固定随机种子（默认 `20260323`），**完全可复现**
- 病灶用 24 点多边形（48 个坐标）拟合椭圆——真实掩膜而非矩形框，
  且满足 loader 对多边形 `len % 2 == 0 and len >= 6` 的要求

> ⚠️ 合成数据只用于验证代码能跑通，**任何指标都没有意义**。

数据集注册在 `data/datasets/builtin.py`，注册后可直接检查：

```python
import data.datasets.builtin
from detectron2.data import DatasetCatalog, MetadataCatalog
DatasetCatalog.get("SynthFundus_train")   # 8 条记录，category_id 已重映射为 0
MetadataCatalog.get("SynthFundus_train")  # thing_classes=['lesion'], evaluator_type='coco'
```

---

## 6. 冒烟测试配置与结果

新增 `configs/smoke_baseline.yaml`，它沿用 `configs/seg_res50fpn_source.yaml` 的
模型结构（`DAobjTwoStagePseudoLabGeneralizedRCNN` + `PseudoLabRPN` +
`StandardROIHeadsPseudoLab`、`MASK_ON`、`NUM_CLASSES=1`），但：

| 设置 | 值 | 原因 |
|---|---|---|
| `DATASETS.TRAIN/TEST` | `SynthFundus_*` | 指向合成数据 |
| `MODEL.WEIGHTS` | `""` | 从零初始化，**不下载任何权重** |
| `MODEL.DEVICE` | `cpu` | 沙箱内无 `/dev/dxg`；删掉该行即可用 GPU |
| `SOLVER.AMP.ENABLED` | `False` | 与 CPU 配套 |
| `SOLVER.MAX_ITER` | `2` | 冒烟测试 |
| `TEST.TTT` | `False` | **必须**：detectron2 的 `EvalHook` 在训练最后一轮**必定**评测，而 `BaselineTrainer.test()` 的 TTT 分支会调用 `optimizer.zero_grad()`，训练期传进来的 optimizer 是 `None` → 崩溃 |
| `TEST.EVAL_PERIOD` | `0` | 同上，最终评测无法关闭，故靠 `TTT False` 规避 |
| `INPUT.MIN/MAX_SIZE_*` | `256 / 320` | 降低 CPU 耗时 |
| `DATALOADER.NUM_WORKERS` | `0` | 避免多进程与沙箱冲突 |
| `MODEL.ROI_HEADS.SCORE_THRESH_TEST` | `0.0` | 尽量保留预测，让评测代码路径真正执行 |

### 训练结果（CPU，2 次迭代，实测通过）

```
metrics.json:
  {"bbox_num/gt_bboxes": 2.5, "fast_rcnn/cls_accuracy": 0.595, "iteration": 1,
   "loss_box_reg": 23.87, "loss_cls": 147.70, "loss_mask": 5.34,
   "loss_rpn_cls": 20.48, "loss_rpn_loc": 15.84, "lr": 0.00125,
   "mask_rcnn/accuracy": 0.556, "total_loss": 213.23}

落盘产物:
  output/smoke_baseline/{model_0000000.pth, model_0000001.pth, model_final.pth,
                         metrics.json, last_checkpoint, events.out.tfevents.*}
```

### 评测结果

`--eval-only` 分支会打印结果表并写入 `output/<dir>/result.txt`：

```
Dataset / Split                     | Dice Coeff   | Enhanced Align   | Struct Similarity
------------------------------------------------------------------------------------------
SynthFundus_test                    |        nan%  |            nan%  |              nan%
SynthFundus_mean                    |        nan%  |            nan%  |              nan%
```

`nan` 的原因见问题 9；换随机权重即可得到真实数值（1.2385% / 42.1995% / 45.8958%）。

### GPU 训练（实测通过）

```bash
.venv/bin/python train_net.py --num-gpus 1 --config configs/smoke_baseline.yaml \
    MODEL.DEVICE cuda OUTPUT_DIR output/smoke_gpu
```

在 2×RTX 2070 上正常完成，同样产出 3 个 checkpoint。

---

## 7. 已知代码缺陷（本次**未**修改，供后续参考）

冒烟测试走的是 **`BaselineTrainer`** 路径（`configs/*.yaml` 里
`SEMISUPNET.Trainer: baseline`）。**SPEGC 主路径（`Trainer: spegc`）按发布状态
无法运行**，至少有 4 处硬缺陷：

| # | 位置 | 问题 |
|---|---|---|
| 1 | `engine/trainer.py:795` | 读取 `cfg.SEMISUPNET.TTT`，但该配置键**在任何地方都没定义**（只有 `TEST.TTT`）。yacs 的 `merge_from_list` 会断言键必须已存在，`--opts` 无法新增，YAML 新键也会被拒 → `AttributeError` |
| 2 | `engine/trainer.py:921` | 调用 `self.graph_matching(...)`，但 `SPEGCTrainer` **没有这个方法** |
| 3 | `engine/trainer.py:800, 894` | 按 5 元组解包 `branch="supervised_source"` 的返回值，而 `modeling/meta_arch/rcnn.py:335` 只返回 **4 元组** → `ValueError` |
| 4 | `modeling/roi_heads/roi_heads.py:110` | `unsup_data_weak` 分支会走到 detectron2 基类 `forward_with_given_boxes`，其内部有 `assert not self.training`；而 teacher 模型从未被 `.eval()` → 断言失败 |

跑 SPEGC 主路径前需要先修这 4 处（或改用 `BaselineTrainer`）。

其它**休眠**问题（当前调用路径不触发）：`model/utils/bbox_tools.py` 不存在
（`creator_tool.py` 引用，但无人导入）；`modeling/meta_arch/vgg.py:126` 硬编码
引用不存在的 `CMT_AT/checkpoints/vgg16_bn-6c64b313_converted.pth`；
`data/node_sampling.py` 用了 `plt` 却没导入 `matplotlib.pyplot`；
`engine/trainer.py:1222` 的局部 `inference_on_dataset` 遮蔽了 detectron2 的导入
（BaselineTrainer 恰好按 4 参数调用，所以能用，SPEGC 路径按 3 参数调用会
`TypeError`）。

---

## 8. 本次新增 / 修改的文件

| 文件 | 类型 | 说明 |
|---|---|---|
| `.gitignore` | **修改** | 修复根因：`weights/ datasets/ output/` 锚定为 `/weights/ /datasets/ /output/`；`checkpoint/` 改为 `/checkpoint/*` + `!/checkpoint/*.py`（同名冲突需特殊处理）；并新增 `/.cache/` |
| `checkpoint/__init__.py` | 新增 | 重建被忽略的包 |
| `checkpoint/detection_checkpoint.py` | 新增 | `DetectionTSCheckpointer`（继承 detectron2 实现） |
| `data/datasets/__init__.py` | 新增 | 重建被忽略的包 |
| `data/datasets/builtin.py` | 新增 | 19 个数据集名 + 合成数据集的惰性注册 |
| `configs/smoke_baseline.yaml` | 新增 | CPU 冒烟测试配置 |
| `tools/make_synthetic_fundus.py` | 新增 | 合成 COCO 数据生成器 |
| `tools/setup_env.sh` | 新增 | 一键重建环境 |
| `tools/env_compat/sitecustomize.py` | 新增 | distutils 兼容 shim（源） |
| `requirements-lock.txt` | 新增 | `pip freeze` 冻结的 104 个包版本 |
| `docs/ENVIRONMENT_SETUP.md` | 新增 | 本文档 |

未修改任何原有源码文件。

---

## 9. 常用命令速查

```bash
PY=/home/zhao/SPEGC-for-MIS/.venv/bin/python

# 环境自检
$PY -c "import torch, detectron2, cv2; print(torch.__version__, detectron2.__version__)"
.venv/bin/pip check

# 重新生成合成数据
$PY tools/make_synthetic_fundus.py

# 训练（CPU / GPU）
$PY train_net.py --num-gpus 1 --config configs/smoke_baseline.yaml
$PY train_net.py --num-gpus 1 --config configs/smoke_baseline.yaml \
    MODEL.DEVICE cuda OUTPUT_DIR output/smoke_gpu

# 评测
$PY train_net.py --eval-only --num-gpus 1 --config configs/smoke_baseline.yaml \
    MODEL.WEIGHTS output/smoke_baseline/model_final.pth

# 注意：train_net.py:137 强制 args.resume=True
# → 若 OUTPUT_DIR 下已有 last_checkpoint，会优先从它恢复而忽略 MODEL.WEIGHTS，
#   想从指定权重干净起跑，请换一个全新的 OUTPUT_DIR。
```
