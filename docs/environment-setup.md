# 环境安装记录

## 1. 目标

为项目创建独立 conda 环境，并安装可用的 GPU 版 PyTorch 与当前阶段所需依赖，避免和系统 Python 混用。

## 2. 创建的 conda 环境

环境名：

```text
industrial-defect-diffusion
```

创建命令：

```powershell
conda create -n industrial-defect-diffusion python=3.10
```

激活命令：

```powershell
conda activate industrial-defect-diffusion
```

## 3. 为什么这样做

这样做的原因：

1. 系统 Python 版本是 3.13，不适合直接作为深度学习主环境。
2. Diffusers、PyTorch、Anomalib、Ultralytics 等库在 Python 3.10 上更稳。
3. 单独环境可以避免和其他项目依赖冲突。

## 4. 硬件与驱动检查

已确认本机有 NVIDIA GPU，并且驱动可用。

检测到的 GPU：

```text
NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

`nvidia-smi` 显示 CUDA 驱动版本：

```text
CUDA Version: 13.1
```

## 5. 已安装核心依赖

### 5.1 GPU 深度学习底座

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
torchaudio 2.11.0+cu128
```

### 5.2 基础科学计算与图像处理

```text
numpy
pillow
matplotlib
pandas
tqdm
scikit-learn
seaborn
opencv-python
pyyaml
ipykernel
```

### 5.3 生成模型相关

```text
diffusers
transformers
accelerate
safetensors
huggingface_hub
```

## 6. 验证结果

PyTorch GPU 验证结果：

```text
torch 2.11.0+cu128
cuda_available = True
cuda_version = 12.8
device_count = 1
device_name = NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

说明：

```text
GPU 版 PyTorch 已经能够正确识别显卡。
```

## 7. 当前结论

当前环境已经足以支持：

1. 数据探索
2. 传统伪缺陷生成
3. 后续 Diffusion Inpainting 生成实验
4. 后续 PyTorch 分割训练

## 8. 后续可选安装

后续如果要做异常检测 baseline，可再安装：

```text
anomalib
```

后续如果要做 YOLOv8 分割，可再安装：

```text
ultralytics
```

这些不必现在就装，等项目推进到对应阶段再补更合适。

