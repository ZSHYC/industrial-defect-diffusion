# 第 4 阶段：PatchCore 风格无监督异常检测 Baseline

## 1. 本阶段目标

本阶段目标是建立一条真实测试集上的异常检测评估基线。

前 3 个阶段已经完成了：

```text
第 1 阶段：确认 MVTec AD tile 数据结构
第 2 阶段：生成传统规则伪缺陷
第 3 阶段：生成 Diffusion Inpainting 伪缺陷
```

但是到第 3 阶段为止，项目只能说明：

```text
我们能生成一些缺陷图。
```

还不能说明：

```text
这些生成图是否真的能帮助工业缺陷检测。
```

所以第 4 阶段先做一个不使用生成数据的 baseline，用来回答：

```text
如果完全不用生成缺陷，只用 train/good 正常图训练，一个经典无监督异常检测方法在真实 test 集上能做到什么水平？
```

这个结果会成为后续第 5 阶段监督分割和数据增强实验的对照线。

## 2. 本阶段做了什么

本阶段新增脚本：

```text
scripts/04_patchcore_baseline.py
```

运行命令：

```powershell
conda run -n industrial-defect-diffusion python scripts/04_patchcore_baseline.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --image-size 224 --seed 42
```

本阶段输入数据：

```text
MVTec_AD/tile/train/good
MVTec_AD/tile/test/good
MVTec_AD/tile/test/crack
MVTec_AD/tile/test/glue_strip
MVTec_AD/tile/test/gray_stroke
MVTec_AD/tile/test/oil
MVTec_AD/tile/test/rough
MVTec_AD/tile/ground_truth/*
```

本阶段输出目录：

```text
outputs/baselines/patchcore/tile/
```

关键输出：

```text
anomaly_maps/
overlays/
metadata/
mask_previews/
preview.png
image_scores.csv
metrics.json
summary.md
```

本阶段真实测试图数量：

```text
good: 33
crack: 17
glue_strip: 18
gray_stroke: 16
oil: 18
rough: 15
total: 117
```

## 3. 为什么先做无监督 baseline

工业异常检测里有一个很常见的设定：

```text
训练时只有正常图，测试时要找出异常图和异常区域。
```

MVTec AD 的 `train/good` 正好符合这个设定。

因此，本阶段不使用传统伪缺陷，也不使用 Diffusion 伪缺陷，而是只用真实正常图训练。

这样做有两个好处：

1. 可以得到一个干净的基准结果。
2. 后续加入生成数据时，才能比较“生成数据到底有没有带来提升”。

如果现在就把生成缺陷混进来，项目会变得难解释：

```text
模型变好是因为方法本来就强？
还是因为生成数据真的有帮助？
```

所以第 4 阶段先把 baseline 做清楚。

## 4. PatchCore 的直观原理

PatchCore 可以用一句话理解：

```text
先记住大量正常图的局部特征；测试时，如果某个局部特征离所有正常特征都很远，就认为它异常。
```

更通俗地说：

```text
正常 tile 的每一小块纹理都会形成一个“正常样本库”。
测试图进来后，把它也切成很多小块。
如果某个小块不像训练集中见过的任何正常小块，它的异常分数就高。
```

本阶段实现的是轻量 PatchCore 风格 baseline，核心步骤是：

```text
1. 用预训练 ResNet18 提取图像中间层特征
2. 只从 train/good 建立正常 memory bank
3. 对 test 图像提取同样的 patch 特征
4. 计算每个 patch 到正常 memory bank 的最近邻距离
5. 把距离还原为 anomaly map
6. 用真实 mask 计算检测和分割指标
```

这里的 `memory bank` 可以理解成：

```text
正常局部纹理特征库。
```

最近邻距离可以理解成：

```text
这个测试区域和正常区域有多不像。
```

## 5. 真实 test 集如何评估

本阶段对 MVTec AD `tile/test` 中所有真实测试图进行评估。

对于 `test/good`：

```text
图像标签为正常
mask 使用全黑图
```

对于真实缺陷类别：

```text
crack
glue_strip
gray_stroke
oil
rough
```

图像标签为异常，并读取 `ground_truth` 中对应的真实 mask。

评估分为两个层面：

```text
Image-level：整张图是否异常
Pixel-level：每个像素是否属于异常区域
```

## 6. 指标是什么意思

`Image AUROC`：

```text
衡量模型能不能把异常图排在正常图前面。
越接近 1 越好。
```

`Pixel AUROC`：

```text
衡量异常热力图能不能把缺陷像素排在正常像素前面。
越接近 1 越好。
```

`Precision`：

```text
模型预测为缺陷的区域里，有多少是真的缺陷。
Precision 低说明误检多。
```

`Recall`：

```text
真实缺陷区域里，有多少被模型找到了。
Recall 低说明漏检多。
```

`F1`：

```text
Precision 和 Recall 的综合指标。
```

`Dice`：

```text
常用于分割任务，表示预测区域和真实区域的重合程度。
本阶段二值分割下 Dice 等价于 F1。
```

`IoU`：

```text
预测缺陷区域和真实缺陷区域的交并比。
比 Dice 更严格一些。
```

## 7. 本阶段结果

本阶段运行结果：

```text
Image AUROC: 0.9942
Pixel AUROC: 0.9244
Image Precision: 0.9880
Image Recall: 0.9762
Image F1: 0.9820
Pixel Precision: 0.4457
Pixel Recall: 0.7602
Pixel F1 / Dice: 0.5620
Pixel IoU: 0.3908
```

结果保存在：

```text
outputs/baselines/patchcore/tile/metrics.json
outputs/baselines/patchcore/tile/image_scores.csv
outputs/baselines/patchcore/tile/preview.png
outputs/baselines/patchcore/tile/summary.md
```

其中：

```text
Image AUROC 很高，说明这个 baseline 很擅长区分整张图是否异常。
Pixel AUROC 也较高，说明异常热力图和真实缺陷区域有一定对应关系。
Pixel Dice 和 IoU 不算高，说明热力图定位还比较粗，不能直接等价于精细分割模型。
```

## 8. 哪些类别容易检出

从 `image_scores.csv` 的类别平均分看：

```text
crack: 0.9701
glue_strip: 0.9267
oil: 0.9065
rough: 0.8486
gray_stroke: 0.7815
good: 0.6982
```

`crack`、`glue_strip`、`oil` 的异常分数明显高于 `good`，说明这些缺陷相对容易被检测到。

`gray_stroke` 的分数最接近 `good`，说明它更难检测。

这和我们前面观察数据集时的直觉一致：

```text
gray_stroke 是局部颜色或纹理变化，和正常 tile 背景更接近。
```

## 9. 哪些地方仍然不足

本阶段有几个重要限制：

1. 当前实现是轻量 PatchCore 风格 baseline，不是完整官方 PatchCore 复现。
2. 当前使用 ResNet18，中间层特征较轻量，定位能力有限。
3. Pixel Dice 和 IoU 只是用测试集最佳 F1 阈值得到的诊断结果，不应该当成真实部署阈值。
4. anomaly map 能指出可疑区域，但边界不如监督分割模型精细。
5. 本阶段没有使用传统伪缺陷或 Diffusion 伪缺陷，因此还没有验证生成数据的价值。

## 10. 遇到的问题

本阶段第一次运行时，torchvision 自动下载 ResNet18 权重，下载进度条在 Windows 中文终端中出现了乱码。

解决方式：

```text
脚本中改为显式关闭权重下载进度条。
```

这不改变模型和算法，只是避免后续运行时出现终端编码噪声。

## 11. 下一步计划

下一步建议进入第 5 阶段：

```text
监督分割训练与生成数据增强对比
```

原因是：

```text
第 4 阶段已经给出了“不使用生成数据”的真实测试集 baseline。
第 5 阶段就可以开始验证“加入传统伪缺陷或 Diffusion 伪缺陷后，下游分割是否变好”。
```

第 5 阶段建议对比：

```text
只用传统伪缺陷训练 U-Net
只用 Diffusion 伪缺陷训练 U-Net
传统伪缺陷 + Diffusion 伪缺陷训练 U-Net
真实 test 集评估 Dice / IoU / Precision / Recall
```

这样项目主线会更完整：

```text
生成图是否好看
→ 生成图能否形成训练数据
→ 训练出的模型能否在真实缺陷上表现更好
```
