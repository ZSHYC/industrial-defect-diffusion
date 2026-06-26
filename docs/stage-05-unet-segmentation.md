# 第 5 阶段：U-Net 监督分割训练与生成数据增强对比

## 1. 本阶段目标

本阶段目标是验证生成缺陷数据是否能用于下游监督分割训练。

前 4 个阶段已经完成：

```text
第 1 阶段：理解 MVTec AD tile 数据结构
第 2 阶段：生成传统规则伪缺陷
第 3 阶段：生成 Diffusion Inpainting 伪缺陷
第 4 阶段：建立不使用生成数据的 PatchCore baseline
```

第 5 阶段要回答的问题是：

```text
用生成缺陷图 + 生成 mask 训练出来的分割模型，能不能在真实缺陷 test 集上找到缺陷区域？
```

这一步非常关键，因为项目的核心不是“生成图看起来像不像”，而是：

```text
生成数据能不能帮助真实工业缺陷检测或分割。
```

## 2. 本阶段做了什么

本阶段新增脚本：

```text
scripts/05_train_unet_segmentation.py
```

运行命令：

```powershell
conda run -n industrial-defect-diffusion python scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --image-size 256 --epochs 30 --batch-size 4 --seed 42
```

本阶段训练了三组 U-Net 分割模型：

```text
traditional：只用第 2 阶段传统伪缺陷训练，25 张
diffusion：只用第 3 阶段 Diffusion 伪缺陷训练，15 张
combined：传统伪缺陷 + Diffusion 伪缺陷一起训练，40 张
```

测试集全部使用真实 MVTec AD `tile/test`：

```text
good: 33
crack: 17
glue_strip: 18
gray_stroke: 16
oil: 18
rough: 15
total: 117
```

注意：

```text
真实 test 缺陷没有参与训练。
训练集是生成缺陷。
测试集是真实缺陷。
```

## 3. 为什么进入监督分割

第 4 阶段的 PatchCore 是无监督异常检测 baseline，它只学习正常图分布。

但是生成缺陷数据天然带有 mask：

```text
生成图像：缺陷长什么样
生成 mask：缺陷在哪里
```

所以生成数据最直接的用途之一就是训练监督分割模型。

如果模型只看生成缺陷训练，却能在真实缺陷 test 集上分割出缺陷区域，就说明生成数据有一定下游价值。

如果模型只能拟合生成图，在真实缺陷上完全失败，就说明生成数据和真实缺陷之间存在较大 domain gap。

## 4. 为什么使用 U-Net

U-Net 是医学图像和工业图像分割中非常经典的结构。

它适合本阶段有三个原因：

1. 结构直观，适合学习和面试讲解。
2. 输入图像，输出像素级 mask，和本项目目标完全对应。
3. 不依赖复杂框架，用 PyTorch 就能实现。

可以把 U-Net 理解成：

```text
左边编码器：逐步理解图像里有什么
右边解码器：逐步恢复空间位置
跳跃连接：把浅层纹理细节传回去，帮助定位边界
```

本阶段使用轻量 U-Net：

```text
输入：RGB 图像
输出：单通道缺陷概率图
loss：BCEWithLogitsLoss + Dice Loss
optimizer：Adam
threshold：默认 0.5
```

## 5. 三组实验分别代表什么

`traditional`：

```text
验证传统规则伪缺陷能不能直接训练分割模型。
```

`diffusion`：

```text
验证 Diffusion Inpainting 生成缺陷是否比传统伪缺陷更接近真实缺陷。
```

`combined`：

```text
验证传统方法提供形状多样性，Diffusion 提供纹理真实化，两者结合是否更好。
```

这三组实验对应项目中的核心对比：

```text
传统伪缺陷
Diffusion 伪缺陷
传统 + Diffusion
```

## 6. 本阶段结果

整体对比结果：

```text
traditional:
  train samples: 25
  Pixel Precision: 0.0725
  Pixel Recall: 0.9918
  Pixel F1 / Dice: 0.1351
  Pixel IoU: 0.0725
  Image F1: 0.8358

diffusion:
  train samples: 15
  Pixel Precision: 0.6999
  Pixel Recall: 0.3893
  Pixel F1 / Dice: 0.5003
  Pixel IoU: 0.3336
  Image F1: 0.8632

combined:
  train samples: 40
  Pixel Precision: 0.7862
  Pixel Recall: 0.8276
  Pixel F1 / Dice: 0.8064
  Pixel IoU: 0.6755
  Image F1: 0.8615
```

结果保存在：

```text
outputs/training/unet_segmentation/tile/
```

关键文件：

```text
comparison_summary.csv
comparison_preview.png
traditional/metrics.json
diffusion/metrics.json
combined/metrics.json
traditional/preview.png
diffusion/preview.png
combined/preview.png
```

## 7. 结果说明

本阶段最重要的结论是：

```text
combined 明显优于 traditional 和 diffusion 单独训练。
```

`traditional` 的问题很明显：

```text
Pixel Recall 很高，但 Pixel Precision 极低。
```

这说明模型几乎把大量区域都预测成缺陷。

抽样检查预测 mask 后发现：

```text
traditional 平均预测正样本面积比例约为 0.9386
diffusion 平均预测正样本面积比例约为 0.0389
combined 平均预测正样本面积比例约为 0.0738
```

也就是说，`traditional` 模型不是精准找到了缺陷，而是过度预测。

`diffusion` 的结果更克制：

```text
Precision 较高，Recall 偏低。
```

这说明 Diffusion 生成图训练出的模型更少乱报，但会漏掉一些真实缺陷。

`combined` 的结果最好：

```text
Precision 和 Recall 都比较高，Dice 和 IoU 明显提升。
```

这说明在当前小样本实验中：

```text
传统伪缺陷提供了形状和位置变化。
Diffusion 伪缺陷提供了更自然的局部纹理。
两者结合比单独使用更有效。
```

## 8. 哪些类别效果好

`combined` 模型按类别结果：

```text
crack:
  Dice: 0.5080
  IoU: 0.3545
  Recall: 0.4586

glue_strip:
  Dice: 0.8883
  IoU: 0.8023
  Recall: 0.9470

gray_stroke:
  Dice: 0.6913
  IoU: 0.5436
  Recall: 0.7279

oil:
  Dice: 0.9199
  IoU: 0.8541
  Recall: 0.9526

rough:
  Dice: 0.7423
  IoU: 0.6365
  Recall: 0.7280
```

效果较好的类别：

```text
oil
glue_strip
rough
```

这些类别通常面积更大、纹理或颜色变化更明显。

相对困难的类别：

```text
crack
gray_stroke
```

`crack` 通常细长，边界和宽度变化大。

`gray_stroke` 和背景纹理更接近，容易出现漏检或边界不准。

## 9. 为什么当前结果只能算小样本验证

虽然 `combined` 结果很好，但必须谨慎表达。

原因有三个：

1. 训练样本很少，只有 40 张生成缺陷。
2. 当前只做了 `tile` 一个类别。
3. 阈值和训练轮数还没有系统调参。

因此，本阶段结论应表述为：

```text
在小样本验证中，传统伪缺陷和 Diffusion 伪缺陷结合训练 U-Net，
能在真实 tile test 集上得到较好的像素级分割结果。
```

不要直接说：

```text
Diffusion 一定提升所有工业缺陷检测效果。
```

## 10. 遇到的问题

本阶段发现并修正了一个指标记录问题：

```text
metrics.json 中 train_samples 最初误写成 test 样本数量。
```

解决方式：

```text
显式把当前实验训练样本数传入 evaluate_model。
traditional / diffusion / combined 分别记录为 25 / 15 / 40。
```

此外，本阶段发现传统伪缺陷单独训练容易产生过度预测。

这不是代码错误，而是一个重要实验现象：

```text
传统规则伪缺陷和真实缺陷的纹理分布差距较大，
模型可能学到“哪里都像缺陷”的不稳定决策。
```

## 11. 下一步计划

下一阶段建议进入：

```text
扩大生成数据 + 质量筛选 + 重新训练对比
```

原因是：

```text
第 5 阶段已经证明当前小样本生成数据有下游可用性。
但样本数量太少，结论还不够稳。
```

下一步建议：

```text
每类生成更多传统伪缺陷和 Diffusion 伪缺陷
增加简单质量筛选，例如 mask 面积、预测面积、视觉 preview
重新训练 U-Net
比较小样本和扩展样本结果
```

这样项目会从“小样本可行性验证”进入“更接近真实实验”的阶段。
