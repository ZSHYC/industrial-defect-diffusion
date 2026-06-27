# 第 11 阶段：leather precision / cut 专项修复

## 1. 本阶段目标

第 11 阶段不是继续扩第四个类别，而是修复第 10 阶段 `leather` 暴露出的核心问题：

```text
leather pipeline 已经跑通，但 pixel-level 分割严重过分割。
```

第 10 阶段基线：

```text
Pixel Precision = 0.0305
Pixel Recall = 0.5488
Pixel F1 = 0.0579
Best Pixel F1 = 0.1444
Image F1 = 0.8519
cut Dice = 0.0215
```

第 10 阶段测试预测中，`good` 图像的 `image_score` 均值约为 `0.994`，说明模型对正常 leather 纹理也产生了很高响应。

因此本阶段目标是：

```text
1. 加入真实 train/good 空 mask 负样本，压低正常图响应。
2. 修复 leather/cut 生成分布。
3. 增加 threshold sweep 和后处理 sweep，解释 precision / recall 权衡。
```

## 2. 为什么不是继续扩类别

第 8、10 阶段已经证明流程能迁移到 `wood` 和 `leather`。

继续扩第四类别会增加工作量，但不能解决当前最清楚的问题：

```text
leather 小面积缺陷下，模型像素级预测过大。
```

所以第 11 阶段选择回到类别级误差分析：

```text
第三类别泛化 -> 发现过分割 -> 加入正常负样本 -> 修复 cut 分布 -> 真实测试集再验证
```

## 3. 为什么加入 good negative samples

第 10 阶段的 U-Net 训练数据只包含 synthetic defect 正样本，没有真实正常图空 mask 负样本。

这会导致模型学到：

```text
leather 纹理区域都可能是缺陷。
```

第 11 阶段在训练脚本中新增：

```text
--good-negative-samples 100
```

训练时从：

```text
MVTec_AD/leather/train/good
```

按 seed 固定抽取 100 张正常图，并使用全 0 mask 作为负样本。

默认值仍为 0，保证旧阶段命令行为不变。

## 4. 评估脚本改造

修改脚本：

```text
scripts/05_train_unet_segmentation.py
```

新增输出：

```text
threshold_sweep.csv
postprocess_sweep.csv
```

`threshold_sweep.csv` 评估：

```text
threshold: 0.05 到 0.95，步长 0.01
```

`postprocess_sweep.csv` 评估：

```text
threshold: 0.50, best_pixel_threshold
min_component_area_ratio: 0, 0.0002, 0.0005, 0.001, 0.002
```

这些结果只用于诊断，不改变 `metrics.json` 默认 threshold=0.5 的主指标。

## 5. cut 分布修复

第 10 阶段旧 synthetic cut 问题：

```text
real cut:
  mean area_ratio = 0.0050
  mean inside_minus_outside = +11.16

old_traditional cut:
  mean area_ratio = 0.0020
  mean inside_minus_outside = -10.36

old_diffusion cut:
  mean area_ratio = 0.0021
  mean inside_minus_outside = -0.67
```

旧生成规则面积偏小，而且灰度方向和真实 cut 相反。

第 11 阶段将 `leather/cut` 改成：

```text
更细、更局部的切痕区域
亮色 cut 主体
轻微暗色核心
局部亮边
area_ratio 目标 0.003 到 0.008
```

新分布：

```text
new_traditional cut:
  mean area_ratio = 0.0037
  mean inside_minus_outside = +23.12

new_diffusion cut:
  mean area_ratio = 0.0036
  mean inside_minus_outside = +4.44
```

面积接近真实 cut，灰度方向也从负值修回正值。

## 6. 质量筛选与合并数据

第 11 阶段只重新生成 cut：

```text
cut traditional: 20
cut diffusion: 10
```

筛选结果：

```text
traditional cut: 20 / 20 accepted
diffusion cut: 10 / 10 accepted
```

合并策略：

```text
复用 stage10 accepted 的非 cut 样本
替换为 stage11 accepted 的 cut 样本
```

合并后 synthetic 样本：

```text
traditional accepted: 90
diffusion accepted: 50
synthetic total: 140
```

训练时额外加入：

```text
good negative samples: 100
total train samples: 240
```

## 7. U-Net 真实测试集结果

训练配置：

```text
category: leather
experiment: combined
image size: 256
epochs: 30
batch size: 4
seed: 604
synthetic samples: 140
good negative samples: 100
total train samples: 240
```

第 11 阶段默认 threshold=0.5 结果：

```text
Pixel Precision = 0.8752
Pixel Recall = 0.3282
Pixel F1 = 0.4774
Best Pixel F1 = 0.5219
Image Precision = 0.9886
Image Recall = 0.9457
Image F1 = 0.9667
```

按类别：

```text
color:
  Dice = 0.8530
  Recall = 0.8361

cut:
  Dice = 0.4064
  Recall = 0.3461

fold:
  Dice = 0.0972
  Recall = 0.0571

glue:
  Dice = 0.8433
  Recall = 0.7752

poke:
  Dice = 0.3475
  Recall = 0.2342
```

## 8. 和第 10 阶段对比

核心指标：

```text
stage10 leather:
  Pixel Precision = 0.0305
  Pixel Recall = 0.5488
  Pixel F1 = 0.0579
  Best Pixel F1 = 0.1444
  Image F1 = 0.8519
  cut Dice = 0.0215

stage11 leather precision / cut fixed:
  Pixel Precision = 0.8752
  Pixel Recall = 0.3282
  Pixel F1 = 0.4774
  Best Pixel F1 = 0.5219
  Image F1 = 0.9667
  cut Dice = 0.4064
```

提升：

```text
Pixel Precision: +0.8446
Pixel F1: +0.4195
Best Pixel F1: +0.3775
Image F1: +0.1148
cut Dice: +0.3849
```

Pixel Recall 从 `0.5488` 降到 `0.3282`，这是修复过分割后的合理权衡。

第 10 阶段模型几乎到处预测缺陷，所以 recall 高但 precision 极低。第 11 阶段加入 good negative 后，模型明显收敛，只在更可信的位置预测缺陷。

## 9. threshold / postprocess 诊断

`threshold_sweep.csv` 最佳点：

```text
threshold = 0.05
Pixel Precision = 0.7997
Pixel Recall = 0.3874
Pixel F1 = 0.5219
```

`postprocess_sweep.csv` 最佳点：

```text
threshold = 0.05
min_component_area_ratio = 0
Pixel F1 = 0.5219
```

小连通域过滤没有进一步提升，说明第 11 阶段主要收益来自训练数据中的 good negative 和 cut 分布修复，而不是简单后处理。

## 10. 剩余问题

第 11 阶段仍有一个明显短板：

```text
fold Dice = 0.0972
fold Recall = 0.0571
```

加入 good negative 后，模型变得更保守，`fold` 这类大而浅的褶皱被压制得比较多。

如果继续做第 12 阶段，建议不要再扩类别，而是：

```text
leather fold 专项修复
```

## 11. 面试表达版本

可以这样表达第 10、11 阶段：

```text
我把流程迁移到 leather 后，Image F1 有 0.8519，但 Pixel Precision 只有 0.0305。
进一步检查发现，模型在 good 图上也给出很高异常响应，这说明训练集中缺少真实正常图约束。
因此我在第 11 阶段加入了 100 张 leather train/good 作为空 mask 负样本，同时修复 cut 生成分布。
修复后 Pixel Precision 从 0.0305 提升到 0.8752，Pixel F1 从 0.0579 提升到 0.4774，cut Dice 从 0.0215 提升到 0.4064。
这个阶段说明，生成增强不仅要关注缺陷样本，也要关注正常样本对模型边界的约束。
```
