# 第 12 阶段：leather fold 专项修复与保守模型召回补强

## 1. 本阶段目标

第 12 阶段不继续扩第四个类别，而是针对第 11 阶段 `leather` 剩余最弱类别做专项修复：

```text
fold Dice = 0.0972
fold Recall = 0.0571
```

第 11 阶段已经证明加入 `train/good` 空 mask 负样本可以显著缓解 leather 过分割问题。
但模型也因此变得更保守，`fold` 这种浅褶皱类缺陷召回明显不足。

## 2. 为什么 fold 是下一步重点

第 11 阶段结果：

```text
Pixel Precision = 0.8752
Pixel Recall = 0.3282
Pixel F1 = 0.4774
Best Pixel F1 = 0.5219
Image F1 = 0.9667

color Dice = 0.8530
cut Dice = 0.4064
fold Dice = 0.0972
glue Dice = 0.8433
poke Dice = 0.3475
```

`color` 和 `glue` 已经较好，`cut` 在第 11 阶段明显修复。
继续扩类别会绕开当前最明确的问题，所以第 12 阶段选择做 `fold` 专项分析。

## 3. 真实 fold 与旧 synthetic fold 的差异

输出文件：

```text
outputs/stage12_leather_fold_fix/analysis/leather_fold_distribution_summary.csv
outputs/stage12_leather_fold_fix/analysis/leather_fold_distribution_samples.csv
```

分布统计：

```text
real fold:
  mean area_ratio = 0.0251
  mean bbox_w = 0.2919
  mean bbox_h = 0.1641
  mean inside_minus_outside = +3.59

old traditional fold:
  mean area_ratio = 0.0569
  mean bbox_w = 0.3507
  mean bbox_h = 0.3601
  mean inside_minus_outside = +4.14

old diffusion fold:
  mean area_ratio = 0.0547
  mean bbox_w = 0.3292
  mean bbox_h = 0.3503
  mean inside_minus_outside = +0.28
```

结论：

```text
旧 synthetic fold 不是太小，而是面积偏大、bbox_h 偏高、形态过宽。
Diffusion 后 fold 的平均亮暗差很弱，接近背景纹理。
```

## 4. traditional fold 怎么改

第 12 阶段只改 `leather/fold`，不改 `color/cut/glue/poke`。

新的 fold 生成包含三种模式：

```text
shallow_crease: 短到中等长度浅褶皱
narrow_band: 较窄长条压痕
double_ridge: 一亮一暗的轻微折痕边缘
```

关键变化：

```text
缩短 band_width
降低 polygon 面积
增加 ridge / shadow 双边结构
让折痕轻微弯曲
mask 不再把模糊外扩区全部算进去
```

修复后的 traditional fold：

```text
new traditional fold:
  mean area_ratio = 0.0153
  mean bbox_w = 0.3450
  mean bbox_h = 0.1540
  mean inside_minus_outside = +3.56
```

这说明新 traditional fold 的 `bbox_h` 和亮暗差已经明显接近真实 fold。

## 5. diffusion prompt 怎么改

第 12 阶段将 leather fold prompt 改成：

```text
a realistic shallow elongated fold crease defect on industrial leather surface,
subtle raised ridge and shadow,
follows fine leather grain texture,
inspection image,
visible local defect
```

Diffusion 仍使用 traditional mask 做 inpainting，不改变主流程。

但分布分析显示：

```text
new diffusion fold:
  mean area_ratio = 0.0154
  mean bbox_h = 0.1355
  mean inside_minus_outside = +1.27
```

Diffusion fold 形状跟随新 mask 变窄，但可见亮暗差仍弱于真实 fold。
同时质量筛选显示 diffusion fold 的背景变化较大，因此最终训练不追加第 12 阶段 diffusion fold，只保留它作为分析结果。

## 6. 质量筛选和最终数据策略

fold-only 质量筛选：

```text
traditional fold: 10 / 20 accepted
diffusion fold: 10 / 10 accepted
```

但是进一步检查发现：

```text
stage12 traditional fold:
  outside_mean_abs_diff ~= 0.0012

stage12 diffusion fold:
  outside_mean_abs_diff ~= 6.6990
```

这说明 diffusion fold 虽通过基础规则，但背景也发生了明显变化。
对 leather 这种容易过分割的类别，背景扰动会增加模型对正常纹理的误报。

最终采用的数据策略：

```text
保留 stage11 accepted leather 全量样本
追加 stage12 accepted traditional fold 样本
不追加 stage12 diffusion fold 样本
good negative samples 从 100 增加到 200
```

最终训练集：

```text
synthetic samples = 150
good negative samples = 200
total train samples = 350
```

类别数量：

```text
traditional color = 17
traditional cut = 20
traditional fold = 23
traditional glue = 20
traditional poke = 20

diffusion color = 10
diffusion cut = 10
diffusion fold = 10
diffusion glue = 10
diffusion poke = 10
```

## 7. U-Net 真实测试集结果

训练命令：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category leather --image-size 256 --epochs 30 --batch-size 4 --seed 604 --traditional-summary outputs/stage12_leather_fold_fix/quality_filter/leather/accepted_traditional_summary.csv --diffusion-summary outputs/stage12_leather_fold_fix/quality_filter/leather/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_stage12_leather_fold_fix --experiments combined --good-negative-samples 200
```

默认 threshold=0.5 结果：

```text
Pixel Precision = 0.2004
Pixel Recall = 0.6771
Pixel F1 = 0.3093
Best Pixel F1 = 0.4011
Image F1 = 0.9735
```

按类别：

```text
color:
  Dice = 0.4625
  Recall = 0.9070

cut:
  Dice = 0.1555
  Recall = 0.6464

fold:
  Dice = 0.4873
  Recall = 0.6660

glue:
  Dice = 0.7240
  Recall = 0.8610

poke:
  Dice = 0.3922
  Recall = 0.5108
```

good 图像：

```text
good image_score mean = 0.1737
good image_score median = 0.0082
```

## 8. 和第 11 阶段对比

核心指标：

```text
stage11 leather:
  Pixel Precision = 0.8752
  Pixel Recall = 0.3282
  Pixel F1 = 0.4774
  Best Pixel F1 = 0.5219
  Image F1 = 0.9667
  fold Dice = 0.0972
  fold Recall = 0.0571

stage12 leather fold fixed:
  Pixel Precision = 0.2004
  Pixel Recall = 0.6771
  Pixel F1 = 0.3093
  Best Pixel F1 = 0.4011
  Image F1 = 0.9735
  fold Dice = 0.4873
  fold Recall = 0.6660
```

变化：

```text
fold Dice: +0.3901
fold Recall: +0.6089
Pixel Recall: +0.3489
Image F1: +0.0068
Pixel Precision: -0.6748
Pixel F1: -0.1681
Best Pixel F1: -0.1209
```

第 12 阶段没有取代第 11 阶段成为 leather overall 最佳模型。
它的价值是证明：`fold` 的低召回确实来自生成分布和保守训练约束之间的冲突；当补充更真实的 fold 后，fold Dice 和 Recall 大幅提升，但整体 precision 会下降。

## 9. threshold / postprocess 诊断

`threshold_sweep.csv` 最佳点：

```text
threshold = 0.95
Pixel Precision = 0.2992
Pixel Recall = 0.6082
Pixel F1 = 0.4011
```

`postprocess_sweep.csv` 最佳点：

```text
threshold = 0.95
min_component_area_ratio = 0.001
Pixel Precision = 0.3256
Pixel Recall = 0.5906
Pixel F1 = 0.4198
```

后处理能小幅提高 F1，但仍低于第 11 阶段默认 Pixel F1。
这说明第 12 阶段的主要问题不是小连通域噪声，而是训练目标从高 precision 转向高 recall。

## 10. 最终结论

第 12 阶段结论：

```text
1. fold 生成分布修复有效：形态、bbox_h、亮暗差更接近真实 fold。
2. fold 下游指标显著提升：Dice 从 0.0972 到 0.4873，Recall 从 0.0571 到 0.6660。
3. 但 overall Pixel F1 从 0.4774 降到 0.3093，说明 fold 召回补强带来了 precision 代价。
4. leather 当前 overall 推荐模型仍是第 11 阶段。
5. 第 12 阶段适合作为“专项召回修复与 precision/recall 权衡分析”的补充实验。
```

## 11. 面试表达版本

可以这样表达第 12 阶段：

```text
第 11 阶段把 leather 的过分割问题修好了，但 fold 类召回很低，Dice 只有 0.0972。
我继续做类别级分布分析，发现旧 synthetic fold 面积偏大、bbox 高度偏高，和真实浅褶皱不一致。
第 12 阶段我把 fold 生成改成更窄、更浅的 ridge/shadow 褶皱，并追加高质量 fold 样本。
修复后 fold Dice 提升到 0.4873，Recall 提升到 0.6660。
但 overall Pixel F1 从 0.4774 降到 0.3093，说明补 fold 召回会牺牲 pixel precision。
所以我没有把它包装成新的最佳模型，而是把第 11 阶段作为 overall best，第 12 阶段作为类别专项修复和 tradeoff 分析。
```
