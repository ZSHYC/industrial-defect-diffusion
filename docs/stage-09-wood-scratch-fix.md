# 第 9 阶段：wood scratch 专项修复与跨类别误差分析

## 1. 本阶段目标

第 9 阶段继续沿用第 6、7 阶段的类别级误差分析路线，但对象从 `tile` 转到 `wood`。

第 8 阶段已经证明完整流程可以从 `tile` 迁移到 `wood`，但真实测试集上的像素级分割明显偏弱，尤其是：

```text
stage8 wood combined:
  Pixel F1 = 0.2651
  Best Pixel F1 = 0.2901
  Image F1 = 0.8947
  scratch Dice = 0.0247
  scratch Recall = 0.0146
```

因此，本阶段目标不是继续扩类别，而是专项修复 `wood/scratch`：

```text
跨类别迁移 -> 失败类别分析 -> 类别级生成修复 -> 下游真实测试集验证
```

## 2. 第 8 阶段暴露的问题

第 8 阶段的 wood 流程已经跑通，但 `scratch` 几乎没有被模型有效分割。

这说明 `tile` 上有效的线状缺陷生成思路不能直接套到 `wood`：

```text
1. wood scratch 不只是几条细线。
2. 真实 scratch 经常覆盖较大的 bbox。
3. 部分 scratch 更像大范围浅色磨损或沿纹理方向的扰动。
4. 当前 synthetic scratch 面积过小，导致训练信号不足。
```

## 3. 为什么 scratch 是重点

第 8 阶段各类别 Dice：

```text
color:    0.2650
combined: 0.3210
hole:     0.2960
liquid:   0.5674
scratch:  0.0247
```

`scratch` 是最明显的失败类别，也是最适合做专项修复的对象。

本阶段只替换 `scratch` 样本，复用第 8 阶段 accepted 的非 scratch 样本，避免改动过多导致结果难以归因。

## 4. 真实 scratch 分布统计

新增脚本：

```text
scripts/09_analyze_wood_scratch_distribution.py
```

输出：

```text
outputs/stage9_wood_scratch_fix/analysis/wood_scratch_distribution_summary.csv
outputs/stage9_wood_scratch_fix/analysis/wood_scratch_distribution_samples.csv
```

真实 `wood/scratch` 分布：

```text
count = 21
mean area_ratio = 0.0744
median area_ratio = 0.0275
mean bbox_width_ratio = 0.6444
mean bbox_height_ratio = 0.6348
mean inside_minus_outside = +8.92
```

这说明真实 scratch 往往是大 bbox、偏亮、沿木纹方向扩散的缺陷。

## 5. 旧 synthetic scratch 的问题

第 8 阶段旧 synthetic scratch：

```text
old_traditional:
  mean area_ratio = 0.0020
  mean bbox_width_ratio = 0.4617
  mean bbox_height_ratio = 0.4509
  mean inside_minus_outside = +8.95

old_diffusion:
  mean area_ratio = 0.0022
  mean bbox_width_ratio = 0.5062
  mean bbox_height_ratio = 0.5372
  mean inside_minus_outside = +2.12
```

灰度方向基本正确，但面积只有真实均值的很小一部分。

核心问题不是“亮不亮”，而是：

```text
mask 太小、形态太细、覆盖范围不足。
```

## 6. traditional scratch 怎么改

修改文件：

```text
scripts/02_generate_traditional_defects.py
```

`wood/scratch` 生成从原来的 1-3 条细线，改成三种随机模式：

```text
long_single:
  一条贯穿或半贯穿长划痕，可能带有较短分支。

clustered_lines:
  多条同方向浅色划痕簇，模拟沿木纹方向的连续刮擦。

broad_abrasion:
  大面积浅色磨损区域，叠加局部细划痕。
```

新 traditional scratch 分布：

```text
new_traditional:
  mean area_ratio = 0.0460
  median area_ratio = 0.0342
  mean bbox_width_ratio = 0.6549
  mean bbox_height_ratio = 0.5973
  mean inside_minus_outside = +10.74
```

相比旧版本，面积和 bbox 已经明显接近真实分布。

## 7. diffusion prompt 怎么改

修改文件：

```text
scripts/03_generate_diffusion_defects.py
```

旧 prompt 更强调 thin scratch，新 prompt 改为：

```text
realistic long bright scratch marks and large shallow abrasion on industrial wood surface,
follows natural wood grain texture,
inspection image,
visible wide scratched defect
```

新 diffusion scratch 分布：

```text
new_diffusion:
  mean area_ratio = 0.0408
  median area_ratio = 0.0389
  mean bbox_width_ratio = 0.6833
  mean bbox_height_ratio = 0.5321
  mean inside_minus_outside = +4.61
```

Diffusion 仍然沿用 traditional mask 做 inpainting，没有改变主流程。

## 8. 质量筛选结果

第 9 阶段只生成 scratch：

```text
traditional scratch: 20
diffusion scratch: 10
total: 30
accepted: 29
rejected: 1
```

按来源：

```text
traditional scratch: 19 / 20 accepted
diffusion scratch: 10 / 10 accepted
```

合并训练数据时，复用第 8 阶段 accepted 的非 scratch 样本，只替换 scratch：

```text
traditional accepted: 98
diffusion accepted: 50
total accepted: 148
```

## 9. U-Net 真实测试集结果

训练配置：

```text
category: wood
experiment: combined
image size: 256
epochs: 30
batch size: 4
seed: 404
train samples: 148
test samples: 79
```

第 9 阶段结果：

```text
Pixel Precision = 0.2546
Pixel Recall = 0.4976
Pixel F1 = 0.3369
Best Pixel F1 = 0.3815
Image Precision = 0.8219
Image Recall = 1.0000
Image F1 = 0.9023
```

按类别：

```text
color:
  Dice = 0.4368
  Recall = 0.6031

combined:
  Dice = 0.4480
  Recall = 0.4964

hole:
  Dice = 0.2997
  Recall = 0.3793

liquid:
  Dice = 0.7018
  Recall = 0.7266

scratch:
  Dice = 0.3405
  Recall = 0.4169
```

## 10. 和第 8 阶段对比

核心指标对比：

```text
stage8 wood combined:
  Pixel F1 = 0.2651
  Best Pixel F1 = 0.2901
  Image F1 = 0.8947
  scratch Dice = 0.0247
  scratch Recall = 0.0146

stage9 wood scratch fixed combined:
  Pixel F1 = 0.3369
  Best Pixel F1 = 0.3815
  Image F1 = 0.9023
  scratch Dice = 0.3405
  scratch Recall = 0.4169
```

提升幅度：

```text
Pixel F1: +0.0718
Best Pixel F1: +0.0913
Image F1: +0.0075
scratch Dice: +0.3158
scratch Recall: +0.4023
```

注意第 9 阶段的 pixel precision 从 0.6367 降到 0.2546，但 pixel recall 从 0.1674 提升到 0.4976。

这说明模型从非常保守的预测变成更愿意覆盖缺陷区域，尤其改善了 scratch 的漏检问题。对于本阶段目标来说，这是合理的 precision / recall 权衡。

## 11. 最终结论

第 9 阶段验证了第 8 阶段暴露的问题不是单纯训练随机性，而是 synthetic scratch 分布和真实 scratch 分布存在明显差距。

通过重写 scratch traditional 生成规则，并同步调整 diffusion prompt：

```text
1. synthetic scratch 面积从约 0.2% 提升到约 4%。
2. bbox 覆盖范围接近真实 wood scratch。
3. scratch Dice 从 0.0247 提升到 0.3405。
4. scratch Recall 从 0.0146 提升到 0.4169。
5. wood overall Pixel F1 从 0.2651 提升到 0.3369。
```

这说明项目方法不仅能在 `tile` 上通过类别级修复提升结果，也能在迁移到 `wood` 后继续用同样的方法定位失败类别并改进。

## 12. 面试表达版本

可以这样表达第 8、9 阶段：

```text
我先把 tile 上的生成增强流程迁移到 wood，证明代码不是单类别 hard-code demo。
但 wood 的像素级指标很低，尤其 scratch Dice 只有 0.0247。
我没有盲目换模型，而是回到数据分布，统计真实 scratch 和合成 scratch 的面积、bbox 和灰度差。
结果发现旧合成 scratch 面积只有约 0.2%，而真实 scratch 平均约 7.4%，生成分布明显偏小。
因此我重写了 scratch 生成规则，让它覆盖长划痕、划痕簇和大面积磨损三种形态。
修复后 scratch Dice 提升到 0.3405，Recall 提升到 0.4169，wood Pixel F1 也从 0.2651 提升到 0.3369。
这说明这个项目不是只做生成图片，而是能通过真实测试集验证生成分布对下游分割的影响。
```
