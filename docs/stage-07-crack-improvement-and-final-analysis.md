# 第 7 阶段：crack 专项改进与最终实验整理

## 1. 本阶段目标

第 7 阶段的目标是沿着第 6 阶段的误差分析路线，继续对困难类别 `crack` 做专项改进，并把项目整理成一个可复现、可解释、适合面试讲解的完整实验闭环。

第 6 阶段已经证明：

```text
gray_stroke 失败的主要原因不是 U-Net 结构不行，而是生成数据和真实缺陷分布不匹配。
```

修复 `gray_stroke` 后，combined 模型从：

```text
Pixel F1: 0.7667
gray_stroke Dice: 0.0022
```

提升到：

```text
Pixel F1: 0.8573
gray_stroke Dice: 0.8409
```

因此第 7 阶段继续验证同一个工业观点：

```text
生成数据是否有价值，取决于它是否匹配真实缺陷分布，并且必须用真实测试集指标验证。
```

## 2. 当前已经做了一部分什么

进入第 7 阶段前，项目已经有一部分 `crack` 修复草稿：

```text
1. scripts/02_generate_traditional_defects.py 中的 crack 生成逻辑已经改过。
2. scripts/03_generate_diffusion_defects.py 中的 crack prompt 已经加强。
3. outputs/crack_fix/ 中已有少量 debug 图和旧分布统计。
```

本阶段把这些零散尝试整理成正式实验：

```text
1. 新增 crack 分布统计脚本。
2. 重新生成中等规模第 7 阶段数据。
3. 重新进行基础质量筛选。
4. 只训练关键的 combined U-Net 实验组。
5. 对比第 5、6、7 阶段结果。
```

新增脚本：

```text
scripts/07_analyze_crack_distribution.py
```

修改脚本：

```text
scripts/02_generate_traditional_defects.py
scripts/03_generate_diffusion_defects.py
scripts/06_filter_synthetic_quality.py
```

其中 `scripts/06_filter_synthetic_quality.py` 只把输出 Markdown 标题从固定的 Stage 6 改成通用标题，方便后续阶段复用。

## 3. 为什么 crack 是下一步重点

第 6 阶段 `gray_stroke` 修复后，模型整体已经比较强：

```text
Pixel F1: 0.8573
Image F1: 0.9492
```

但 `crack` 仍然偏弱：

```text
crack Dice: 0.6120
crack Recall: 0.4732
```

这说明模型对 crack 的定位仍然不够充分，尤其容易漏掉真实裂纹区域。

从工业检测角度看，crack 是非常重要的缺陷类型，因为它通常对应材料破裂或结构风险。如果模型漏检 crack，即使整体指标不错，也会影响项目说服力。

## 4. crack 修复前的问题

修复前的旧 crack 生成主要有两个问题：

```text
1. 面积太小，mask_area_ratio 明显低于真实 crack。
2. 形态不够接近真实贯穿式裂纹。
```

量化统计如下：

```text
real crack mean_area_ratio: 0.0221
old traditional crack mean_area_ratio: 0.0070
old diffusion crack mean_area_ratio: 0.0070
```

这说明旧生成 crack 只有真实 crack 的约三分之一面积。模型训练时看到的 crack 太小、太窄，就容易在真实测试集上漏掉较大或贯穿式裂纹。

灰度差方面：

```text
real inside - outside: -39.43
old traditional inside - outside: -90.35
old diffusion inside - outside: -31.87
```

旧 traditional crack 太黑，old diffusion 的灰度差更接近真实，但形态和面积仍然偏小。

## 5. traditional crack 生成规则怎么改

本阶段将 traditional crack 从局部短线改为更接近真实 tile crack 的结构：

```text
1. 从图像边界采样起点和终点，生成更长的贯穿式主裂纹。
2. 主裂纹由多个控制点组成，加入轻微弯曲。
3. 增加 1 到 3 条分叉裂纹。
4. 使用较宽的 soft mask 和更深的 core mask，模拟裂纹中心更暗、边缘过渡的结构。
5. 调整 target_gray 和 alpha，让缺陷区域灰度差更接近真实 crack。
```

修复后的 traditional crack 分布：

```text
new traditional mean_area_ratio: 0.0151
new traditional inside - outside: -41.02
```

面积仍略小于真实 crack 的 0.0221，但已经明显接近真实范围；灰度差则非常接近真实的 -39.43。

## 6. diffusion prompt 怎么改

Diffusion prompt 从：

```text
a realistic long thin dark crack defect on industrial ceramic tile surface, inspection image, natural texture
```

改为：

```text
a realistic long branching dark hairline crack fracture across industrial ceramic tile surface, thin irregular fracture lines, inspection image, natural texture
```

调整重点是让模型更关注：

```text
1. long branching
2. dark hairline crack
3. fracture across tile surface
4. thin irregular fracture lines
```

修复后的 diffusion crack 分布：

```text
new diffusion mean_area_ratio: 0.0174
new diffusion inside - outside: -22.52
```

Diffusion 生成后的 crack 面积比旧版本更接近真实，但灰度差仍然偏浅。这说明 Diffusion 对裂纹视觉效果有一定自然化作用，但也可能把传统规则中的暗裂纹变淡。

## 7. 运行命令

生成第 7 阶段 traditional 数据：

```powershell
conda run -n industrial-defect-diffusion python scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --samples-per-type 20 --seed 204 --output-dir outputs/stage7_synthetic/traditional
```

生成第 7 阶段 diffusion 数据：

```powershell
conda run -n industrial-defect-diffusion python scripts/03_generate_diffusion_defects.py --category tile --traditional-summary outputs/stage7_synthetic/traditional/tile/summary.csv --samples-per-type 10 --num-inference-steps 30 --seed 204 --local-files-only --output-dir outputs/stage7_synthetic/diffusion
```

质量筛选：

```powershell
conda run -n industrial-defect-diffusion python scripts/06_filter_synthetic_quality.py --traditional-summary outputs/stage7_synthetic/traditional/tile/summary.csv --diffusion-summary outputs/stage7_synthetic/diffusion/tile/summary.csv --output-dir outputs/stage7_quality_filter/tile
```

crack 分布统计：

```powershell
conda run -n industrial-defect-diffusion python scripts/07_analyze_crack_distribution.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile
```

训练第 7 阶段 combined U-Net：

```powershell
conda run -n industrial-defect-diffusion python scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --image-size 256 --epochs 30 --batch-size 4 --seed 104 --traditional-summary outputs/stage7_quality_filter/tile/accepted_traditional_summary.csv --diffusion-summary outputs/stage7_quality_filter/tile/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_stage7 --experiments combined
```

## 8. 输出文件

第 7 阶段主要输出：

```text
outputs/stage7_synthetic/traditional/tile/summary.csv
outputs/stage7_synthetic/diffusion/tile/summary.csv
outputs/stage7_quality_filter/tile/summary.md
outputs/stage7_quality_filter/tile/filter_summary.csv
outputs/crack_fix/crack_distribution_summary.csv
outputs/crack_fix/crack_distribution_samples.csv
outputs/training/unet_segmentation_stage7/tile/comparison_summary.csv
outputs/training/unet_segmentation_stage7/tile/combined/metrics.json
```

样本数量：

```text
traditional: 100
diffusion: 50
quality accepted: 150
quality rejected: 0
```

## 9. 分布统计对比

crack 分布统计结果：

```text
real:
  count: 17
  mean_area_ratio: 0.0221
  mean_inside_minus_outside: -39.43

old traditional:
  count: 20
  mean_area_ratio: 0.0070
  mean_inside_minus_outside: -90.35

old diffusion:
  count: 10
  mean_area_ratio: 0.0070
  mean_inside_minus_outside: -31.87

new traditional:
  count: 20
  mean_area_ratio: 0.0151
  mean_inside_minus_outside: -41.02

new diffusion:
  count: 10
  mean_area_ratio: 0.0174
  mean_inside_minus_outside: -22.52
```

结论：

```text
1. 新 traditional crack 的面积分布明显更接近真实 crack。
2. 新 traditional crack 的灰度差非常接近真实 crack。
3. 新 diffusion crack 的面积也更接近真实 crack。
4. Diffusion 会把 crack 变浅，灰度差小于真实 crack。
```

这说明第 7 阶段的生成规则确实修复了旧 crack 太小的问题，但 Diffusion 输出仍存在裂纹偏浅的 domain gap。

## 10. U-Net 真实测试集结果

第 7 阶段只训练 combined 实验组：

```text
train samples: 150
test samples: 117
Pixel Precision: 0.9462
Pixel Recall: 0.7606
Pixel F1 / Dice: 0.8433
Pixel IoU: 0.7291
Best Pixel F1: 0.8668
Image Precision: 0.9438
Image Recall: 1.0000
Image F1: 0.9711
```

按类别结果：

```text
crack:
  Dice: 0.7589
  IoU: 0.6169
  Recall: 0.6841

glue_strip:
  Dice: 0.9246
  IoU: 0.8617
  Recall: 0.8877

gray_stroke:
  Dice: 0.8398
  IoU: 0.7281
  Recall: 0.7512

oil:
  Dice: 0.9062
  IoU: 0.8290
  Recall: 0.8345

rough:
  Dice: 0.7119
  IoU: 0.5795
  Recall: 0.5962
```

## 11. 和第 6 阶段最佳结果对比

第 6 阶段 `gray_stroke` 修复后 combined：

```text
Pixel F1: 0.8573
Best Pixel F1: 0.8719
Image F1: 0.9492
crack Dice: 0.6120
crack Recall: 0.4732
gray_stroke Dice: 0.8409
```

第 7 阶段 combined：

```text
Pixel F1: 0.8433
Best Pixel F1: 0.8668
Image F1: 0.9711
crack Dice: 0.7589
crack Recall: 0.6841
gray_stroke Dice: 0.8398
```

关键变化：

```text
crack Dice: 0.6120 -> 0.7589，提高 0.1469
crack Recall: 0.4732 -> 0.6841，提高 0.2109
gray_stroke Dice: 0.8409 -> 0.8398，基本保持
Image F1: 0.9492 -> 0.9711，提高
Pixel F1: 0.8573 -> 0.8433，小幅下降
Best Pixel F1: 0.8719 -> 0.8668，小幅下降
```

这个结果说明：

```text
1. crack 专项改进是有效的，真实 crack 的分割 Dice 和 Recall 都明显提升。
2. gray_stroke 修复成果基本保住了，没有因为重做数据而明显退化。
3. 默认阈值 0.5 下 overall Pixel F1 小幅下降，说明类别级改进会带来整体 precision/recall 权衡。
4. Image F1 提升到 0.9711，说明模型对是否存在缺陷的判断更稳。
```

因此第 7 阶段不是简单意义上的“所有指标都变高”，而是一个更真实的工业实验结论：

```text
针对困难类别改进生成分布，可以显著提升该类别召回和定位能力；
但整体像素级指标会受到类别占比、阈值和其他类别表现的共同影响。
```

## 12. 最终项目结论

本项目最终最重要的结论不是“Diffusion 能生成图片”，而是：

```text
生成数据必须通过真实测试集验证，且生成分布是否贴近真实缺陷，会直接影响下游检测/分割效果。
```

从阶段结果看：

```text
stage5 small combined:
  Pixel F1: 0.8064
  Image F1: 0.8615

stage6 expanded combined:
  Pixel F1: 0.7667
  Image F1: 0.8715
  gray_stroke Dice: 0.0022

stage6 gray_stroke fixed combined:
  Pixel F1: 0.8573
  Image F1: 0.9492
  gray_stroke Dice: 0.8409

stage7 crack improved combined:
  Pixel F1: 0.8433
  Best Pixel F1: 0.8668
  Image F1: 0.9711
  crack Dice: 0.7589
```

最适合作为项目主结果的是：

```text
第 6 阶段 gray_stroke fixed combined：默认 Pixel F1 最高，整体像素级分割最强。
```

第 7 阶段适合作为误差分析和迭代优化案例：

```text
它证明 crack 生成分布修复后，crack Dice 和 Recall 明显提升。
```

这两个结果结合起来，比单独追一个最高分更有面试价值。

## 13. 面试表达版本

可以这样讲：

```text
这个项目不是简单用 Stable Diffusion 生成一些工业缺陷图，而是围绕真实工业检测任务做闭环验证。

我先用传统规则和 Diffusion 生成 tile 缺陷，再用这些合成数据训练 U-Net，并在真实 MVTec AD 测试集上评估。

实验中我发现，直接扩大生成数据不一定提升效果。比如第 6 阶段 expanded combined 的 Pixel F1 是 0.7667，而且 gray_stroke 类几乎失败，Dice 只有 0.0022。

我没有只继续堆数据，而是做了类别级误差分析，发现 gray_stroke 的生成面积、颜色和真实缺陷不匹配。修复后，combined Pixel F1 提升到 0.8573，gray_stroke Dice 提升到 0.8409。

之后我又对 crack 做专项分析，发现旧 crack 生成面积只有真实 crack 的约三分之一。调整成更长、更接近贯穿裂纹的生成规则后，crack Dice 从 0.6120 提升到 0.7589，Recall 从 0.4732 提升到 0.6841。

所以我对这个项目的理解是：工业 AIGC 不是生成越多越好，而是要可控生成、质量筛选、类别级误差分析，并且最终一定要用真实测试集验证它是否真的提升下游模型。
```

## 14. 下一步建议

如果继续深入，可以做三件事：

```text
1. 针对 Diffusion crack 偏浅的问题，尝试调低 strength 或改用 source image conditioning。
2. 引入类别加权 loss 或困难类别采样，提高 crack 和 rough 的像素召回。
3. 在第二个 MVTec AD 类别上复现实验，验证结论是否不只适用于 tile。
```

但作为面试项目，目前已经形成比较完整的故事：

```text
数据探索 -> 传统生成 -> Diffusion 生成 -> 下游分割验证 -> 质量筛选 -> 类别级误差分析 -> 定向修复 -> 最终表达。
```
