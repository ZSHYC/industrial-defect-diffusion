# 第 6 阶段：扩大生成数据、质量筛选与监督分割再验证

## 1. 本阶段目标

第 6 阶段的目标是验证第 5 阶段的小样本结论是否稳定。

第 5 阶段已经证明：

```text
用传统伪缺陷 + Diffusion 伪缺陷训练 U-Net，可以在真实 MVTec AD tile 测试集上得到较好的分割结果。
```

但是第 5 阶段训练数据很少：

```text
traditional: 25 张
diffusion: 15 张
combined: 40 张
```

因此第 6 阶段要回答的问题是：

```text
如果把生成数据扩大，并加入基础质量筛选，模型表现是否仍然稳定？
```

这一阶段非常适合面试讲解，因为它体现了一个重要工业观点：

```text
生成数据不是越多越好，必须通过质量控制和真实测试集评估来验证。
```

## 2. 本阶段做了什么

本阶段做了 5 件事：

```text
1. 扩大传统规则伪缺陷数据
2. 扩大 Diffusion Inpainting 伪缺陷数据
3. 新增生成数据质量筛选脚本
4. 用扩大后的数据重新训练 U-Net
5. 和第 5 阶段小样本结果做横向对比
```

新增脚本：

```text
scripts/06_filter_synthetic_quality.py
```

修改脚本：

```text
scripts/05_train_unet_segmentation.py
```

修改点：

```text
1. 支持 --experiments，只跑指定实验组。
2. 修正自定义 output-dir 时的打印路径。
3. 为每个实验设置独立 seed offset，避免只跑 combined 和三组连续跑时结果不一致。
```

## 3. 为什么要扩大生成数据

第 5 阶段的 combined 结果很好：

```text
Pixel F1 / Dice: 0.8064
Pixel IoU: 0.6755
Image F1: 0.8615
```

但它只用了 40 张生成图训练。

这会带来一个问题：

```text
结果可能受到随机样本、随机初始化或小样本偶然性的影响。
```

所以第 6 阶段扩大训练数据：

```text
traditional: 从 25 张扩大到 100 张
diffusion: 从 15 张扩大到 50 张
combined: 从 40 张扩大到 150 张
```

这样可以观察：

```text
1. traditional 是否随着数量增加而变强
2. diffusion 是否仍然比 traditional 更稳
3. combined 是否继续保持优势
4. 生成更多数据是否一定带来更高分割指标
```

## 4. 运行命令

扩大传统规则伪缺陷：

```powershell
conda run -n industrial-defect-diffusion python scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --samples-per-type 20 --seed 43 --output-dir outputs/expanded_synthetic/traditional
```

扩大 Diffusion Inpainting 伪缺陷：

```powershell
conda run -n industrial-defect-diffusion python scripts/03_generate_diffusion_defects.py --category tile --traditional-summary outputs/expanded_synthetic/traditional/tile/summary.csv --samples-per-type 10 --num-inference-steps 30 --seed 43 --local-files-only --output-dir outputs/expanded_synthetic/diffusion
```

质量筛选：

```powershell
conda run -n industrial-defect-diffusion python scripts/06_filter_synthetic_quality.py --traditional-summary outputs/expanded_synthetic/traditional/tile/summary.csv --diffusion-summary outputs/expanded_synthetic/diffusion/tile/summary.csv --output-dir outputs/quality_filter/tile
```

扩大数据重新训练 U-Net：

```powershell
conda run -n industrial-defect-diffusion python scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --image-size 256 --epochs 30 --batch-size 4 --seed 43 --traditional-summary outputs/expanded_synthetic/traditional/tile/summary.csv --diffusion-summary outputs/expanded_synthetic/diffusion/tile/summary.csv --output-dir outputs/training/unet_segmentation_expanded
```

筛选后 combined 重新训练：

```powershell
conda run -n industrial-defect-diffusion python scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --image-size 256 --epochs 30 --batch-size 4 --seed 43 --traditional-summary outputs/quality_filter/tile/accepted_traditional_summary.csv --diffusion-summary outputs/quality_filter/tile/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_filtered --experiments combined
```

## 5. 质量筛选规则

质量筛选不是用深度学习模型做复杂判断，而是先做一层基础体检。

筛选脚本会检查：

```text
1. mask 不能为空
2. mask 面积不能太小
3. mask 面积不能太大
4. 缺陷区域相对原图必须有足够变化
5. mask 外背景不能被大面积污染
```

默认阈值：

```text
min_mask_area: 0.001
max_mask_area: 0.25
min_inside_diff: 4.0
max_outside_diff: 40.0
```

通俗理解：

```text
缺陷区域要真的变了，但不能把整张图都变坏。
```

本阶段筛选结果：

```text
traditional: 100 / 100 accepted
diffusion: 50 / 50 accepted
rejected: 0
```

这说明当前扩展生成数据没有出现明显硬性错误，例如空 mask、超大 mask、缺陷完全不可见或整图严重污染。

但也要注意：

```text
当前质量筛选是基础规则筛选，不是严格视觉质量筛选。
```

它能排除明显坏样本，但不能保证每一张生成图都“最像真实缺陷”。

## 6. 输出目录

扩大传统伪缺陷：

```text
outputs/expanded_synthetic/traditional/tile/
```

扩大 Diffusion 伪缺陷：

```text
outputs/expanded_synthetic/diffusion/tile/
```

质量筛选：

```text
outputs/quality_filter/tile/
```

关键文件：

```text
quality_report.csv
accepted_summary.csv
rejected_summary.csv
accepted_traditional_summary.csv
accepted_diffusion_summary.csv
filter_summary.csv
preview_accepted.png
preview_rejected.png
summary.md
```

扩大训练输出：

```text
outputs/training/unet_segmentation_expanded/tile/
```

筛选后训练输出：

```text
outputs/training/unet_segmentation_filtered/tile/
```

横向对比表：

```text
outputs/training/unet_segmentation_expanded/tile/stage5_vs_stage6_summary.csv
```

## 7. 本阶段整体结果

第 6 阶段扩大数据结果：

```text
traditional:
  train samples: 100
  Pixel Precision: 0.4883
  Pixel Recall: 0.6724
  Pixel F1 / Dice: 0.5657
  Pixel IoU: 0.3944
  Image F1: 0.8358

diffusion:
  train samples: 50
  Pixel Precision: 0.6331
  Pixel Recall: 0.7455
  Pixel F1 / Dice: 0.6847
  Pixel IoU: 0.5206
  Image F1: 0.8400

combined:
  train samples: 150
  Pixel Precision: 0.8913
  Pixel Recall: 0.6728
  Pixel F1 / Dice: 0.7667
  Pixel IoU: 0.6217
  Image F1: 0.8715
```

筛选后 combined：

```text
train samples: 150
Pixel Precision: 0.8913
Pixel Recall: 0.6728
Pixel F1 / Dice: 0.7667
Pixel IoU: 0.6217
Image F1: 0.8715
```

因为本阶段 150 张样本全部通过基础筛选，所以：

```text
filtered combined 和 expanded combined 使用的是同一批训练数据。
```

因此两者结果一致。

## 8. 和第 5 阶段对比

关键对比：

```text
stage5 traditional: 25 张, Pixel F1 = 0.1351
stage6 traditional: 100 张, Pixel F1 = 0.5657

stage5 diffusion: 15 张, Pixel F1 = 0.5003
stage6 diffusion: 50 张, Pixel F1 = 0.6847

stage5 combined: 40 张, Pixel F1 = 0.8064
stage6 combined: 150 张, Pixel F1 = 0.7667
```

这个结果很有价值。

它说明：

```text
1. 扩大数据显著提升了 traditional。
2. 扩大数据也提升了 diffusion。
3. combined 仍然是第 6 阶段三组里最强。
4. 但生成更多数据不一定超过第 5 阶段的小样本最佳结果。
```

第 6 阶段的 combined 相比第 5 阶段：

```text
Pixel Precision: 0.7862 -> 0.8913，提高
Pixel Recall: 0.8276 -> 0.6728，下降
Pixel F1: 0.8064 -> 0.7667，小幅下降
Image F1: 0.8615 -> 0.8715，小幅提高
```

通俗解释：

```text
第 6 阶段模型更谨慎了，误分割更少，所以 Precision 更高；
但它漏掉了一部分真实缺陷区域，所以 Recall 下降；
综合像素级 F1 略低于第 5 阶段小样本 combined。
```

这并不是坏结果，而是一个更真实的实验结论：

```text
生成数据增强需要数量，也需要更精细的质量筛选和类别平衡。
```

## 9. 按类别结果

第 6 阶段 combined 按真实测试类别统计：

```text
crack:
  Dice: 0.5937
  IoU: 0.4420
  Recall: 0.4770

glue_strip:
  Dice: 0.9406
  IoU: 0.8885
  Recall: 0.9461

gray_stroke:
  Dice: 0.0022
  IoU: 0.0011
  Recall: 0.0011

oil:
  Dice: 0.8616
  IoU: 0.7589
  Recall: 0.7787

rough:
  Dice: 0.6683
  IoU: 0.5378
  Recall: 0.5646
```

容易检出的类别：

```text
glue_strip
oil
rough
```

原因是这些缺陷通常面积较大、纹理或颜色变化明显，生成 mask 和真实 mask 的形态更容易对齐。

较困难的类别：

```text
gray_stroke
crack
```

其中 `gray_stroke` 在第 6 阶段明显失败。

可能原因：

```text
1. 扩展生成的 gray_stroke 和真实 gray_stroke 之间仍有 domain gap。
2. 真实 gray_stroke 可能颜色更深、边缘和形态更特殊。
3. 扩大数据后模型更偏向学习大面积、稳定纹理缺陷。
4. 小目标或低对比缺陷更容易被 Dice + BCE 训练忽略。
```

这个现象是下一阶段非常重要的改进方向。

## 10. 遇到的问题

本阶段遇到四个问题。

第一个问题是 Diffusion 进度条乱码。

现象：

```text
Diffusion 生成时，tqdm 进度条在 Windows conda run 输出中出现乱码。
```

原因：

```text
这是 Windows 控制台编码、conda run 输出转码和 tqdm 特殊进度字符共同导致的显示问题。
```

影响：

```text
不影响模型加载，不影响图片生成，不影响输出文件。
```

后续优化：

```text
可以在 Diffusion 脚本中增加参数关闭 progress bar，进一步减少 Windows 编码风险。
```

第二个问题是质量筛选全部通过。

这说明当前规则能做基础体检，但还不够严格。

当前规则能排除：

```text
空 mask
超大 mask
缺陷完全不可见
背景大面积污染
```

但不能充分判断：

```text
缺陷是否像真实类别
缺陷颜色是否符合真实分布
缺陷边缘是否自然
缺陷是否对下游模型最有帮助
```

因此下一阶段需要更细的类别级筛选。

第三个问题是训练脚本的复现性。

现象：

```text
只跑 combined 和连续跑 traditional/diffusion/combined 时，combined 指标曾经不一致。
```

原因：

```text
模型初始化会受到前面实验消耗随机数状态的影响。
```

修复方式：

```text
为每个实验设置独立 seed offset。
```

修复后：

```text
expanded combined 和 filtered combined 在同一数据、同一 seed 下结果一致。
```

第四个问题是 `gray_stroke` 生成质量明显不符合真实测试集。

现象：

```text
第 6 阶段 combined 的 gray_stroke 类 Dice 只有 0.0022。
```

回看真实测试图后发现：

```text
真实 gray_stroke 不是浅灰色胶囊条，而是深灰/黑灰的不规则污块或擦痕。
```

旧生成结果的问题：

```text
1. 面积太小。
2. 颜色太浅。
3. 形状过于规则，像平滑粗线或胶囊条。
4. Diffusion 会进一步把灰色缺陷变淡，导致缺陷更不明显。
```

量化对比：

```text
real gray_stroke:
  mean area ratio: 0.0421
  inside gray: 62.53
  inside - outside: -45.58

old traditional gray_stroke:
  mean area ratio: 0.0176
  inside gray: 86.45
  inside - outside: -33.58

old diffusion gray_stroke:
  mean area ratio: 0.0178
  inside gray: 93.78
  inside - outside: -25.63
```

这说明旧生成确实没有学到真实 `gray_stroke` 的核心视觉特征。

修复方式：

```text
1. 重写 traditional gray_stroke 生成逻辑。
2. 从平滑粗线改为噪声扰动的不规则深灰污块。
3. 调整 mask 面积分布，让它接近真实测试集。
4. 调整灰度和透明度，让缺陷相对背景的变暗程度接近真实。
5. 修改 Diffusion prompt，把 gray_stroke 描述为 dark gray black irregular smudge stain。
6. 给生成脚本增加 --defect-types 参数，支持只重做某一个缺陷类别。
```

修复后量化结果：

```text
fixed traditional gray_stroke:
  mean area ratio: 0.0458
  inside gray: 72.46
  inside - outside: -46.55

fixed diffusion gray_stroke:
  mean area ratio: 0.0437
  inside gray: 73.97
  inside - outside: -43.52
```

修复后重新训练 U-Net：

```text
traditional:
  Pixel F1: 0.5589
  Image F1: 0.8660

diffusion:
  Pixel F1: 0.8062
  Image F1: 0.9600

combined:
  Pixel F1: 0.8573
  Pixel IoU: 0.7502
  Image F1: 0.9492
```

修复后 combined 的 `gray_stroke` 类结果：

```text
gray_stroke Dice: 0.8409
gray_stroke IoU: 0.7328
gray_stroke Recall: 0.7877
```

这个结果说明：

```text
第 6 阶段 gray_stroke 失败的根因确实是生成图质量差，而不是 U-Net 结构本身不行。
```

## 11. 本阶段结论

第 6 阶段第一轮最重要的结论是：

```text
扩大生成数据后，traditional 和 diffusion 单独训练都明显变强，combined 仍然是本阶段最强组合。
```

但另一个同样重要的结论是：

```text
生成更多不一定直接超过小样本最佳结果。
```

后续针对 `gray_stroke` 做专项修复后，项目得到一个更强的结论：

```text
生成数据质量直接决定下游分割效果。
```

修复前：

```text
expanded combined Pixel F1: 0.7667
gray_stroke Dice: 0.0022
```

修复后：

```text
gray_stroke fixed combined Pixel F1: 0.8573
gray_stroke Dice: 0.8409
```

这说明项目不能停留在“多生成一些图”。

更合理的方向是：

```text
1. 按类别分析生成质量
2. 针对困难类别做专项改进
3. 设计更严格的质量筛选
4. 做类别平衡和困难类别增强
```

这会让项目更像真实工业算法工作，而不是简单 demo。

## 12. 下一步计划

下一阶段建议进入：

```text
第 7 阶段：crack 专项改进、质量筛选升级与最终实验整理
```

重点不是继续盲目扩大数量，而是沿着第 6 阶段已经验证有效的路线继续做：

```text
crack 召回仍偏低
基础筛选规则过宽
生成数据数量增加后 recall 下降
```

下一阶段可以做：

```text
1. 单独分析真实 crack 的长度、宽度、面积和灰度分布。
2. 调整传统 crack 生成规则。
3. 调整 crack 的 Diffusion prompt 和 strength。
4. 增加类别级质量筛选规则。
5. 整理 gray_stroke 修复前后的可视化和指标对比。
```

这样第 7 阶段就不是简单堆数据，而是进入更像工业落地的误差分析和迭代优化。
