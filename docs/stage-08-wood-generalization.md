# 第 8 阶段：wood 类别泛化验证与复现实验包

## 1. 本阶段目标

第 8 阶段的目标是验证当前流程是否不只适用于 `tile`，而是可以迁移到第二个 MVTec AD 类别。

本阶段选择 `wood`，原因是：

```text
1. wood 和 tile 都是表面纹理类工业图像。
2. wood 的缺陷类型直观，包括 color、hole、liquid、scratch、combined。
3. wood 既适合复用 tile 阶段的生成思路，也能暴露跨类别 domain gap。
```

本阶段不是追求 wood 指标立刻超过 tile，而是验证：

```text
数据探索 -> 传统生成 -> Diffusion 生成 -> 质量筛选 -> U-Net 训练 -> 真实测试集评估
```

这条完整链路能否迁移到第二个类别。

## 2. 本阶段做了什么

本阶段做了 6 件事：

```text
1. 将 traditional 生成脚本从 tile hard-code 改成类别配置。
2. 为 wood 新增五类 traditional 生成规则。
3. 将 diffusion prompt 改成按类别配置。
4. 将 U-Net 训练评估脚本改成按类别读取 defect types。
5. 新增 wood 一键复现实验入口。
6. 跑完整 wood 泛化实验并记录结果。
```

新增脚本：

```text
scripts/08_run_wood_generalization.py
```

修改脚本：

```text
scripts/02_generate_traditional_defects.py
scripts/03_generate_diffusion_defects.py
scripts/05_train_unet_segmentation.py
scripts/06_filter_synthetic_quality.py
```

其中 `scripts/06_filter_synthetic_quality.py` 修复了一个第 8 阶段暴露的问题：原脚本的汇总逻辑仍然按 tile 缺陷类型写死，导致 wood 的 `filter_summary.csv` 和 `summary.md` 为空。现在汇总会从输入数据中动态读取缺陷类别。

## 3. wood 数据集结构

wood 数据集统计如下：

```text
train/good: 247
test/good: 19
test/color: 8
test/combined: 11
test/hole: 10
test/liquid: 10
test/scratch: 21
```

mask 完整性：

```text
color: 8 / 8
combined: 11 / 11
hole: 10 / 10
liquid: 10 / 10
scratch: 21 / 21
missing masks: 0
```

真实缺陷面积范围：

```text
min area ratio: 0.263%
mean area ratio: 5.085%
max area ratio: 41.269%
```

这说明 wood 的真实缺陷面积跨度很大，尤其 `scratch` 中存在很大面积的划痕区域，这会比 tile 更难用简单规则生成器覆盖。

## 4. 类别配置改造

第 8 阶段前，生成和训练脚本中存在 hard-code：

```text
DEFECT_TYPES = ["crack", "glue_strip", "gray_stroke", "oil", "rough"]
```

这意味着脚本本质上只能服务 `tile`。

本阶段改成类别配置：

```text
tile:
  crack
  glue_strip
  gray_stroke
  oil
  rough

wood:
  color
  hole
  liquid
  scratch
  combined
```

这样 `--category wood` 时，生成、Diffusion 和 U-Net 评估都会自动使用 wood 的缺陷类别。

## 5. wood traditional 生成规则

本阶段为 wood 新增五类规则生成：

```text
color:
  深色局部色斑，模拟木材表面变色。

hole:
  深色不规则破损区域，模拟木材孔洞或局部破洞。

liquid:
  复用 oil/liquid stain 思路，模拟木材表面液体污渍。

scratch:
  复用线状缺陷思路，生成较细、较浅的木材划痕。

combined:
  随机组合两种 wood 缺陷，例如 scratch + color 或 hole + liquid。
```

这一步的目标不是完美拟合 wood 真实缺陷，而是先完成跨类别迁移的最小闭环。

## 6. wood Diffusion prompt

wood 的 Diffusion prompt 加入了类别和材质约束：

```text
wood surface
industrial inspection
natural grain texture
visible local defect
```

示例：

```text
scratch:
  a realistic thin scratch defect on industrial wood surface,
  shallow bright irregular scratch lines,
  natural grain texture,
  inspection image
```

Diffusion 流程仍然使用 traditional mask 做 inpainting，不改变第 3 阶段以来的主流程。

## 7. 运行命令

数据探索：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/01_explore_dataset.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category wood --output-dir outputs/eda
```

traditional 生成：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category wood --samples-per-type 20 --seed 304 --output-dir outputs/stage8_wood_synthetic/traditional
```

Diffusion 生成：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/03_generate_diffusion_defects.py --category wood --traditional-summary outputs/stage8_wood_synthetic/traditional/wood/summary.csv --samples-per-type 10 --num-inference-steps 30 --seed 304 --local-files-only --output-dir outputs/stage8_wood_synthetic/diffusion
```

质量筛选：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/06_filter_synthetic_quality.py --traditional-summary outputs/stage8_wood_synthetic/traditional/wood/summary.csv --diffusion-summary outputs/stage8_wood_synthetic/diffusion/wood/summary.csv --output-dir outputs/stage8_wood_quality_filter/wood
```

U-Net combined 训练：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category wood --image-size 256 --epochs 30 --batch-size 4 --seed 304 --traditional-summary outputs/stage8_wood_quality_filter/wood/accepted_traditional_summary.csv --diffusion-summary outputs/stage8_wood_quality_filter/wood/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_stage8_wood --experiments combined
```

也可以使用一键入口：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/08_run_wood_generalization.py --local-files-only
```

说明：

```text
本阶段使用环境内 python.exe 直接执行，而不是 conda run。
原因是 Windows 下 conda run 打印中文路径时可能触发 GBK 编码错误。
```

## 8. 输出目录

主要输出：

```text
outputs/eda/wood
outputs/stage8_wood_synthetic/traditional/wood
outputs/stage8_wood_synthetic/diffusion/wood
outputs/stage8_wood_quality_filter/wood
outputs/training/unet_segmentation_stage8_wood/wood
```

关键文件：

```text
outputs/stage8_wood_synthetic/traditional/wood/summary.csv
outputs/stage8_wood_synthetic/diffusion/wood/summary.csv
outputs/stage8_wood_quality_filter/wood/summary.md
outputs/stage8_wood_quality_filter/wood/filter_summary.csv
outputs/training/unet_segmentation_stage8_wood/wood/comparison_summary.csv
outputs/training/unet_segmentation_stage8_wood/wood/combined/metrics.json
```

## 9. 质量筛选结果

本阶段生成：

```text
traditional: 100
diffusion: 50
total: 150
accepted: 144
rejected: 6
```

按类别筛选：

```text
traditional color: 20 / 20 accepted
traditional combined: 20 / 20 accepted
traditional hole: 20 / 20 accepted
traditional liquid: 19 / 20 accepted
traditional scratch: 16 / 20 accepted

diffusion color: 10 / 10 accepted
diffusion combined: 10 / 10 accepted
diffusion hole: 10 / 10 accepted
diffusion liquid: 10 / 10 accepted
diffusion scratch: 9 / 10 accepted
```

最明显的问题是 `scratch`：

```text
traditional scratch accept rate: 0.800
diffusion scratch accept rate: 0.900
```

这说明 wood 的 scratch 比 tile 的 crack 更难用当前简单线状规则稳定生成。

## 10. U-Net 真实测试集结果

第 8 阶段训练：

```text
experiment: combined
train samples: 144
test samples: 79
image size: 256
epochs: 30
batch size: 4
seed: 304
```

整体结果：

```text
Pixel Precision: 0.6367
Pixel Recall: 0.1674
Pixel F1 / Dice: 0.2651
Pixel IoU: 0.1528
Best Pixel F1: 0.2901
Image Precision: 0.9444
Image Recall: 0.8500
Image F1: 0.8947
```

按类别结果：

```text
color:
  Dice: 0.2650
  Recall: 0.2978

combined:
  Dice: 0.3210
  Recall: 0.2752

hole:
  Dice: 0.2960
  Recall: 0.2167

liquid:
  Dice: 0.5674
  Recall: 0.4392

scratch:
  Dice: 0.0247
  Recall: 0.0146
```

## 11. 结果说明

第 8 阶段的结论分两层。

第一层是正向结论：

```text
完整流程已经从 tile 迁移到 wood。
代码不再是单类别 hard-code demo。
wood 可以完成生成、筛选、训练和真实测试集评估。
Image F1 达到 0.8947，说明模型对缺陷有无的判断具备一定泛化能力。
```

第二层是问题暴露：

```text
wood 的像素级分割明显弱于 tile。
Pixel F1 只有 0.2651，主要受低 recall 影响。
scratch 类几乎失败，Dice 只有 0.0247。
```

这说明：

```text
1. 当前 tile 上调好的规则不能直接保证 wood 高质量分割。
2. wood scratch 的真实形态和当前生成 scratch 存在明显 domain gap。
3. wood 的真实缺陷面积跨度更大，尤其 scratch 可能是大面积、复杂纹理扰动。
4. 复用 U-Net 和简单规则生成可以跑通流程，但还不足以得到 tile 级别的像素定位效果。
```

这个结果并不是失败，而是泛化实验应该暴露的问题：

```text
跨类别迁移不仅要迁移代码流程，还要重新做类别级生成质量分析。
```

## 12. 和 tile 主线的关系

tile 主线最强结果：

```text
stage6 gray_stroke fixed combined:
  Pixel F1: 0.8573
  Image F1: 0.9492
  gray_stroke Dice: 0.8409

stage7 crack improved combined:
  Pixel F1: 0.8433
  Image F1: 0.9711
  crack Dice: 0.7589
```

wood 泛化结果：

```text
stage8 wood combined:
  Pixel F1: 0.2651
  Best Pixel F1: 0.2901
  Image F1: 0.8947
  liquid Dice: 0.5674
  scratch Dice: 0.0247
```

对比说明：

```text
tile 阶段证明了生成质量优化可以显著提高下游分割。
wood 阶段证明了流程可以迁移，但也说明每个新类别都需要重新做缺陷分布分析。
```

因此，项目现在从单类别实验升级为：

```text
一个可迁移的工业缺陷生成与检测实验框架。
```

## 13. 下一步建议

如果继续做第 9 阶段，建议不要盲目换更多类别，而是针对 wood 做一次像第 6、7 阶段那样的误差分析：

```text
1. 专项分析真实 wood scratch 的面积、长度、方向、灰度变化。
2. 重写 wood scratch 生成规则，让它覆盖大面积和复杂纹理划痕。
3. 对比 old wood scratch 与 real wood scratch 的分布。
4. 重新训练 wood combined，看 scratch Dice 是否提升。
```

面试表达可以这样说：

```text
我进一步把 tile 上的流程迁移到 wood，发现代码流程可以复用，但像素级效果明显下降，尤其 scratch 类失败。
这说明工业 AIGC 方案不能只写一个生成器到处套用，每个产品类别都需要重新分析真实缺陷分布。
```
