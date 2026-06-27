# 第 10 阶段：leather 第三类别泛化验证

## 1. 本阶段目标

第 10 阶段把当前流程从 `tile`、`wood` 继续扩展到第三个 MVTec AD 类别 `leather`。

目标不是立刻追求最高指标，而是验证：

```text
数据探索 -> traditional 生成 -> diffusion 生成 -> 质量筛选 -> U-Net combined 训练 -> 真实测试集评估
```

这条完整链路能否迁移到第三个工业表面类别。

选择 `leather` 的原因：

```text
1. leather 是表面纹理类，和 tile / wood 有连续性。
2. leather 缺陷类型直观，适合写规则生成器。
3. 缺陷面积整体不极端，适合作为第三类别泛化验证。
4. 相比 bottle 等对象类，leather 风险更可控，更适合先完成稳定闭环。
```

## 2. leather 数据结构

数据探索输出：

```text
outputs/eda/leather/dataset_summary.csv
```

leather 数据集结构：

```text
train/good: 245
test/good: 32
test/color: 19
test/cut: 19
test/fold: 17
test/glue: 19
test/poke: 18
missing masks: 0
```

真实缺陷面积：

```text
min area ratio: 0.089%
mean area ratio: 0.874%
max area ratio: 6.859%
```

这说明 leather 的真实缺陷整体偏小，像素级分割会比较考验 precision。

## 3. 类别配置扩展

本阶段在三个核心脚本中加入 `leather`：

```text
scripts/02_generate_traditional_defects.py
scripts/03_generate_diffusion_defects.py
scripts/05_train_unet_segmentation.py
```

新增 defect types：

```text
leather:
  color
  cut
  fold
  glue
  poke
```

这样 `--category leather` 时，生成、Diffusion prompt 和 U-Net 真实测试集评估都会自动使用 leather 的类别集合。

## 4. leather traditional 生成规则

新增五类 traditional 生成器：

```text
color:
  小面积局部深色变色斑。

cut:
  较细的局部切痕，带轻微弯曲和暗色核心。

fold:
  较宽的长条褶皱或压痕，模拟 leather 表面折痕。

glue:
  半透明浅色胶痕，类似短条状局部污染。

poke:
  小圆形或不规则孔点，面积控制较小。
```

traditional 生成规模：

```text
每类 20 张，共 100 张
seed = 504
```

traditional mask 面积分布：

```text
color mean area_ratio = 0.0056
cut mean area_ratio = 0.0020
fold mean area_ratio = 0.0569
glue mean area_ratio = 0.0140
poke mean area_ratio = 0.0038
```

## 5. leather Diffusion prompt

本阶段为 leather 五类新增 prompt。

共同约束：

```text
industrial leather surface
fine leather grain texture
inspection image
visible local defect
```

Diffusion 仍然使用 traditional mask 做 inpainting，没有改变主流程。

Diffusion 生成规模：

```text
每类 10 张，共 50 张
seed = 504
num inference steps = 30
local files only = true
```

## 6. 质量筛选结果

质量筛选输出：

```text
outputs/stage10_leather_quality_filter/leather
```

整体筛选结果：

```text
total: 150
accepted: 135
rejected: 15
```

按类别和来源：

```text
traditional color: 17 / 20 accepted
traditional cut: 15 / 20 accepted
traditional fold: 13 / 20 accepted
traditional glue: 20 / 20 accepted
traditional poke: 20 / 20 accepted

diffusion color: 10 / 10 accepted
diffusion cut: 10 / 10 accepted
diffusion fold: 10 / 10 accepted
diffusion glue: 10 / 10 accepted
diffusion poke: 10 / 10 accepted
```

`fold` 的 traditional accept rate 最低，主要原因是部分样本 inside diff 偏弱。这说明 leather fold 还可以作为后续专项优化候选。

## 7. U-Net 真实测试集结果

训练配置：

```text
category: leather
experiment: combined
train samples: 135
test samples: 124
image size: 256
epochs: 30
batch size: 4
seed: 504
```

整体结果：

```text
Pixel Precision = 0.0305
Pixel Recall = 0.5488
Pixel F1 = 0.0579
Best Pixel F1 = 0.1444
Image Precision = 0.7419
Image Recall = 1.0000
Image F1 = 0.8519
```

按类别结果：

```text
color:
  Dice = 0.2760
  Recall = 0.8886

cut:
  Dice = 0.0215
  Recall = 0.5031

fold:
  Dice = 0.2453
  Recall = 0.4317

glue:
  Dice = 0.1703
  Recall = 0.8299

poke:
  Dice = 0.2118
  Recall = 0.5480
```

## 8. 结果解释

第 10 阶段有一个清晰结论：

```text
leather 第三类别 pipeline 跑通，image-level 缺陷识别有效，但 pixel-level 分割明显过分割。
```

证据：

```text
Image F1 = 0.8519
Pixel Recall = 0.5488
Pixel Precision = 0.0305
Pixel F1 = 0.0579
Best Pixel F1 = 0.1444
```

模型对缺陷有无比较敏感，但像素预测范围过大，导致 precision 很低。

最弱类别是 `cut`：

```text
cut Dice = 0.0215
```

这说明 leather 的细小切痕和当前 synthetic cut 分布仍存在明显 gap。后续如果继续做第 11 阶段，建议不要再扩类别，而是针对 leather 做 precision / cut 专项分析。

## 9. 和 tile / wood 的关系

目前项目形成了三层泛化证据：

```text
tile:
  证明生成数据修复可以显著提升真实测试集分割效果。

wood:
  证明流程可以迁移到第二个表面类别，并能通过 scratch 专项修复改善失败类别。

leather:
  证明流程可以迁移到第三个表面类别，但也暴露了小面积缺陷下的 pixel precision 问题。
```

这让项目结论更完整：

```text
当前系统不是单类别 hard-code demo，而是一个可迁移的生成增强实验框架。
但每个新类别仍需要重新分析真实缺陷分布，尤其是小面积或细线状缺陷。
```

## 10. 运行命令

推荐使用环境内 Python：

```powershell
$PY="D:\miniforge3\envs\industrial-defect-diffusion\python.exe"
```

一键入口：

```powershell
& $PY scripts/10_run_leather_generalization.py --local-files-only
```

分步命令见 README 第 10 阶段部分。

## 11. 面试表达版本

可以这样表达：

```text
在 tile 和 wood 之后，我继续扩展到 leather，验证这套流程能否迁移到第三个工业表面类别。
我为 leather 加了 color、cut、fold、glue、poke 五类配置、传统生成规则和 Diffusion prompt。
完整跑通后，leather 的 Image F1 达到 0.8519，说明模型能识别缺陷有无。
但 Pixel F1 只有 0.0579，主要原因是 pixel precision 只有 0.0305，模型预测区域明显过大。
这说明流程具备跨类别迁移能力，但 leather 这种小面积缺陷类别需要进一步做 precision 和 cut 类专项修复。
```
