# 面试表达稿：工业缺陷生成与分割验证项目

## 1 分钟版本

这个项目解决的是工业缺陷样本少、标注贵的问题。

我没有只做一个 Diffusion 生成图片 demo，而是把它做成了完整验证闭环：

```text
传统规则生成 -> Diffusion Inpainting -> U-Net 分割训练 -> 真实 MVTec AD 测试集评估 -> 类别级误差分析和修复
```

项目覆盖了 `tile`、`wood`、`leather` 三个表面类类别。

最重要的结论是：生成图片看起来像不代表有用，必须看真实测试集指标。比如 `tile` 上我发现扩大生成数据后 `gray_stroke` 几乎失败，Dice 只有 `0.0022`。通过分析真实和 synthetic 分布并修复生成规则后，`gray_stroke Dice` 提升到 `0.8409`，整体 `Pixel F1` 提升到 `0.8573`。

后面我又在 `wood/scratch` 和 `leather/cut` 上做了类似修复，证明这套方法可以迁移，但每个类别都要做分布分析。

## 3 分钟版本

项目从 MVTec AD 的 `tile` 开始。

第一步是数据探索，确认真实异常图和 mask 对齐。然后我做了两套 synthetic defect：

```text
traditional：用规则方法控制缺陷形状和 mask
diffusion：用 Stable Diffusion Inpainting 在 mask 区域做局部重绘
```

这里我没有直接 text-to-image 生成整图，因为工业图像的背景纹理很重要，整图生成会破坏真实数据分布。我的策略是让 traditional 负责 mask 和大致形状，Diffusion 负责局部纹理融合。

之后我训练 U-Net，在真实测试集上评估。小样本阶段 combined 效果最好：

```text
tile stage5 combined Pixel F1 = 0.8064
```

扩大数据后我发现一个反直觉现象：更多数据不一定更好。Stage 6 expanded combined 的 `Pixel F1` 是 `0.7667`，而 `gray_stroke Dice` 只有 `0.0022`。这说明问题不是模型结构，而是某个类别的 synthetic 分布错了。

我做了类别级误差分析，修复 `gray_stroke` 的面积、颜色和形态后：

```text
Pixel F1 = 0.8573
gray_stroke Dice = 0.8409
```

这成为 `tile` 的 overall best。

然后我继续做跨类别验证。`wood` 首次迁移时流程跑通，但 `scratch Dice` 只有 `0.0247`。分析发现真实 scratch 是大范围纹理扰动，而旧 synthetic scratch 太细太小。修复后：

```text
wood Pixel F1 = 0.3369
scratch Dice = 0.3405
```

第三个类别 `leather` 更有挑战。Stage 10 的 `Pixel Precision` 只有 `0.0305`，good 图也高响应，说明模型严重过分割。原因是训练集中只有 synthetic defect 正样本，没有真实正常图负样本。所以 Stage 11 我加入 100 张 `train/good` 空 mask negative samples，并修复 cut：

```text
Pixel Precision = 0.8752
Pixel F1 = 0.4774
cut Dice = 0.4064
```

最后 Stage 12 我修了 fold。fold Dice 从 `0.0972` 提到 `0.4873`，但 overall Pixel F1 降到 `0.3093`，所以我没有把它包装成最佳模型，而是把它作为 precision / recall tradeoff 分析。

## 5 分钟版本

这个项目最核心的工程判断是：生成模型的价值不能靠视觉主观评价，必须通过真实测试集下游任务验证。

我把项目拆成三个层次。

第一层是生成闭环。

我先做 traditional synthetic defects，因为它可控、可解释、能提供 mask。然后基于 traditional mask 做 Diffusion Inpainting。这样 Diffusion 不负责凭空创造整张工业图，而是只在局部缺陷区域做纹理融合。这能保证输出仍然和原始 MVTec AD 背景分布接近。

第二层是下游验证。

我用 U-Net 做监督分割，并在真实 test set 上评估 Pixel F1、Image F1 和类别 Dice。Stage 5 已经看到 combined synthetic 数据有明显价值：

```text
traditional Pixel F1 = 0.1351
diffusion Pixel F1 = 0.5003
combined Pixel F1 = 0.8064
```

这说明 traditional 和 diffusion 是互补的。

第三层是误差分析和修复。

最典型的是 `gray_stroke`。Stage 6 扩大数据后，整体 Pixel F1 是 `0.7667`，但 `gray_stroke Dice` 只有 `0.0022`。如果只看 overall，很容易误判为模型不行；但类别级分析说明是 synthetic gray_stroke 分布不匹配。修复后：

```text
Pixel F1 = 0.8573
gray_stroke Dice = 0.8409
```

这个结果是项目主线最强证据。

同样的方法迁移到 `wood`。Stage 8 跑通了流程，但 `scratch Dice` 只有 `0.0247`。我统计真实和 synthetic scratch，发现真实 scratch 的面积大得多，是大范围浅色磨损，而 synthetic scratch 只有很细的线。修复后：

```text
scratch Dice = 0.3405
wood Pixel F1 = 0.3369
```

再迁移到 `leather` 时，问题变成过分割。Stage 10 的 Pixel Precision 只有 `0.0305`，good 图 image_score 均值接近 `0.994`。这说明模型学到的是“leather 纹理都可能异常”。根因是训练集中没有真实正常图空 mask。因此 Stage 11 我给 U-Net 增加了 `--good-negative-samples`，加入 100 张真实 `train/good` 空 mask 图，并修复 cut：

```text
Pixel Precision = 0.8752
Pixel F1 = 0.4774
cut Dice = 0.4064
```

这说明 synthetic defect 增强不能只补正样本，还要用正常样本约束决策边界。

Stage 12 我继续修 fold，fold Dice 从 `0.0972` 提到 `0.4873`，Recall 从 `0.0571` 到 `0.6660`。但 overall Pixel F1 降到 `0.3093`，因为模型变得更偏 recall，precision 下降。我保留了这个结果，但没有把它说成最佳模型。最终 leather 推荐 Stage 11，Stage 12 是一个 tradeoff 分析。

最终推荐结果是：

```text
tile overall: Stage 6 gray_stroke fixed, Pixel F1 = 0.8573
wood overall: Stage 9 scratch fixed, Pixel F1 = 0.3369
leather overall: Stage 11 precision / cut fixed, Pixel F1 = 0.4774
leather fold specialist: Stage 12, fold Dice = 0.4873
```

如果总结成一句话：

```text
这个项目证明了 synthetic defect 数据可以提升工业缺陷分割，但真正关键的是类别级生成分布匹配，而不是单纯生成更多图片。
```

## 常见追问

### 为什么不用 Diffusion 直接生成整图？

因为工业视觉背景纹理很重要。整图生成容易改变正常纹理分布，也很难保证 mask 对齐。Inpainting 可以只修改局部缺陷区域，更适合分割训练。

### 为什么 traditional 还需要保留？

traditional 方法负责可控形状、mask 和对照组。Diffusion 负责局部纹理融合。没有 traditional baseline，就很难证明 Diffusion 是否真的带来增益。

### 为什么 Stage 12 不作为最佳模型？

因为 Stage 12 虽然把 fold Dice 从 `0.0972` 提升到 `0.4873`，但整体 Pixel F1 从 `0.4774` 降到 `0.3093`。这说明它是 fold 召回补强实验，不是 overall 最优模型。

### 这个项目最大的亮点是什么？

不是会调用 Diffusion，而是把生成模型放进了真实工业缺陷分割验证闭环，并通过类别级误差分析持续修复生成分布。
