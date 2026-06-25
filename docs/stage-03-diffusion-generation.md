# 第 3 阶段：Diffusion Inpainting 缺陷生成

## 1. 本阶段目标

本阶段目标是在第 2 阶段传统规则伪缺陷的基础上，使用 Diffusion Inpainting 生成一小批更自然的工业缺陷样例。

本阶段仍然不训练模型，只做生成流程验证和小规模样例检查。

生成类别继续对齐 MVTec AD `tile` 的真实缺陷类型：

```text
crack
glue_strip
gray_stroke
oil
rough
```

默认每类生成 3 张，总计 15 张。

## 2. 本阶段做了什么

本阶段新增脚本：

```text
scripts/03_generate_diffusion_defects.py
```

运行命令：

```powershell
conda run -n industrial-defect-diffusion python scripts/03_generate_diffusion_defects.py --samples-per-type 3 --num-inference-steps 30 --seed 42 --local-files-only
```

本阶段使用的默认模型：

```text
stable-diffusion-v1-5/stable-diffusion-inpainting
```

关键参数：

```text
variant: fp16
use_safetensors: true
image_size: 512
num_inference_steps: 30
guidance_scale: 7.5
strength: 0.35
conditioning_image: traditional
```

输出结果：

```text
crack: 3
glue_strip: 3
gray_stroke: 3
oil: 3
rough: 3
total: 15
```

## 3. 为什么使用 Diffusion Inpainting

Inpainting 的核心思想是：

```text
给模型一张图，再给它一个 mask，只让它重绘 mask 指定的局部区域。
```

这非常适合工业缺陷生成，因为工业图像里的背景纹理很重要。我们不希望模型重新生成整张 tile 图，而是希望它只修改缺陷区域。

如果整张图都由生成模型重画，会出现两个问题：

1. 背景纹理可能被整体改变，导致图像不再像原始数据集。
2. 缺陷位置和 mask 很难严格对应，后续分割训练会不可靠。

因此，本阶段采用 Inpainting，而不是 text-to-image 直接生成整张缺陷图。

## 4. 为什么复用传统伪缺陷的 mask

本阶段没有重新随机生成 mask，而是复用第 2 阶段传统伪缺陷的 mask。

原因有三个：

1. 可以保证传统方法和 Diffusion 方法缺陷位置一致，方便公平对比。
2. 可以直接继承第 2 阶段已经验证过的输出结构和 metadata 追溯链路。
3. 可以把 Diffusion 的作用聚焦在“局部纹理真实化”，而不是重新解决 mask 生成问题。

最终每个 Diffusion 样本都能追溯到：

```text
原始 train/good 正常图
第 2 阶段传统合成图
第 2 阶段传统 mask
第 3 阶段 Diffusion 输出图
```

## 5. 为什么默认使用传统合成图作为条件图

最开始测试过直接使用正常图作为 Inpainting 条件图：

```text
conditioning_image = source
```

结果发现，通用 Stable Diffusion Inpainting 模型倾向于把 mask 区域补成正常纹理，缺陷可见性会变弱。

这说明一个重要问题：

```text
通用自然图 Diffusion 模型并不天然理解工业缺陷。
```

因此，本阶段默认改为：

```text
conditioning_image = traditional
```

也就是先让第 2 阶段传统规则方法给出缺陷的大致形态，再让 Diffusion 在这个基础上做局部重绘和纹理融合。

这样做更适合当前阶段，因为它把任务拆成了两个更容易理解的部分：

```text
传统方法：控制缺陷位置和大致形状
Diffusion：尝试让局部纹理和边缘更自然
```

## 6. 五类缺陷的 prompt 设计

本阶段对五类缺陷分别使用了不同 prompt：

```text
crack: long thin dark crack
glue_strip: pale translucent glue strip
gray_stroke: dark gray stroke smudge
oil: yellow brown translucent oil stain
rough: rough damaged texture
```

prompt 的设计原则不是写得越复杂越好，而是明确告诉模型三个信息：

1. 缺陷是什么类型。
2. 缺陷出现在 industrial ceramic tile surface 上。
3. 图像应该像 inspection image，而不是艺术图或普通照片。

同时使用 negative prompt 抑制不需要的内容：

```text
cartoon, painting, text, watermark, logo, unrealistic, oversaturated, blurry, distorted, object, people
```

## 7. 结果与输出

输出目录：

```text
outputs/diffusion_synthetic/tile/
```

输出结构：

```text
images/
masks/
metadata/
preview.png
traditional_vs_diffusion_preview.png
summary.csv
```

文件数量检查：

```text
images: 15
masks: 15
metadata: 15
preview.png: exists
traditional_vs_diffusion_preview.png: exists
summary.csv: exists
```

关键预览图：

```text
outputs/diffusion_synthetic/tile/preview.png
outputs/diffusion_synthetic/tile/traditional_vs_diffusion_preview.png
```

## 8. 结果说明

本阶段结果说明：

1. Diffusion Inpainting 生成流程已经跑通。
2. 每个样本都有对应二值 mask。
3. 每个样本都有 metadata，可以追溯到原始正常图、传统图、mask、prompt、seed 和模型参数。
4. 输出图尺寸保持原图尺寸，便于后续下游检测或分割实验使用。
5. 背景主体纹理没有被大面积破坏。

## 9. 哪些地方比传统方法更好

相对第 2 阶段传统伪缺陷，本阶段的优势主要体现在：

1. 局部缺陷区域和背景之间的融合更柔和。
2. `oil` 这类半透明斑块比纯规则叠加更像被重新渲染过。
3. `glue_strip` 的边缘比传统规则贴片更柔和。
4. 输出流程已经具备后续大规模生成的雏形。

## 10. 哪些地方仍然不够好

本阶段也暴露了明显局限：

1. 通用 Stable Diffusion Inpainting 模型不是工业缺陷专用模型。
2. `gray_stroke` 和 `rough` 这类细粒度纹理缺陷容易被模型弱化。
3. 如果 `strength` 太高，模型会倾向于把缺陷区域补回正常纹理。
4. 如果 `strength` 太低，Diffusion 的变化有限，更像传统缺陷的轻微平滑版本。
5. 当前结果只能说明生成流程可用，不能证明生成数据一定能提升检测效果。

这也是后续必须做下游实验的原因。

## 11. 遇到的问题

本阶段遇到两个主要问题。

第一个问题是默认模型 ID。

最初尝试的模型：

```text
stabilityai/stable-diffusion-2-inpainting
```

当前访问 Hugging Face 时返回 401，不能作为稳定默认模型。

解决方式：

```text
改用 stable-diffusion-v1-5/stable-diffusion-inpainting
```

第二个问题是模型权重下载和加载。

一开始模型缓存不完整，缺少关键 `unet` 权重。后来改为默认使用：

```text
variant: fp16
use_safetensors: true
```

这样可以减少不必要的大文件下载，也更适合 GPU 推理。

## 12. 为什么它仍然只是生成阶段

本阶段不要得出“Diffusion 已经证明有效”的结论。

原因是：

```text
生成图看起来更自然，不等于它能提升真实工业缺陷检测性能。
```

真正有说服力的结论必须来自下一步实验：

```text
把传统伪缺陷和 Diffusion 伪缺陷分别加入训练或验证流程，
再看真实 test 集上的检测 / 分割指标是否提升。
```

## 13. 下一步计划

下一步建议进入：

```text
异常检测 baseline
```

原因：

1. 现在已经有真实数据、传统伪缺陷、Diffusion 伪缺陷三类数据。
2. 需要建立一个下游评估流程，避免只凭肉眼评价生成图。
3. 对面试来说，最关键的问题是“生成数据是否真的有用”，而不是“能不能生成图片”。

下一阶段建议先做轻量 baseline，而不是立刻训练复杂分割网络。

推荐优先验证：

```text
PatchCore / PaDiM / 简单重建误差 baseline
```

如果环境和时间允许，再进入 U-Net 或其他监督分割实验。
