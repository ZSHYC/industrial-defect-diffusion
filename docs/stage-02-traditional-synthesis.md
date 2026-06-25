# 第 2 阶段：传统规则伪缺陷生成

## 1. 本阶段目标

本阶段目标是基于 MVTec AD `tile/train/good` 的正常图像，生成一小批与 `tile` 真实缺陷类别对齐的传统规则伪缺陷。

本阶段只做传统图像处理合成，不使用 Diffusion，不训练模型。

生成类别对齐 `tile` 数据集真实缺陷类型：

```text
crack
glue_strip
gray_stroke
oil
rough
```

## 2. 本阶段做了什么

本阶段新增脚本：

```text
scripts/02_generate_traditional_defects.py
```

运行命令：

```powershell
conda run -n industrial-defect-diffusion python scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --samples-per-type 5 --seed 42
```

生成结果：

```text
crack: 5
glue_strip: 5
gray_stroke: 5
oil: 5
rough: 5
total: 25
```

每个样本包含：

```text
合成图
二值 mask
metadata JSON
```

## 3. 为什么先做传统伪缺陷

传统伪缺陷是后续 Diffusion 缺陷生成的对照组。

如果直接做 Diffusion，只能说明“生成模型能生成图”，但不能说明它比简单规则方法更有价值。

先做传统伪缺陷有三个作用：

1. 建立最基础的 synthetic defect baseline。
2. 验证图像、mask、metadata、preview 的输出流程。
3. 为后续对比 Diffusion 生成缺陷提供参照。

## 4. 五类缺陷生成方式

### 4.1 crack

生成细长黑色裂纹。

方法：

```text
随机折线
轻微分叉
随机线宽
轻微高斯模糊
```

### 4.2 glue_strip

生成半透明胶条或贴片。

方法：

```text
随机旋转长条多边形
浅灰/浅白色 tint
透明融合
边缘轻微模糊
```

### 4.3 gray_stroke

生成灰色擦痕或笔触。

方法：

```text
较宽随机曲线
灰色覆盖
柔和边缘
局部透明融合
```

### 4.4 oil

生成油污状不规则斑块。

方法：

```text
随机不规则多边形
偏黄/偏暗 tint
大范围模糊边缘
半透明融合
```

### 4.5 rough

生成局部粗糙纹理扰动。

方法：

```text
随机不规则区域
局部噪声
局部对比度增强
保持原始纹理背景
```

## 5. 结果与输出

输出目录：

```text
outputs/traditional_synthetic/tile/
```

输出结构：

```text
images/
masks/
metadata/
preview.png
summary.csv
```

文件数量：

```text
images: 25
masks: 25
metadata: 25
```

预览图：

```text
outputs/traditional_synthetic/tile/preview.png
```

汇总表：

```text
outputs/traditional_synthetic/tile/summary.csv
```

## 6. 结果说明

本阶段结果说明：

1. 五类传统伪缺陷都可以从正常 `tile` 图中生成。
2. 每张合成图都有对应二值 mask。
3. 每个样本都有 metadata，可以追溯到原始正常图和生成参数。
4. `preview.png` 能展示原图、合成图、mask 和 overlay，便于人工检查。

## 7. 哪些地方像真实缺陷

相对接近真实缺陷的地方：

1. `crack` 的细线和分叉与真实裂纹有相似形态。
2. `glue_strip` 的半透明长条区域和真实胶条类别有一定对应关系。
3. `oil` 的半透明不规则斑块与油污扩散效果接近。

## 8. 哪些地方明显不真实

传统规则方法的局限也很明显：

1. 部分边缘仍然偏人工。
2. `gray_stroke` 和 `rough` 的纹理变化不够自然。
3. 合成区域和真实成像物理规律之间仍有差距。
4. 规则方法很难学习真实缺陷中的复杂纹理、透明度和局部结构变化。

## 9. 为什么它只是 baseline

传统伪缺陷的价值不是最终效果，而是提供可解释的对照组。

后续如果 Diffusion 生成图更自然，或者能让下游模型在真实测试集上获得更高召回率和更好分割指标，就可以说明 Diffusion 相比传统规则合成更有价值。

## 10. 遇到的问题

第一次使用 `conda run` 运行时，脚本已经生成文件，但 `conda run` 在输出中文路径时触发了 Windows `gbk` 编码错误。

解决方式：

```text
脚本打印输出改为相对路径和 ASCII 文本。
```

第二次运行成功，无编码报错。

## 11. 下一步计划

下一步建议进入：

```text
Diffusion Inpainting 缺陷生成
```

原因：

1. 传统规则伪缺陷已经完成 baseline。
2. 现在可以对比 Diffusion 是否生成得更自然。
3. 后续可以验证 Diffusion 合成缺陷是否比传统规则缺陷更有助于下游检测或分割。

