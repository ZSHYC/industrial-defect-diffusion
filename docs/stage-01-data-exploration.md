# 第 1 阶段：MVTec AD 数据探索与校验

## 1. 本阶段目标

本阶段的目标是确认项目使用的数据集结构正确、异常图与 mask 能一一对应，并生成第一批可视化结果，为后续传统伪缺陷生成、Diffusion 缺陷生成和分割训练打基础。

## 2. 本阶段做了什么

本阶段完成了以下工作：

1. 创建项目目录骨架：

```text
industrial-defect-diffusion/
  configs/
  scripts/
  src/
  outputs/
```

2. 创建基础文件：

```text
README.md
requirements.txt
scripts/01_explore_dataset.py
```

3. 检查本地数据集路径：

```text
C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD
```

4. 确认 `tile` 类别目录结构完整：

```text
train/good
test/good
test/crack
test/glue_strip
test/gray_stroke
test/oil
test/rough
ground_truth/*
```

5. 统计 `tile` 类别中的正常图、异常图和 mask 数量。
6. 编写并运行数据探索脚本，生成：

```text
数据摘要 CSV
缺陷面积统计 CSV
正常样本图
异常图 + mask + overlay 叠加图
缺陷面积比例箱线图
```

## 3. 为什么这样做

这一步是整个项目的地基。

如果数据结构不清楚，后面会出现很多连锁问题：

1. 训练图和标签对不上。
2. 生成缺陷时不知道正常图从哪里取。
3. 评估时可能把异常图和 mask 搞错。
4. 面试时无法清楚说明数据集结构和实验基础。

先把数据探索做扎实，后面每一步都会更稳。

## 4. 结果与输出

### 4.1 数据统计结果

`tile` 类别统计如下：

```text
train/good: 230 张
test/good: 33 张
test/crack: 17 张
test/glue_strip: 18 张
test/gray_stroke: 16 张
test/oil: 18 张
test/rough: 15 张
```

异常图总数：

```text
84 张
```

mask 检查结果：

```text
缺失 mask: 0
```

### 4.2 缺陷面积比例

缺陷面积比例统计：

```text
最小值: 0.842%
平均值: 9.802%
最大值: 35.487%
```

### 4.3 生成文件

输出目录：

```text
outputs/eda/tile/
```

生成文件：

```text
dataset_summary.csv
defect_area_stats.csv
normal_train_samples.png
defect_mask_overlays.png
defect_area_ratio_boxplot.png
```

## 5. 结果说明

本阶段结果说明：

1. 数据路径正确，数据集可以稳定读取。
2. `tile` 类别结构完整，适合作为项目第一类实验对象。
3. 异常图和 mask 可以正确对应，后续可以安全用于分割训练与评估。
4. 当前数据探索脚本已经能够支撑后续实验复盘和面试展示。

## 6. 遇到的问题

本阶段没有遇到严重阻塞问题。

注意点有个：

1. Windows 终端里中文路径显示有编码乱码，但不影响实际文件读写。

## 7. 下一步计划

下一步建议做：

```text
传统规则伪缺陷生成
```

原因：

1. 它不依赖复杂深度学习环境。
2. 它可以快速产出第一批 synthetic defects。
3. 它是 Diffusion 缺陷生成的重要对照组。
4. 没有传统对照组，后面很难证明 Diffusion 的价值。
