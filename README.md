# 基于 Diffusion Inpainting 的工业缺陷图像生成与分割验证

## 项目简介

本项目面向工业视觉缺陷样本稀缺问题，构建了一套可复现、可解释的 synthetic defect 数据增强与真实测试集分割验证流程。

核心问题不是：

```text
Diffusion 图片看起来像不像缺陷。
```

而是：

```text
synthetic defect 数据是否能在真实 MVTec AD 测试集上提升分割效果。
```

完整流程：

```text
数据探索
-> traditional 规则伪缺陷生成
-> Diffusion Inpainting 局部缺陷生成
-> 质量筛选
-> U-Net 真实测试集分割评估
-> 类别级误差分析
-> 生成分布专项修复
-> 跨类别泛化验证
```

当前已覆盖三个 MVTec AD 表面类别：

```text
tile
wood
leather
```

## 最终结论

本项目最终证明：

```text
1. synthetic defect 数据可以提升真实测试集分割效果。
2. 生成数据不是越多越好，类别分布匹配比数量更关键。
3. 跨类别迁移可行，但每个类别仍需要误差分析和生成分布修复。
4. 正常图空 mask negative samples 对 leather 这类易过分割类别非常关键。
```

最终项目定位：

```text
一个可迁移的工业表面缺陷 synthetic data + U-Net 分割验证框架，
核心亮点是可解释的类别级生成修复，而不是单纯生成图片。
```

## 最终推荐结果

| 类别 / 目标 | 推荐阶段 | Pixel F1 | Best Pixel F1 | Image F1 | 关键类别指标 |
| --- | --- | ---: | ---: | ---: | --- |
| tile overall | Stage 6 gray_stroke fixed | 0.8573 | 0.8719 | 0.9492 | gray_stroke Dice = 0.8409 |
| tile crack specialist | Stage 7 crack fixed | 0.8433 | 0.8668 | 0.9711 | crack Dice = 0.7589, Recall = 0.6841 |
| wood overall | Stage 9 scratch fixed | 0.3369 | 0.3815 | 0.9023 | scratch Dice = 0.3405, Recall = 0.4169 |
| leather overall | Stage 11 precision / cut fixed | 0.4774 | 0.5219 | 0.9667 | cut Dice = 0.4064 |
| leather fold tradeoff | Stage 12 fold fixed | 0.3093 | 0.4011 | 0.9735 | fold Dice = 0.4873, Recall = 0.6660 |

说明：

```text
Stage 6 是 tile overall best。
Stage 9 是 wood overall best。
Stage 11 是 leather overall best。
Stage 12 是 fold 召回补强实验，不是 leather overall best。
```

## 关键实验故事

### tile: gray_stroke 修复

Stage 6 expanded combined 暴露出：

```text
gray_stroke Dice = 0.0022
```

通过类别级分布分析修复生成规则后：

```text
Pixel F1 = 0.8573
gray_stroke Dice = 0.8409
```

### tile: crack 专项提升

Stage 7 修复 crack 生成分布后：

```text
crack Dice = 0.7589
crack Recall = 0.6841
Image F1 = 0.9711
```

但 default Pixel F1 小幅低于 Stage 6，因此作为 crack specialist。

### wood: scratch 修复

Stage 8 wood 流程跑通，但：

```text
scratch Dice = 0.0247
scratch Recall = 0.0146
```

修复 scratch 从细小局部线条到大范围纹理扰动后：

```text
scratch Dice = 0.3405
scratch Recall = 0.4169
wood Pixel F1 = 0.3369
```

### leather: good negatives 修复过分割

Stage 10 leather 泛化时：

```text
Pixel Precision = 0.0305
good image_score mean ~= 0.994
```

说明模型对正常 leather 纹理也高响应。Stage 11 加入真实 `train/good` 空 mask negative samples 并修复 cut 后：

```text
Pixel Precision = 0.8752
Pixel F1 = 0.4774
cut Dice = 0.4064
```

### leather: fold tradeoff

Stage 12 修复 fold 后：

```text
fold Dice = 0.4873
fold Recall = 0.6660
```

但：

```text
Pixel F1 = 0.3093
Pixel Precision = 0.2004
```

所以 Stage 12 是类别召回补强和 precision / recall tradeoff 分析，不作为 overall best。

## 文档入口

最终汇总：

- [最终项目总结](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/final-project-summary.md)
- [面试表达稿](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/interview-talking-points.md)
- [环境安装记录](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/environment-setup.md)

阶段文档：

- [第 1 阶段：MVTec AD 数据探索与校验](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-01-data-exploration.md)
- [第 2 阶段：传统规则伪缺陷生成](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-02-traditional-synthesis.md)
- [第 3 阶段：Diffusion Inpainting 缺陷生成](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-03-diffusion-generation.md)
- [第 4 阶段：PatchCore 风格无监督异常检测 Baseline](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-04-patchcore-baseline.md)
- [第 5 阶段：U-Net 监督分割训练与生成数据增强对比](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-05-unet-segmentation.md)
- [第 6 阶段：扩大生成数据、质量筛选与监督分割再验证](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-06-expanded-synthesis-and-filtering.md)
- [第 7 阶段：crack 专项改进与最终实验整理](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-07-crack-improvement-and-final-analysis.md)
- [第 8 阶段：wood 类别泛化验证与复现实验包](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-08-wood-generalization.md)
- [第 9 阶段：wood scratch 专项修复与跨类别误差分析](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-09-wood-scratch-fix.md)
- [第 10 阶段：leather 第三类别泛化验证](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-10-leather-generalization.md)
- [第 11 阶段：leather precision / cut 专项修复](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-11-leather-precision-cut-fix.md)
- [第 12 阶段：leather fold 专项修复与保守模型召回补强](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-12-leather-fold-fix.md)

## 复现入口

默认数据路径：

```text
C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD
```

推荐 Python：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe
```

收集最终指标汇总：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/13_collect_final_results.py
```

输出：

```text
outputs/final_report/final_metrics_summary.csv
outputs/final_report/final_class_metrics.csv
outputs/final_report/final_experiment_timeline.md
```

重新运行关键最终模型示例：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category leather --image-size 256 --epochs 30 --batch-size 4 --seed 604 --traditional-summary outputs/stage11_leather_precision_cut_fix/quality_filter/leather/accepted_traditional_summary.csv --diffusion-summary outputs/stage11_leather_precision_cut_fix/quality_filter/leather/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_stage11_leather_precision_cut_fix --experiments combined --good-negative-samples 100
```

## 主要脚本

```text
scripts/01_explore_dataset.py
scripts/02_generate_traditional_defects.py
scripts/03_generate_diffusion_defects.py
scripts/04_patchcore_baseline.py
scripts/05_train_unet_segmentation.py
scripts/06_filter_synthetic_quality.py
scripts/07_analyze_crack_distribution.py
scripts/08_run_wood_generalization.py
scripts/09_analyze_wood_scratch_distribution.py
scripts/09_prepare_wood_scratch_fix_dataset.py
scripts/10_run_leather_generalization.py
scripts/11_analyze_leather_cut_distribution.py
scripts/11_prepare_leather_cut_fix_dataset.py
scripts/12_analyze_leather_fold_distribution.py
scripts/12_prepare_leather_fold_fix_dataset.py
scripts/13_collect_final_results.py
```

## 项目结构

```text
industrial-defect-diffusion/
  README.md
  AGENTS.md
  requirements.txt
  configs/
  docs/
  scripts/
  src/
  outputs/
```

目录职责：

```text
README.md：最终入口和推荐结果
docs/：阶段文档、最终总结、面试稿
scripts/：可复现实验脚本
outputs/：实验输出、CSV、metrics、预览图
```

## 维护原则

```text
1. README 只保留长期有效的项目入口和最终结论。
2. 阶段过程写入 docs/stage-*.md。
3. 最终汇总写入 docs/final-project-summary.md。
4. 不提交大量 PNG、模型权重、预测 mask。
5. 提交代码、文档、关键 CSV/JSON 指标。
```
