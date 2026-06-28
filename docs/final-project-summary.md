# 最终项目总结：工业缺陷生成与真实测试集分割验证

## 1. 项目目标

本项目面向工业视觉中的缺陷样本稀缺问题，目标不是单纯生成好看的缺陷图片，而是验证：

```text
synthetic defect 数据是否能在真实 MVTec AD 测试集上提升缺陷分割效果。
```

最终评价标准固定为真实测试集指标：

```text
Pixel Precision / Recall / F1
Best Pixel F1
Image F1
类别级 Dice / Recall
```

项目覆盖三个表面类 MVTec AD 类别：

```text
tile
wood
leather
```

## 2. 方法路线

整体流程：

```text
数据探索
-> traditional 规则缺陷生成
-> Diffusion Inpainting 局部缺陷生成
-> 基础质量筛选
-> U-Net 真实测试集分割验证
-> 类别级误差分析
-> 生成分布专项修复
-> 跨类别泛化验证
```

关键设计：

```text
1. Diffusion 不直接整图生成，而是使用 traditional mask 做 inpainting。
2. 每个 synthetic 样本都有 image / mask / metadata / summary，可追溯。
3. 每次生成改动都必须回到真实测试集验证，而不是只看预览图。
4. 失败类别通过真实分布统计和下游指标共同定位。
```

## 3. 阶段时间线

最终时间线由脚本自动汇总：

```text
outputs/final_report/final_experiment_timeline.md
```

概括如下：

| 阶段 | 主题 | 结果 |
| --- | --- | --- |
| Stage 1 | 数据探索 | 校验 MVTec AD tile 结构和 mask 对齐 |
| Stage 2 | traditional 生成 | 生成 tile 规则缺陷 baseline |
| Stage 3 | Diffusion Inpainting | 跑通局部 inpainting 缺陷生成 |
| Stage 4 | PatchCore baseline | 建立无 synthetic 数据的检测 baseline |
| Stage 5 | U-Net 小样本验证 | combined synthetic 数据首次证明有效 |
| Stage 6 | tile 扩大数据与 gray_stroke 修复 | 修复 gray_stroke 后得到 tile overall best |
| Stage 7 | tile crack 修复 | crack Dice / Recall 明显提升 |
| Stage 8 | wood 泛化 | 流程迁移到 wood，但 scratch 失败 |
| Stage 9 | wood scratch 修复 | wood overall 和 scratch 同时提升 |
| Stage 10 | leather 泛化 | 流程迁移到 leather，但严重过分割 |
| Stage 11 | leather precision / cut 修复 | 加入 good negatives 后修复 precision 和 cut |
| Stage 12 | leather fold 修复 | fold 召回大幅提升，但出现 precision tradeoff |
| Stage 13 | 最终结果整理 | 汇总最终指标、时间线和面试表达版本 |
| Stage 14 | 工程化复现与展示升级 | 增加共享配置、复现检查和最终 Dashboard |
| Stage 15 | 开源级健康检查 | 移除本机路径默认值，增加 configs、tests、CI 和 health check |
| Stage 16 | 最终展示资产 | 生成最终图表、论文式表格和项目卡片 |
| Stage 17 | 诊断证据汇总 | 汇总 threshold、postprocess、baseline 和分布修复证据 |

## 4. 最终指标总表

汇总文件：

```text
outputs/final_report/final_metrics_summary.csv
outputs/final_report/final_class_metrics.csv
outputs/final_report/final_results_dashboard.md
outputs/final_report/project_health_check.md
outputs/final_report/final_paper_tables.md
outputs/final_report/diagnostic_summary.md
outputs/final_report/figures/
```

关键实验：

| 实验 | 角色 | Pixel F1 | Best Pixel F1 | Image F1 | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| tile stage5 combined | baseline | 0.8064 | 0.8184 | 0.8615 | 小样本 combined 初次验证有效 |
| tile stage6 expanded combined | diagnostic_baseline | 0.7667 | 0.7836 | 0.8715 | 扩大数据后暴露 gray_stroke 失败 |
| tile stage6 gray_stroke fixed | overall_best | 0.8573 | 0.8719 | 0.9492 | tile overall 推荐模型 |
| tile stage7 crack fixed | class_specialist | 0.8433 | 0.8668 | 0.9711 | crack 专项最佳 |
| wood stage8 generalization | generalization_baseline | 0.2651 | 0.2901 | 0.8947 | wood 迁移跑通但 scratch 失败 |
| wood stage9 scratch fixed | overall_best | 0.3369 | 0.3815 | 0.9023 | wood overall 推荐模型 |
| leather stage10 generalization | generalization_baseline | 0.0579 | 0.1444 | 0.8519 | leather 跑通但严重过分割 |
| leather stage11 precision cut fixed | overall_best | 0.4774 | 0.5219 | 0.9667 | leather overall 推荐模型 |
| leather stage12 fold fixed | diagnostic_tradeoff | 0.3093 | 0.4011 | 0.9735 | fold 召回补强但牺牲 precision |

## 5. 最终推荐结果

### tile overall best

```text
Stage 6 gray_stroke fixed
Pixel F1 = 0.8573
Best Pixel F1 = 0.8719
Image F1 = 0.9492
gray_stroke Dice = 0.8409
```

为什么推荐它：

```text
默认 Pixel F1 最高，且修复了 Stage 6 expanded 中 gray_stroke 几乎失败的问题。
```

### tile crack specialist

```text
Stage 7 crack fixed
Pixel F1 = 0.8433
Best Pixel F1 = 0.8668
Image F1 = 0.9711
crack Dice = 0.7589
crack Recall = 0.6841
```

为什么不替代 tile overall best：

```text
crack 显著提升，但 overall Pixel F1 从 0.8573 小幅降到 0.8433。
```

### wood overall best

```text
Stage 9 scratch fixed
Pixel F1 = 0.3369
Best Pixel F1 = 0.3815
Image F1 = 0.9023
scratch Dice = 0.3405
scratch Recall = 0.4169
```

相对 Stage 8：

```text
scratch Dice: 0.0247 -> 0.3405
scratch Recall: 0.0146 -> 0.4169
Pixel F1: 0.2651 -> 0.3369
```

### leather overall best

```text
Stage 11 precision / cut fixed
Pixel Precision = 0.8752
Pixel F1 = 0.4774
Best Pixel F1 = 0.5219
Image F1 = 0.9667
cut Dice = 0.4064
```

相对 Stage 10：

```text
Pixel Precision: 0.0305 -> 0.8752
Pixel F1: 0.0579 -> 0.4774
cut Dice: 0.0215 -> 0.4064
```

### leather fold tradeoff

```text
Stage 12 fold fixed
Pixel F1 = 0.3093
Best Pixel F1 = 0.4011
Image F1 = 0.9735
fold Dice = 0.4873
fold Recall = 0.6660
```

为什么它不是 leather overall best：

```text
fold Dice 大幅提升，但 Pixel Precision 从 0.8752 降到 0.2004，
overall Pixel F1 从 0.4774 降到 0.3093。
```

## 6. 关键失败案例与修复逻辑

### gray_stroke

Stage 6 expanded combined 中：

```text
gray_stroke Dice = 0.0022
```

问题：

```text
synthetic gray_stroke 的面积、颜色和真实分布不匹配。
```

修复后：

```text
gray_stroke Dice = 0.8409
Pixel F1 = 0.8573
```

### crack

Stage 6 gray_stroke fixed 中：

```text
crack Dice = 0.6120
crack Recall = 0.4732
```

修复：

```text
改为边界贯穿式、分叉式 crack，并强化 Diffusion prompt。
```

Stage 7：

```text
crack Dice = 0.7589
crack Recall = 0.6841
```

### wood scratch

Stage 8：

```text
scratch Dice = 0.0247
scratch Recall = 0.0146
```

真实分布问题：

```text
真实 scratch mean area_ratio ~= 0.0744
旧 synthetic scratch mean area_ratio ~= 0.0020
```

Stage 9 修复后：

```text
scratch Dice = 0.3405
scratch Recall = 0.4169
```

### leather cut / precision

Stage 10：

```text
Pixel Precision = 0.0305
good image_score mean ~= 0.994
cut Dice = 0.0215
```

问题：

```text
训练中只有 synthetic defect 正样本，没有真实 train/good 空 mask 负样本。
```

Stage 11 修复：

```text
加入 100 张 train/good 空 mask negative samples
修复 cut 生成分布
```

结果：

```text
Pixel Precision = 0.8752
Pixel F1 = 0.4774
cut Dice = 0.4064
```

### leather fold

Stage 11：

```text
fold Dice = 0.0972
fold Recall = 0.0571
```

Stage 12 修复：

```text
fold 生成改为更窄、更浅的 ridge / shadow 褶皱
追加高质量 traditional fold
good negatives 增加到 200
```

结果：

```text
fold Dice = 0.4873
fold Recall = 0.6660
```

但：

```text
overall Pixel F1 = 0.3093，低于 Stage 11 的 0.4774。
```

所以 Stage 12 是 tradeoff 分析，不是 overall best。

## 7. 最终结论

本项目最终证明了三件事：

```text
1. synthetic defect 数据确实可以提升真实测试集分割效果。
2. 生成数据不是越多越好，类别分布匹配比数量更关键。
3. 跨类别迁移可行，但每个类别仍需要误差分析和生成分布修复。
```

最终项目定位：

```text
一个可迁移的工业表面缺陷 synthetic data + U-Net 分割验证框架，
核心亮点是可解释的类别级生成修复，而不是单纯生成图片。
```

## 8. 复现入口

推荐先配置环境和数据路径：

```powershell
conda activate industrial-defect-diffusion
$env:PYTHONUTF8="1"
$env:DATA_ROOT="<path-to-MVTec_AD>"
```

快速复现最终结果表：

```powershell
python scripts/13_collect_final_results.py
python scripts/14_generate_final_dashboard.py
python scripts/14_reproduction_check.py
python scripts/15_project_health_check.py
python scripts/16_generate_final_visuals.py
python scripts/17_collect_diagnostics.py
```

输出：

```text
outputs/final_report/final_metrics_summary.csv
outputs/final_report/final_class_metrics.csv
outputs/final_report/final_experiment_timeline.md
outputs/final_report/final_results_dashboard.md
outputs/final_report/reproduction_check.md
outputs/final_report/project_health_check.md
outputs/final_report/final_paper_tables.md
outputs/final_report/diagnostic_summary.md
```

重新训练推荐模型时，统一使用 `--data-root "$env:DATA_ROOT"`，不要把个人电脑的绝对路径写进命令或文档。

如果要检查完整训练环境和数据集，可运行：

```powershell
python scripts/14_reproduction_check.py --data-root "$env:DATA_ROOT" --strict
```

关键推荐模型：

```text
tile overall: outputs/training/unet_segmentation_gray_stroke_fix/tile/combined/metrics.json
wood overall: outputs/training/unet_segmentation_stage9_wood_scratch_fix/wood/combined/metrics.json
leather overall: outputs/training/unet_segmentation_stage11_leather_precision_cut_fix/leather/combined/metrics.json
leather fold tradeoff: outputs/training/unet_segmentation_stage12_leather_fold_fix/leather/combined/metrics.json
```

阶段详情见：

```text
docs/stage-01-data-exploration.md
...
docs/stage-12-leather-fold-fix.md
docs/stage-14-engineering-reproducibility.md
docs/stage-15-open-source-health-checks.md
docs/stage-16-final-visuals-and-paper-tables.md
docs/stage-17-diagnostic-evidence-and-ablation-summary.md
docs/project-card.md
```
