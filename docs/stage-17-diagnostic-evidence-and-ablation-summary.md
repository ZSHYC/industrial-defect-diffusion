# 第 17 阶段：诊断证据与消融汇总

## 本阶段目标

第 17 阶段不重新训练模型，也不修改最终推荐结果。目标是把已经存在但分散的诊断结果整理成一套可复现的证据链。

核心问题：

```text
最终结论是否只来自单次指标，还是能被 threshold、postprocess、baseline 和分布统计共同解释？
```

## 新增脚本

```text
scripts/17_collect_diagnostics.py
```

运行：

```powershell
python scripts/17_collect_diagnostics.py
```

## 汇总输出

```text
outputs/final_report/diagnostic_summary.md
outputs/final_report/threshold_postprocess_summary.csv
outputs/final_report/distribution_repair_summary.csv
outputs/final_report/baseline_comparison_summary.csv
```

新增诊断图：

```text
outputs/final_report/figures/threshold_sweep_leather.png
outputs/final_report/figures/distribution_repair_summary.png
outputs/final_report/figures/baseline_vs_synthetic.png
```

## 诊断内容

### Threshold / Postprocess

汇总 leather Stage 11 和 Stage 12 的：

```text
default threshold = 0.5
best threshold
best postprocess setting
Pixel Precision / Recall / F1
```

结论：

```text
Stage 11 仍是 leather overall best。
Stage 12 的 fold recall 可以补强，但默认 Pixel F1 和 precision tradeoff 使其保持 diagnostic specialist。
```

### PatchCore Baseline

汇总 tile PatchCore-style baseline 与 synthetic-data U-Net：

```text
PatchCore image-level 很强
Synthetic-data U-Net pixel-level segmentation 更强
```

这说明项目不是单纯替代 anomaly detection，而是在真实 mask 分割目标上验证 synthetic data。

### Distribution Repair

汇总：

```text
tile/gray_stroke
tile/crack
wood/scratch
leather/cut
leather/fold
```

对比：

```text
real mean_area_ratio
old synthetic mean_area_ratio
new synthetic mean_area_ratio
Dice before
Dice after
```

核心证据：

```text
类别级生成分布修复通常伴随真实测试集 Dice 提升。
```

## 验收

```powershell
python scripts/17_collect_diagnostics.py
python -m py_compile scripts/17_collect_diagnostics.py
python scripts/15_project_health_check.py
```

第 17 阶段完成后，项目从“结果可展示”进一步升级为“诊断证据链更扎实”。

