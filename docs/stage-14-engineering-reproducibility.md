# 第 14 阶段：工程化复现与项目展示升级

## 本阶段目标

第 14 阶段不重新训练模型，也不继续扩展新类别。目标是把前 13 个阶段沉淀成更容易维护、复现和展示的项目版本。

核心升级点：

```text
1. 抽出共享类别配置，减少 tile / wood / leather 配置漂移。
2. 增加 collect-only 复现检查入口。
3. 增加最终结果 dashboard，方便 GitHub 和面试展示。
4. 修正文档入口，让复现命令不依赖个人电脑绝对路径。
```

## 为什么做工程化收束

前 13 个阶段已经覆盖：

```text
tile overall best
tile crack specialist
wood overall best
leather overall best
leather fold tradeoff
```

继续盲目扩类别或调参会降低故事聚焦度。此时更重要的是让项目具备三个特征：

```text
可解释
可复现
可维护
```

## 代码结构调整

新增轻量公共包：

```text
src/industrial_defect/
  __init__.py
  config.py
  final_results.py
  io.py
```

职责：

```text
config.py: 统一维护类别 defect types、Diffusion prompts、模型默认配置。
final_results.py: 统一维护最终实验清单和时间线。
io.py: 统一维护 CSV / JSON / 图片列表等基础读写工具。
```

现有阶段脚本仍然保留在 `scripts/` 中，保证历史命令可继续运行。

## 新增复现检查

新增：

```text
scripts/14_reproduction_check.py
```

默认模式用于 collect-only 检查，不要求本机已经配置完整训练环境或 MVTec AD 数据集：

```powershell
python scripts/14_reproduction_check.py
```

输出：

```text
outputs/final_report/reproduction_check.md
```

如果要检查完整训练环境和数据集，可使用：

```powershell
python scripts/14_reproduction_check.py --data-root "$env:DATA_ROOT" --strict
```

## 新增最终结果 Dashboard

新增：

```text
scripts/14_generate_final_dashboard.py
```

命令：

```powershell
python scripts/13_collect_final_results.py
python scripts/14_generate_final_dashboard.py
```

输出：

```text
outputs/final_report/final_results_dashboard.md
```

Dashboard 包含：

```text
推荐模型表
阶段指标走势
关键类别修复对比
结果阅读说明
```

## 本阶段不做什么

```text
1. 不重新训练模型。
2. 不扩第四类别。
3. 不继续调 leather fold。
4. 不提交大图、模型权重、预测 mask。
```

## 验收结果

本阶段通过以下检查：

```powershell
python scripts/13_collect_final_results.py
python scripts/14_reproduction_check.py
python scripts/14_generate_final_dashboard.py
python -m py_compile scripts/13_collect_final_results.py scripts/14_reproduction_check.py scripts/14_generate_final_dashboard.py
```

关键输出：

```text
outputs/final_report/final_metrics_summary.csv
outputs/final_report/final_class_metrics.csv
outputs/final_report/final_experiment_timeline.md
outputs/final_report/reproduction_check.md
outputs/final_report/final_results_dashboard.md
```

## 最终结论

第 14 阶段把项目从“实验结果已经完成”推进到“工程入口更清楚、结果更容易复现、展示更聚焦”的状态。

项目现在更适合作为：

```text
GitHub 开源作品
工业视觉方向面试项目
synthetic data 真实下游验证案例
```

