# 第 16 阶段：最终展示资产与论文式结果可视化

## 本阶段目标

第 16 阶段不重新训练模型，不修改实验结论。目标是把前 15 个阶段形成的最终结果转成更适合 GitHub、答辩和面试展示的资产。

新增内容：

```text
1. 最终结果图。
2. 类别修复前后对比图。
3. precision / recall tradeoff 图。
4. 论文式 Markdown 表格。
5. 项目卡片。
6. README 首屏方法流程和结果图。
```

## 新增脚本

```text
scripts/16_generate_final_visuals.py
```

运行：

```powershell
python scripts/16_generate_final_visuals.py
```

输入：

```text
outputs/final_report/final_metrics_summary.csv
outputs/final_report/final_class_metrics.csv
```

输出：

```text
outputs/final_report/figures/final_recommended_models.png
outputs/final_report/figures/category_progression.png
outputs/final_report/figures/class_repair_before_after.png
outputs/final_report/figures/precision_recall_tradeoff.png
outputs/final_report/final_paper_tables.md
```

## 展示图说明

```text
final_recommended_models.png:
  展示最终推荐模型的 Pixel F1 和 Image F1。

category_progression.png:
  展示 tile / wood / leather 在关键阶段的 Pixel F1 变化。

class_repair_before_after.png:
  展示 gray_stroke、crack、scratch、cut、fold 修复前后的 Dice 提升。

precision_recall_tradeoff.png:
  展示各阶段 pixel precision / recall 权衡，尤其解释 Stage 12。
```

## README 升级

README 首屏新增：

```text
英文项目定位
最终推荐结果图
Mermaid 方法流程图
论文式结果表入口
项目卡片入口
```

## 项目卡片

新增：

```text
docs/project-card.md
```

用于快速说明：

```text
Task
Dataset
Method
Evaluation
Recommended Results
Reproducibility
Limitations
Non-Committed Artifacts
```

## 验收

```powershell
python scripts/16_generate_final_visuals.py
python -m py_compile scripts/16_generate_final_visuals.py
python scripts/15_project_health_check.py
```

第 16 阶段完成后，项目从“结果可复现”进一步升级为“结果容易展示、容易讲清楚、容易被快速理解”。

