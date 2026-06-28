# 第 18 阶段：轻量推理演示与部署入口

## 本阶段目标

第 18 阶段不继续扩类别，也不重新训练模型，而是补齐项目的最后一公里：

```text
输入一张工业表面图像
-> 加载本地训练好的 U-Net checkpoint
-> 输出 defect probability mask / binary mask / overlay
-> 明确说明部署边界和权重策略
```

本阶段的重点是让项目从“实验结果完整”进一步变成“使用入口清楚”。

## 为什么需要这一阶段

Stage 1-17 已经证明：

```text
synthetic defect 数据可以通过真实测试集分割指标验证其有效性。
```

但一个 GitHub 访问者或面试官还会继续问：

```text
如果我给你一张新图，这个项目怎么跑一次推理？
```

因此第 18 阶段新增轻量 inference demo，展示模型在本地 checkpoint 可用时如何使用，同时保持仓库不提交大模型权重。

## 新增内容

新增脚本：

```text
scripts/18_inference_demo.py
```

支持两种模式：

| 模式 | 是否需要 checkpoint | 作用 |
| --- | --- | --- |
| dry-run | 否 | 验证推理接口、生成 demo requirements 和 deployment readiness |
| local inference | 是 | 对单张图输出 probability mask、binary mask 和 overlay |

## Dry-Run 模式

dry-run 不依赖 torch、不依赖 checkpoint、不依赖数据集，适合公开仓库和 CI 轻量检查：

```powershell
python scripts/18_inference_demo.py --category leather --dry-run
```

输出：

```text
outputs/demo_inference/demo_requirements.md
outputs/final_report/deployment_readiness.md
```

## 本地推理模式

如果本地已经有训练得到的 `best_model.pt`，可以运行：

```powershell
python scripts/18_inference_demo.py `
  --category leather `
  --image path/to/inspection_image.png `
  --checkpoint path/to/best_model.pt `
  --output-dir outputs/demo_inference
```

输出：

```text
outputs/demo_inference/probability_mask.png
outputs/demo_inference/binary_mask.png
outputs/demo_inference/overlay.png
outputs/demo_inference/metadata.json
```

其中：

| 文件 | 含义 |
| --- | --- |
| `probability_mask.png` | 像素级缺陷概率图 |
| `binary_mask.png` | 按阈值二值化后的缺陷 mask |
| `overlay.png` | 红色缺陷区域叠加到原图上的可视化 |
| `metadata.json` | category、threshold、image_score、defect_area_ratio 和输出路径 |

## 权重策略

本仓库不提交：

```text
*.pt
*.pth
*.ckpt
prediction images
large generated samples
```

原因：

```text
1. 权重和预测图是运行产物，体积大，不适合放入 Git。
2. 项目重点是可复现实验流程和真实测试集验证逻辑。
3. 用户可以按 README 的训练命令在本地重新生成 checkpoint。
```

## 与前面阶段的关系

第 18 阶段不改变任何最终指标。

最终推荐仍然是：

| 类别 / 目标 | 推荐阶段 |
| --- | --- |
| tile overall | Stage 6 gray_stroke fixed |
| tile crack specialist | Stage 7 crack fixed |
| wood overall | Stage 9 scratch fixed |
| leather overall | Stage 11 precision / cut fixed |
| leather fold tradeoff | Stage 12 fold fixed |

第 18 阶段只补充：

```text
实验结果 -> 本地推理入口 -> 部署边界说明
```

## 健康检查

Stage 18 已纳入项目 health check：

```powershell
python scripts/15_project_health_check.py
```

新增检查项包括：

```text
1. scripts/18_inference_demo.py 可以 py_compile。
2. dry-run inference demo 可以运行。
3. deployment_readiness.md 可以生成。
4. README 包含 inference / deployment 入口。
```

## 面试表达版本

可以这样讲：

```text
前 17 个阶段主要证明 synthetic defect 数据在真实 MVTec AD 测试集上的有效性。
第 18 阶段我没有继续堆实验，而是补了一个轻量推理入口。
仓库不提交模型权重，但提供 dry-run 和本地 checkpoint 两种模式：
dry-run 用于公开复现和 CI 检查，本地模式用于单图输出 mask 和 overlay。
这样项目不仅有实验结论，也说明了实际使用时模型输入、输出和部署边界。
```
