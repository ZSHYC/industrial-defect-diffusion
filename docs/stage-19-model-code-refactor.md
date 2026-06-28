# 第 19 阶段：模型与推理代码模块化重构

## 本阶段目标

第 19 阶段不重新训练、不扩类别、不修改最终推荐指标，而是修复第 18 阶段后暴露出的工程问题：

```text
训练脚本和推理脚本各自维护了一份 LightUNet 定义。
```

这在短期内能跑通，但长期存在 checkpoint drift 风险：

```text
训练时的模型结构和推理时的模型结构如果不完全一致，
本地 best_model.pt 就可能无法加载，或者加载后行为不一致。
```

因此本阶段把模型和视觉输出工具抽到 `src/industrial_defect/`，让训练、评估和推理共享同一套实现。

## 为什么这一步比继续扩类别更重要

项目到 Stage 18 已经具备：

```text
实验结果
诊断证据
可视化图表
健康检查
轻量推理入口
```

继续扩第四类别的边际收益较低，而训练/推理代码重复会影响项目专业度和可维护性。第 19 阶段优先解决工程一致性问题，让项目从“实验能跑”更接近“长期可维护”。

## 主要改动

新增共享模型模块：

```text
src/industrial_defect/models.py
```

包含：

```text
DoubleConv
LightUNet
build_light_unet()
load_light_unet_checkpoint()
```

新增共享视觉工具：

```text
src/industrial_defect/vision.py
```

包含：

```text
load_rgb_image()
binary_mask_from_probability()
save_probability_mask()
save_binary_mask()
save_overlay()
predict_probability()
```

重构脚本：

| 脚本 | 改动 |
| --- | --- |
| `scripts/05_train_unet_segmentation.py` | 使用共享 `LightUNet`、`predict_probability`、`save_overlay`、`save_binary_mask` |
| `scripts/18_inference_demo.py` | 使用共享 checkpoint 加载和视觉输出工具 |
| `scripts/15_project_health_check.py` | 增加新模块 py_compile 检查 |
| `.github/workflows/ci.yml` | 增加第 18 脚本和共享模块编译检查 |

## 行为保持

本阶段保持以下内容不变：

```text
1. LightUNet 的层命名和 state_dict key。
2. 训练参数和最终 metrics。
3. dry-run 推理入口。
4. 本地 checkpoint 推理输出格式。
```

因此历史 checkpoint 仍应兼容新的共享模型模块。

## 测试

新增测试：

```text
tests/test_models.py
tests/test_vision.py
```

覆盖：

```text
1. LightUNet forward 输出 shape。
2. state_dict 保存与重新加载。
3. probability -> binary mask。
4. overlay 输出尺寸。
```

如果 CI 环境没有安装 torch / torchvision，模型和视觉测试会自动 skip，避免轻量 CI 被重依赖阻塞。

## 验证命令

```powershell
python -m py_compile `
  scripts/05_train_unet_segmentation.py `
  scripts/18_inference_demo.py `
  src/industrial_defect/models.py `
  src/industrial_defect/vision.py
```

```powershell
python -m pytest -q
```

```powershell
python scripts/15_project_health_check.py
```

## 面试表达版本

可以这样讲：

```text
第 18 阶段补了推理入口后，我发现训练脚本和推理脚本里各有一份 U-Net 定义。
这会带来 checkpoint drift 风险，所以第 19 阶段我把模型定义和 mask / overlay 输出工具抽到了 src 包。
现在训练、评估、推理都用同一套 LightUNet 和视觉工具，历史 checkpoint 的 state_dict key 保持不变。
这一步没有包装成新指标，而是提升项目的工程可靠性和长期可维护性。
```
