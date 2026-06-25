# 基于 Diffusion Inpainting 的工业缺陷图像生成与检测增强

## 1. 项目简介

本项目面向工业视觉中的缺陷样本稀缺问题，目标是构建一个完整、可解释、适合面试展示的实验流程，用来验证：

```text
Diffusion 生成的工业缺陷图像，是否能够提升真实缺陷检测或分割效果。
```

项目不是单纯的图像生成 demo，而是一个围绕工业问题展开的完整闭环：

```text
数据探索
→ 传统伪缺陷生成
→ Diffusion 缺陷生成
→ 异常检测 / 分割训练
→ 真实测试集评估
→ 项目报告与面试表达
```

---

## 2. 项目目标

本项目要解决的问题是：

```text
真实工业缺陷样本少、标注成本高、缺陷分布长尾。
```

因此，项目核心问题不是“能不能生成图片”，而是：

```text
能不能生成对下游工业缺陷检测真正有帮助的缺陷图像。
```

最终需要通过真实测试集上的指标来验证，而不是只看生成图片是否“好看”。

---

## 3. 项目结构

当前项目建议结构如下：

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
README.md：项目概览、环境说明、文档入口
AGENTS.md：长期有效的协作规则和项目约束
docs/：阶段记录、环境安装记录
scripts/：每个阶段可直接执行的脚本
src/：可复用代码
outputs/：图、表格、模型、实验结果
```

---

## 4. 数据集

当前默认使用本地 MVTec AD 数据集。

默认路径：

```text
C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD
```

当前起始类别：

```text
tile
```

说明：

```text
如果后续脚本、配置文件或命令行参数显式指定了新的数据路径，应以显式配置为准。
```

---

## 5. 环境

本项目必须使用独立 conda 环境：

```text
industrial-defect-diffusion
```

激活命令：

```powershell
conda activate industrial-defect-diffusion
```

本项目环境策略：

```text
优先使用 Python 3.10
优先使用 GPU 版 PyTorch
优先把生成模型、异常检测、分割训练放在同一项目环境中
```

详细环境安装记录见：

- [环境安装记录](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/environment-setup.md)

---

## 6. 项目流程

本项目按阶段推进：

```text
第 1 阶段：数据探索与校验
第 2 阶段：传统规则伪缺陷生成
第 3 阶段：Diffusion Inpainting 缺陷生成
第 4 阶段：异常检测 baseline
第 5 阶段：分割训练与实验评估
第 6 阶段：项目报告、README、面试表达整理
```

重要原则：

```text
不要跳过传统伪缺陷阶段。
```

因为传统伪缺陷是 Diffusion 方法的重要对照组，没有它就很难说明 Diffusion 生成是否真的有价值。

---

## 7. 文档入口

阶段性结果不长期堆积在 `README.md` 中，而是写入 `docs/`。

当前文档入口：

- [环境安装记录](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/environment-setup.md)
- [第 1 阶段：MVTec AD 数据探索与校验](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-01-data-exploration.md)

后续阶段文档将继续放在：

```text
docs/stage-*.md
```

---

## 8. 当前可运行脚本

当前已完成并可运行的脚本：

### 数据探索脚本

```powershell
python scripts/01_explore_dataset.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile
```

输出目录：

```text
outputs/eda/tile
```

---

## 9. Git 约定

本项目按阶段进行 git 记录。

要求：

1. 每完成一个明确阶段，都要及时进行一次提交。
2. 每个阶段都要同步补充对应的 Markdown 阶段文档。
3. 提交信息必须能反映本阶段做了什么。

---

## 10. 当前维护原则

为便于长期维护：

1. `README.md` 只保留长期有效的项目概览和文档入口。
2. 动态阶段结果写入 `docs/stage-*.md`。
3. 协作规则和约束写入 `AGENTS.md`。
4. 临时实验数据、图像、统计表写入 `outputs/`。

