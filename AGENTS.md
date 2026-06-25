# AGENTS.md

本文件是给所有后续参与本项目的代码助手看的项目说明与协作规则。
所有代码助手在修改本仓库前，都应该先阅读并遵守本文档。

---

## 1. 项目定位

本项目是一个用于面试展示的计算机视觉项目，目标岗位是：

```text
视觉算法实习生 / 工业缺陷检测 / AIGC 数据生成 / Python / PyTorch / Diffusion
```

项目中文名：

```text
基于 Diffusion Inpainting 的工业缺陷图像生成与检测增强
```

项目英文名：

```text
Diffusion-Based Synthetic Defect Generation for Industrial Anomaly Detection
```

项目核心目标：

```text
构建一个完整、可解释、能用于面试讲解的工业缺陷生成与检测增强流程，验证 Diffusion 生成的工业缺陷图像是否能提升真实缺陷检测或分割效果。
```

本项目不是普通的 Stable Diffusion 画图 demo，也不是单纯的图像分割练习。
所有工作都必须围绕下面这个工业问题展开：

```text
真实工业缺陷样本稀缺、标注昂贵、缺陷类型长尾。
生成式 AI 只有在能帮助真实缺陷检测模型变强时，才有实际价值。
```

---

## 2. 和用户的协作规则

用户正在从基础较薄弱的状态开始做这个项目，明确要求：

```text
不要一股脑全部做完。
必须一步一步来。
每一步都要详细说明做什么、为什么做、做完之后是什么结果、下一步做什么、为什么。
```

因此，后续任何代码助手都必须遵守：

1. 每个阶段开始前，先说明本阶段要做什么。
2. 说明为什么这个阶段对项目和面试有价值。
3. 每次只执行一个明确阶段，不要跨多个大阶段。
4. 执行后说明具体修改了哪些文件。
5. 执行后说明生成了哪些结果文件。
6. 执行后解释这些结果说明了什么。
7. 执行后说明下一步建议做什么，以及为什么。
8. 进入下一个大阶段前，必须等待用户确认。

以下都属于“大阶段”，不能擅自连续推进：

```text
数据探索
传统伪缺陷生成
Diffusion Inpainting 缺陷生成
异常检测 baseline
U-Net / 分割模型训练
模型评估
README / 项目报告整理
简历和面试话术整理
```

---

## 3. 数据集约定

本项目当前默认使用本地 MVTec AD 数据集。

默认路径：

```text
C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD
```

当前默认起始类别：

```text
tile
```

说明：

```text
该路径属于当前机器上的默认本地路径。
如果后续脚本、配置文件或用户输入显式指定了新的数据路径，应以显式配置为准。
```

代码助手在处理数据前必须先做这些检查：

1. 确认数据集根目录存在。
2. 确认目标类别目录存在。
3. 确认 `train`、`test`、`ground_truth` 结构完整。
4. 确认异常图和 mask 的命名可正确对应。
5. 在开始训练或生成前，先产出一轮可视化与统计结果。

MVTec AD 的 `tile` 类别应符合如下结构：

```text
MVTec_AD/
  tile/
    train/
      good/
    test/
      crack/
      glue_strip/
      good/
      gray_stroke/
      oil/
      rough/
    ground_truth/
      crack/
      glue_strip/
      gray_stroke/
      oil/
      rough/
```

具体统计结果、截图和本轮数据检查结论不应长期写死在 `AGENTS.md` 中。
这些内容应写入：

```text
README.md
docs/stage-*.md
outputs/eda/
```

---

## 4. 仓库约定

本项目仓库根目录：

```text
C:\Users\zsh\Desktop\昂坤视觉\industrial-defect-diffusion
```

`AGENTS.md` 不负责维护“当前完成到哪一步”的动态状态。
它只定义仓库协作规则、目录约定、执行流程和文档要求。

代码助手应默认使用并维护以下结构：

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

仓库中的内容应按职责区分：

```text
README.md：项目概览、环境说明、简要当前状态
AGENTS.md：长期有效的协作规则和项目约束
docs/stage-*.md：每个阶段的详细记录
scripts/：可直接执行的阶段脚本
src/：可复用代码
outputs/：图、表格、模型、实验结果
```

不要把阶段进度、一次性统计结果、临时实验结论长期写进 `AGENTS.md`。

---

## 4.1 Git 仓库要求

本项目必须按 git 仓库方式管理。

要求如下：

1. 每完成一个明确阶段，都要及时进行一次 git 记录。
2. 不要把很多阶段堆在一起再提交。
3. 每个阶段都要保留清晰、可追踪的提交历史。
4. 提交信息要说明本阶段做了什么，而不是只写“update”。
5. 如有必要，可以在阶段结束后先让用户确认，再做 git 提交说明。

建议的提交粒度：

```text
第 1 阶段：数据探索完成
第 2 阶段：传统伪缺陷生成完成
第 3 阶段：Diffusion 生成完成
第 4 阶段：baseline 训练完成
第 5 阶段：实验评估完成
第 6 阶段：README / 报告整理完成
```

建议提交信息格式：

```text
stage1: explore MVTec AD tile dataset and verify masks
stage2: generate traditional synthetic defects
stage3: build diffusion-based defect generation pipeline
stage4: run anomaly detection baseline
stage5: evaluate downstream segmentation results
stage6: finalize project documentation
```

---

## 5. 环境约定

本项目必须使用独立 conda 环境，不得默认复用系统 Python 或用户其他项目环境。

项目专用环境名称：

```text
industrial-defect-diffusion
```

激活命令：

```powershell
conda activate industrial-defect-diffusion
```

环境管理原则：

1. 所有依赖安装默认在该 conda 环境内进行。
2. 安装深度学习库前，优先确认 Python 版本、GPU、CUDA 驱动是否兼容。
3. 不要依赖用户系统 Python 的已有包状态。
4. 不要把“当前临时检测到的库列表”长期写入 `AGENTS.md`。
5. 实际安装结果应写入 `README.md`、阶段文档或独立环境安装记录。

本项目环境策略：

```text
优先使用 Python 3.10
优先使用 GPU 版 PyTorch
优先保持生成模型、异常检测、分割训练在同一项目环境中
```

在安装以下库之前，代码助手必须先说明用途：

```text
PyTorch
torchvision
torchaudio
diffusers
transformers
accelerate
anomalib
ultralytics
opencv-python
```

说明内容至少包括：

```text
这个库解决项目哪一步
为什么现在需要装
是否可能带来兼容性风险
装完后如何验证
```

动态环境信息，例如：

```text
当前 Python 版本
当前已安装库
当前 CUDA 是否可用
当前 torch 是否识别 GPU
```

这些都不应该长期写死在 `AGENTS.md` 中，而应记录在：

```text
README.md
docs/stage-*.md
docs/environment-setup.md
```

Windows 终端与 `conda run` 注意事项：

```text
脚本打印到控制台的内容应尽量使用 ASCII 文本和相对路径。
避免直接打印包含中文字符的绝对路径，防止 Windows GBK 编码或 conda run 输出转码报错。
```

---

## 5.1 阶段文档要求

每完成一个阶段，都必须新增或更新一份 中文Markdown 阶段记录文档。

文档要求：

1. 说明本阶段做了什么。
2. 说明为什么要这么做。
3. 说明执行结果是什么。
4. 说明结果意味着什么。
5. 说明下一步要做什么。
6. 说明下一步为什么这么做。

推荐文件命名：

```text
docs/stage-01-data-exploration.md
docs/stage-02-traditional-synthesis.md
docs/stage-03-diffusion-generation.md
docs/stage-04-baseline-training.md
docs/stage-05-evaluation.md
docs/stage-06-final-report.md
```

如果 `docs/` 目录不存在，应先创建。

阶段文档应尽量保持以下结构：

```text
1. 本阶段目标
2. 本阶段做了什么
3. 为什么这样做
4. 结果与截图
5. 结果说明
6. 遇到的问题
7. 下一步计划
```

每完成一个阶段后，除了更新对应的阶段文档，还必须同步检查并更新 `README.md`。

`README.md` 至少要同步这些内容：

```text
当前项目文档入口
当前可运行脚本
环境说明（如有变化）
项目结构（如有变化）
简要阶段入口或阶段索引
```

注意：

```text
README.md` 不应写成详细阶段日志，但必须反映当前仓库对外可读的最新状态。
```

---

## 6. 项目总流程

项目规划流程如下：

```text
1. 探索 MVTec AD tile 数据
2. 生成传统规则伪缺陷
3. 生成 Diffusion Inpainting 伪缺陷
4. 跑无监督工业异常检测 baseline
5. 训练 U-Net 等分割模型
6. 对比真实数据、传统伪缺陷、Diffusion 原始生成、Diffusion 筛选生成
7. 整理 README、项目报告和面试讲法
```

注意：

```text
不能跳过传统伪缺陷。
```

因为传统伪缺陷是 Diffusion 方法的对照组。
没有传统对照组，就很难证明 Diffusion 生成缺陷到底有没有价值。

---

## 7. 实验原则

本项目不能只凭“生成图片看起来很像”就宣称成功。

核心实验原则：

```text
生成数据是否有价值，必须通过真实测试集上的检测或分割效果来验证。
```

后续主要对比组应该包括：

```text
只用真实数据
真实数据 + 传统伪缺陷
真实数据 + Diffusion 原始生成缺陷
真实数据 + Diffusion 筛选后生成缺陷
```

主要评估指标：

```text
Recall
Precision
F1
Dice
IoU
Image AUROC
Pixel AUROC
误检情况
漏检情况
失败案例
```

项目面试核心表达：

```text
工业 AIGC 不是生成越多越好，而是要可控生成、质量筛选，并且必须在真实工业缺陷检测任务中验证是否有效。
```

---

## 8. 代码风格要求

代码要适合基础较薄弱的用户学习和复盘。

优先使用：

```text
小脚本
清晰函数名
明确路径
简单 CSV / PNG 输出
可视化结果
容易解释的流程
```

避免：

```text
过度封装
一上来写复杂框架
隐藏太多细节
没有解释的高级库调用
一次性生成大量难以检查的结果
```

修改文件时：

```text
每次只围绕当前阶段修改。
不要删除已有输出，除非用户明确要求。
不要重写无关文件。
```

手动编辑文件时应使用 `apply_patch`。

---

## 9. 推荐目录规范

当前项目建议结构：

```text
industrial-defect-diffusion/
  README.md
  AGENTS.md
  requirements.txt
  configs/
  scripts/
    01_explore_dataset.py
    02_generate_traditional_defects.py
    03_generate_diffusion_defects.py
    04_train_unet.py
    05_evaluate_unet.py
  src/
  outputs/
    eda/
    traditional_synthetic/
    diffusion_synthetic/
    baselines/
    training/
    evaluation/
```

传统伪缺陷输出建议：

```text
outputs/traditional_synthetic/tile/
  images/
  masks/
  metadata/
  preview.png
```

Diffusion 生成缺陷输出建议：

```text
outputs/diffusion_synthetic/tile/
  images/
  masks/
  metadata/
  preview.png
```

---

## 10. 阶段状态记录位置

`AGENTS.md` 主要用于记录长期有效的项目规则，不应该堆积容易过期的阶段进度。

因此：

1. 当前阶段进度不要长期写死在 `AGENTS.md` 里。
2. 每个阶段的执行结果应写入对应的阶段文档。
3. `README.md` 可以保留简短的项目当前状态说明。
4. 详细阶段记录应放在 `docs/` 目录下。

推荐做法：

```text
AGENTS.md：写规则、约束、流程、协作方式
README.md：写项目概览、环境、运行方式、简要状态
docs/stage-xx-*.md：写每个阶段做了什么、为什么、结果、下一步
```

后续如果需要查看当前项目进展，应优先查看：

```text
README.md
docs/stage-*.md
```

而不是持续往 `AGENTS.md` 里追加阶段性结果。
