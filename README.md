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
第 6 阶段：扩大生成数据、质量筛选与监督分割再验证
第 7 阶段：项目报告、README、面试表达整理
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
- [第 2 阶段：传统规则伪缺陷生成](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-02-traditional-synthesis.md)
- [第 3 阶段：Diffusion Inpainting 缺陷生成](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-03-diffusion-generation.md)
- [第 4 阶段：PatchCore 风格无监督异常检测 Baseline](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-04-patchcore-baseline.md)
- [第 5 阶段：U-Net 监督分割训练与生成数据增强对比](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-05-unet-segmentation.md)
- [第 6 阶段：扩大生成数据、质量筛选与监督分割再验证](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-06-expanded-synthesis-and-filtering.md)

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

### 传统规则伪缺陷生成脚本

```powershell
conda run -n industrial-defect-diffusion python scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --samples-per-type 5 --seed 42
```

输出目录：

```text
outputs/traditional_synthetic/tile
```

关键输出：

```text
preview.png
real_vs_traditional_preview.png
summary.csv
```

### Diffusion Inpainting 缺陷生成脚本

```powershell
conda run -n industrial-defect-diffusion python scripts/03_generate_diffusion_defects.py --samples-per-type 3 --num-inference-steps 30 --seed 42 --local-files-only
```

输出目录：

```text
outputs/diffusion_synthetic/tile
```

关键输出：

```text
preview.png
traditional_vs_diffusion_preview.png
summary.csv
```

### PatchCore 风格异常检测 baseline 脚本

```powershell
conda run -n industrial-defect-diffusion python scripts/04_patchcore_baseline.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --image-size 224 --seed 42
```

输出目录：

```text
outputs/baselines/patchcore/tile
```

关键输出：

```text
preview.png
image_scores.csv
metrics.json
summary.md
```

### U-Net 监督分割训练脚本

```powershell
conda run -n industrial-defect-diffusion python scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --image-size 256 --epochs 30 --batch-size 4 --seed 42
```

输出目录：

```text
outputs/training/unet_segmentation/tile
```

默认运行三组实验：

```text
traditional：只用传统规则伪缺陷训练
diffusion：只用 Diffusion Inpainting 伪缺陷训练
combined：传统伪缺陷 + Diffusion 伪缺陷一起训练
```

关键输出：

```text
comparison_summary.csv
comparison_preview.png
traditional/metrics.json
diffusion/metrics.json
combined/metrics.json
```

### 第 6 阶段扩大生成数据与质量筛选

扩大传统规则伪缺陷：

```powershell
conda run -n industrial-defect-diffusion python scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --samples-per-type 20 --seed 43 --output-dir outputs/expanded_synthetic/traditional
```

扩大 Diffusion Inpainting 伪缺陷：

```powershell
conda run -n industrial-defect-diffusion python scripts/03_generate_diffusion_defects.py --category tile --traditional-summary outputs/expanded_synthetic/traditional/tile/summary.csv --samples-per-type 10 --num-inference-steps 30 --seed 43 --local-files-only --output-dir outputs/expanded_synthetic/diffusion
```

质量筛选：

```powershell
conda run -n industrial-defect-diffusion python scripts/06_filter_synthetic_quality.py --traditional-summary outputs/expanded_synthetic/traditional/tile/summary.csv --diffusion-summary outputs/expanded_synthetic/diffusion/tile/summary.csv --output-dir outputs/quality_filter/tile
```

扩大数据重新训练：

```powershell
conda run -n industrial-defect-diffusion python scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --image-size 256 --epochs 30 --batch-size 4 --seed 43 --traditional-summary outputs/expanded_synthetic/traditional/tile/summary.csv --diffusion-summary outputs/expanded_synthetic/diffusion/tile/summary.csv --output-dir outputs/training/unet_segmentation_expanded
```

第 6 阶段输出目录：

```text
outputs/expanded_synthetic/traditional/tile
outputs/expanded_synthetic/diffusion/tile
outputs/quality_filter/tile
outputs/training/unet_segmentation_expanded/tile
outputs/training/unet_segmentation_filtered/tile
```

第 6 阶段关键对比结果：

```text
expanded traditional: Pixel F1 = 0.5657
expanded diffusion: Pixel F1 = 0.6847
expanded combined: Pixel F1 = 0.7667
filtered combined: Pixel F1 = 0.7667
```

第 6 阶段发现 `gray_stroke` 生成质量是主要瓶颈后，已进行专项修复：

```powershell
conda run -n industrial-defect-diffusion python scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --samples-per-type 20 --seed 104 --defect-types gray_stroke --output-dir outputs/gray_stroke_fix/traditional
```

修复后重新训练结果：

```text
gray_stroke fixed combined: Pixel F1 = 0.8573
gray_stroke fixed combined: Image F1 = 0.9492
gray_stroke class Dice = 0.8409
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
