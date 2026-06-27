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
第 7 阶段：crack 专项改进与最终实验整理
第 8 阶段：wood 类别泛化验证与复现实验包
第 9 阶段：wood scratch 专项修复与跨类别误差分析
第 10 阶段：leather 第三类别泛化验证
第 11 阶段：leather precision / cut 专项修复
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
- [第 7 阶段：crack 专项改进与最终实验整理](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-07-crack-improvement-and-final-analysis.md)
- [第 8 阶段：wood 类别泛化验证与复现实验包](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-08-wood-generalization.md)
- [第 9 阶段：wood scratch 专项修复与跨类别误差分析](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-09-wood-scratch-fix.md)
- [第 10 阶段：leather 第三类别泛化验证](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-10-leather-generalization.md)
- [第 11 阶段：leather precision / cut 专项修复](C:/Users/zsh/Desktop/昂坤视觉/industrial-defect-diffusion/docs/stage-11-leather-precision-cut-fix.md)

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

### 第 7 阶段 crack 专项改进与最终实验整理

第 7 阶段沿用第 6 阶段的误差分析路线，针对 `crack` 类继续做专项改进：

```powershell
conda run -n industrial-defect-diffusion python scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --samples-per-type 20 --seed 204 --output-dir outputs/stage7_synthetic/traditional
```

```powershell
conda run -n industrial-defect-diffusion python scripts/03_generate_diffusion_defects.py --category tile --traditional-summary outputs/stage7_synthetic/traditional/tile/summary.csv --samples-per-type 10 --num-inference-steps 30 --seed 204 --local-files-only --output-dir outputs/stage7_synthetic/diffusion
```

```powershell
conda run -n industrial-defect-diffusion python scripts/06_filter_synthetic_quality.py --traditional-summary outputs/stage7_synthetic/traditional/tile/summary.csv --diffusion-summary outputs/stage7_synthetic/diffusion/tile/summary.csv --output-dir outputs/stage7_quality_filter/tile
```

```powershell
conda run -n industrial-defect-diffusion python scripts/07_analyze_crack_distribution.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile
```

```powershell
conda run -n industrial-defect-diffusion python scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category tile --image-size 256 --epochs 30 --batch-size 4 --seed 104 --traditional-summary outputs/stage7_quality_filter/tile/accepted_traditional_summary.csv --diffusion-summary outputs/stage7_quality_filter/tile/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_stage7 --experiments combined
```

第 7 阶段关键结果：

```text
stage7 combined: Pixel F1 = 0.8433
stage7 combined: Best Pixel F1 = 0.8668
stage7 combined: Image F1 = 0.9711
stage7 crack Dice = 0.7589
stage7 crack Recall = 0.6841
```

第 7 阶段结论：

```text
crack 生成分布修复后，crack Dice 从 0.6120 提升到 0.7589，
crack Recall 从 0.4732 提升到 0.6841。
整体默认 Pixel F1 从 0.8573 小幅下降到 0.8433，
说明类别级改进会带来 overall pixel precision / recall 权衡。
```

最终推荐指标摘要：

```text
stage5 small combined:
  Pixel F1 = 0.8064
  Image F1 = 0.8615

stage6 expanded combined:
  Pixel F1 = 0.7667
  Image F1 = 0.8715
  gray_stroke Dice = 0.0022

stage6 gray_stroke fixed combined:
  Pixel F1 = 0.8573
  Image F1 = 0.9492
  gray_stroke Dice = 0.8409

stage7 crack improved combined:
  Pixel F1 = 0.8433
  Best Pixel F1 = 0.8668
  Image F1 = 0.9711
  crack Dice = 0.7589
```

最终项目表达：

```text
本项目不是证明 Diffusion 图片“看起来像”，而是通过真实测试集验证生成数据是否提升分割效果。
第 6 阶段发现 gray_stroke 失败后，通过类别级误差分析修复生成分布，Pixel F1 从 0.7667 提升到 0.8573。
第 7 阶段继续对 crack 做专项优化，验证生成质量、类别分布和下游指标之间的关系。
```

### 第 8 阶段 wood 类别泛化验证

第 8 阶段将流程从 `tile` 迁移到 `wood`，验证项目不是单类别 hard-code demo。

主要改动：

```text
1. traditional 生成脚本支持 tile / wood 类别配置。
2. Diffusion prompt 支持 tile / wood 类别配置。
3. U-Net 训练评估脚本按 category 自动读取 defect types。
4. 新增 wood 一键复现实验入口。
```

wood 一键复现实验：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/08_run_wood_generalization.py --local-files-only
```

分步命令：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/01_explore_dataset.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category wood --output-dir outputs/eda
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category wood --samples-per-type 20 --seed 304 --output-dir outputs/stage8_wood_synthetic/traditional
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/03_generate_diffusion_defects.py --category wood --traditional-summary outputs/stage8_wood_synthetic/traditional/wood/summary.csv --samples-per-type 10 --num-inference-steps 30 --seed 304 --local-files-only --output-dir outputs/stage8_wood_synthetic/diffusion
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/06_filter_synthetic_quality.py --traditional-summary outputs/stage8_wood_synthetic/traditional/wood/summary.csv --diffusion-summary outputs/stage8_wood_synthetic/diffusion/wood/summary.csv --output-dir outputs/stage8_wood_quality_filter/wood
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category wood --image-size 256 --epochs 30 --batch-size 4 --seed 304 --traditional-summary outputs/stage8_wood_quality_filter/wood/accepted_traditional_summary.csv --diffusion-summary outputs/stage8_wood_quality_filter/wood/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_stage8_wood --experiments combined
```

第 8 阶段关键结果：

```text
wood traditional: 100
wood diffusion: 50
quality accepted: 144
quality rejected: 6

wood combined:
  Pixel F1 = 0.2651
  Best Pixel F1 = 0.2901
  Image F1 = 0.8947
  liquid Dice = 0.5674
  scratch Dice = 0.0247
```

第 8 阶段结论：

```text
流程已经从 tile 迁移到 wood，说明项目不是单类别 hard-code demo。
但 wood 像素级分割明显弱于 tile，尤其 scratch 类几乎失败。
这说明跨类别迁移不仅要迁移代码流程，还要重新做每个类别的真实缺陷分布分析。
```

### 第 9 阶段 wood scratch 专项修复

第 9 阶段针对第 8 阶段最弱的 `wood/scratch` 做专项修复。

核心判断：

```text
真实 wood scratch mean area_ratio ~= 0.0744
第 8 阶段 synthetic scratch mean area_ratio ~= 0.0020
旧生成规则过细、过小、过局部，导致 scratch recall 几乎失败。
```

生成第 9 阶段 scratch-only traditional：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category wood --defect-types scratch --samples-per-type 20 --seed 404 --output-dir outputs/stage9_wood_scratch_fix/traditional
```

生成第 9 阶段 scratch-only diffusion：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/03_generate_diffusion_defects.py --category wood --defect-types scratch --traditional-summary outputs/stage9_wood_scratch_fix/traditional/wood/summary.csv --samples-per-type 10 --num-inference-steps 30 --seed 404 --local-files-only --output-dir outputs/stage9_wood_scratch_fix/diffusion
```

质量筛选、分布分析、合并训练数据：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/06_filter_synthetic_quality.py --traditional-summary outputs/stage9_wood_scratch_fix/traditional/wood/summary.csv --diffusion-summary outputs/stage9_wood_scratch_fix/diffusion/wood/summary.csv --output-dir outputs/stage9_wood_scratch_fix/quality_filter/scratch
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/09_analyze_wood_scratch_distribution.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --old-traditional-summary outputs/stage8_wood_synthetic/traditional/wood/summary.csv --old-diffusion-summary outputs/stage8_wood_synthetic/diffusion/wood/summary.csv --new-traditional-summary outputs/stage9_wood_scratch_fix/traditional/wood/summary.csv --new-diffusion-summary outputs/stage9_wood_scratch_fix/diffusion/wood/summary.csv --output-dir outputs/stage9_wood_scratch_fix/analysis
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/09_prepare_wood_scratch_fix_dataset.py --stage8-quality-dir outputs/stage8_wood_quality_filter/wood --stage9-scratch-quality-dir outputs/stage9_wood_scratch_fix/quality_filter/scratch --output-dir outputs/stage9_wood_scratch_fix/quality_filter/wood
```

第 9 阶段 U-Net combined 训练：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category wood --image-size 256 --epochs 30 --batch-size 4 --seed 404 --traditional-summary outputs/stage9_wood_scratch_fix/quality_filter/wood/accepted_traditional_summary.csv --diffusion-summary outputs/stage9_wood_scratch_fix/quality_filter/wood/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_stage9_wood_scratch_fix --experiments combined
```

wood 第 8 / 第 9 阶段对比：

```text
stage8 wood combined:
  Pixel F1 = 0.2651
  Best Pixel F1 = 0.2901
  Image F1 = 0.8947
  scratch Dice = 0.0247
  scratch Recall = 0.0146

stage9 wood scratch fixed combined:
  Pixel F1 = 0.3369
  Best Pixel F1 = 0.3815
  Image F1 = 0.9023
  scratch Dice = 0.3405
  scratch Recall = 0.4169
```

第 9 阶段结论：

```text
第 8 阶段证明流程可以迁移到 wood；第 9 阶段证明迁移后仍需要类别级误差分析和生成分布修复。
```

### 第 10 阶段 leather 第三类别泛化验证

第 10 阶段将流程继续迁移到第三个表面类别 `leather`。

新增类别配置：

```text
leather:
  color
  cut
  fold
  glue
  poke
```

一键复现实验：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/10_run_leather_generalization.py --local-files-only
```

分步命令：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/01_explore_dataset.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category leather --output-dir outputs/eda
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/02_generate_traditional_defects.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category leather --samples-per-type 20 --seed 504 --output-dir outputs/stage10_leather_synthetic/traditional
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/03_generate_diffusion_defects.py --category leather --traditional-summary outputs/stage10_leather_synthetic/traditional/leather/summary.csv --samples-per-type 10 --num-inference-steps 30 --seed 504 --local-files-only --output-dir outputs/stage10_leather_synthetic/diffusion
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/06_filter_synthetic_quality.py --traditional-summary outputs/stage10_leather_synthetic/traditional/leather/summary.csv --diffusion-summary outputs/stage10_leather_synthetic/diffusion/leather/summary.csv --output-dir outputs/stage10_leather_quality_filter/leather
```

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category leather --image-size 256 --epochs 30 --batch-size 4 --seed 504 --traditional-summary outputs/stage10_leather_quality_filter/leather/accepted_traditional_summary.csv --diffusion-summary outputs/stage10_leather_quality_filter/leather/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_stage10_leather --experiments combined
```

第 10 阶段关键结果：

```text
leather traditional: 100
leather diffusion: 50
quality accepted: 135
quality rejected: 15

leather combined:
  Pixel Precision = 0.0305
  Pixel Recall = 0.5488
  Pixel F1 = 0.0579
  Best Pixel F1 = 0.1444
  Image F1 = 0.8519
  color Dice = 0.2760
  cut Dice = 0.0215
  fold Dice = 0.2453
  glue Dice = 0.1703
  poke Dice = 0.2118
```

第 10 阶段结论：

```text
流程已经迁移到 tile、wood、leather 三个类别。
leather 证明第三类别闭环可以跑通，但也暴露出小面积缺陷下 pixel precision 很低的问题。
如果继续推进，下一步应优先做 leather precision / cut 专项修复，而不是继续盲目扩类别。
```

### 第 11 阶段 leather precision / cut 专项修复

第 11 阶段修复第 10 阶段 `leather` 的过分割问题：

```text
stage10 good 图 image_score 均值约 0.994
训练数据只有 synthetic defect 正样本
缺少真实 train/good 空 mask 负样本约束
```

主要改动：

```text
1. U-Net 训练支持 --good-negative-samples。
2. 训练时加入 100 张 leather train/good 空 mask 负样本。
3. 修复 leather/cut 生成分布。
4. 输出 threshold_sweep.csv 和 postprocess_sweep.csv。
```

第 11 阶段关键训练命令：

```powershell
D:\miniforge3\envs\industrial-defect-diffusion\python.exe scripts/05_train_unet_segmentation.py --data-root "C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD" --category leather --image-size 256 --epochs 30 --batch-size 4 --seed 604 --traditional-summary outputs/stage11_leather_precision_cut_fix/quality_filter/leather/accepted_traditional_summary.csv --diffusion-summary outputs/stage11_leather_precision_cut_fix/quality_filter/leather/accepted_diffusion_summary.csv --output-dir outputs/training/unet_segmentation_stage11_leather_precision_cut_fix --experiments combined --good-negative-samples 100
```

第 10 / 第 11 阶段对比：

```text
stage10 leather:
  Pixel Precision = 0.0305
  Pixel Recall = 0.5488
  Pixel F1 = 0.0579
  Best Pixel F1 = 0.1444
  Image F1 = 0.8519
  cut Dice = 0.0215

stage11 leather precision / cut fixed:
  Pixel Precision = 0.8752
  Pixel Recall = 0.3282
  Pixel F1 = 0.4774
  Best Pixel F1 = 0.5219
  Image F1 = 0.9667
  cut Dice = 0.4064
```

第 11 阶段结论：

```text
加入真实 good negative 后，leather 的过分割问题明显缓解。
这说明生成增强不仅要补缺陷样本，也要用正常样本约束模型边界。
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
