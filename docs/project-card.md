# Project Card

## Task

Industrial surface defect synthesis and segmentation validation on real MVTec AD test sets.

The core question is:

```text
Can synthetic defect data improve real test-set segmentation?
```

The project does not use visual realism alone as the success criterion.

## Dataset

MVTec AD surface categories:

```text
tile
wood
leather
```

Training data uses `train/good` images as normal source images. Evaluation uses real `test` images and ground-truth masks.

## Method

```text
traditional rule-based masks
-> diffusion inpainting
-> quality filtering
-> U-Net segmentation
-> real test-set evaluation
-> class-level distribution repair
```

## Evaluation

Metrics:

```text
Pixel Precision / Recall / F1
Best Pixel F1
Image F1
Class-level Dice / Recall
```

## Recommended Results

| category / goal | stage | pixel_f1 | image_f1 | key class metric |
| --- | --- | ---: | ---: | --- |
| tile overall | Stage 6 | 0.8573 | 0.9492 | gray_stroke Dice = 0.8409 |
| tile crack specialist | Stage 7 | 0.8433 | 0.9711 | crack Dice = 0.7589 |
| wood overall | Stage 9 | 0.3369 | 0.9023 | scratch Dice = 0.3405 |
| leather overall | Stage 11 | 0.4774 | 0.9667 | cut Dice = 0.4064 |
| leather fold tradeoff | Stage 12 | 0.3093 | 0.9735 | fold Dice = 0.4873 |

## Reproducibility

Key commands:

```powershell
python scripts/13_collect_final_results.py
python scripts/14_generate_final_dashboard.py
python scripts/16_generate_final_visuals.py
python scripts/15_project_health_check.py
```

Configuration:

```text
configs/categories.json
configs/final_experiments.json
```

Health check:

```text
outputs/final_report/project_health_check.md
```

## Limitations

```text
1. Results are validated on three surface categories, not the full MVTec AD benchmark.
2. U-Net is intentionally lightweight; the project focuses on synthetic data validation, not model architecture search.
3. Stage 12 improves leather fold recall but is not the leather overall best due to precision tradeoff.
4. Diffusion generation depends on local model cache or external model access when regenerating images.
```

## Non-Committed Artifacts

The repository avoids committing large artifacts:

```text
PNG previews and overlays
model checkpoints
prediction masks
large generated image folders
```

It keeps lightweight CSV / JSON / Markdown summaries needed for collect-only reproduction.

