from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENTS = [
    {
        "stage": "stage5",
        "category": "tile",
        "name": "tile stage5 combined",
        "role": "baseline",
        "metrics_path": Path("outputs/training/unet_segmentation/tile/combined/metrics.json"),
        "note": "Small-sample combined traditional + diffusion baseline.",
    },
    {
        "stage": "stage6",
        "category": "tile",
        "name": "tile stage6 expanded combined",
        "role": "diagnostic_baseline",
        "metrics_path": Path("outputs/training/unet_segmentation_expanded/tile/combined/metrics.json"),
        "note": "Expanded generation exposed gray_stroke failure.",
    },
    {
        "stage": "stage6",
        "category": "tile",
        "name": "tile stage6 gray_stroke fixed",
        "role": "overall_best",
        "metrics_path": Path("outputs/training/unet_segmentation_gray_stroke_fix/tile/combined/metrics.json"),
        "note": "Tile overall recommended model after gray_stroke distribution repair.",
    },
    {
        "stage": "stage7",
        "category": "tile",
        "name": "tile stage7 crack fixed",
        "role": "class_specialist",
        "metrics_path": Path("outputs/training/unet_segmentation_stage7/tile/combined/metrics.json"),
        "note": "Tile crack specialist; crack improves with a small overall tradeoff.",
    },
    {
        "stage": "stage8",
        "category": "wood",
        "name": "wood stage8 generalization",
        "role": "generalization_baseline",
        "metrics_path": Path("outputs/training/unet_segmentation_stage8_wood/wood/combined/metrics.json"),
        "note": "First wood run; pipeline transferred but scratch failed.",
    },
    {
        "stage": "stage9",
        "category": "wood",
        "name": "wood stage9 scratch fixed",
        "role": "overall_best",
        "metrics_path": Path("outputs/training/unet_segmentation_stage9_wood_scratch_fix/wood/combined/metrics.json"),
        "note": "Wood overall recommended model after scratch distribution repair.",
    },
    {
        "stage": "stage10",
        "category": "leather",
        "name": "leather stage10 generalization",
        "role": "generalization_baseline",
        "metrics_path": Path("outputs/training/unet_segmentation_stage10_leather/leather/combined/metrics.json"),
        "note": "Third-category pipeline worked but severely over-segmented.",
    },
    {
        "stage": "stage11",
        "category": "leather",
        "name": "leather stage11 precision cut fixed",
        "role": "overall_best",
        "metrics_path": Path("outputs/training/unet_segmentation_stage11_leather_precision_cut_fix/leather/combined/metrics.json"),
        "note": "Leather overall recommended model after good negatives and cut repair.",
    },
    {
        "stage": "stage12",
        "category": "leather",
        "name": "leather stage12 fold fixed",
        "role": "diagnostic_tradeoff",
        "metrics_path": Path("outputs/training/unet_segmentation_stage12_leather_fold_fix/leather/combined/metrics.json"),
        "note": "Fold recall specialist; improves fold but sacrifices overall precision.",
    },
]


TIMELINE = [
    ("stage1", "Data exploration", "Validated MVTec AD tile structure and mask alignment."),
    ("stage2", "Traditional synthesis", "Generated rule-based tile synthetic defects as a baseline."),
    ("stage3", "Diffusion inpainting", "Used traditional masks and Stable Diffusion Inpainting for localized defect synthesis."),
    ("stage4", "PatchCore baseline", "Built a no-synthetic-data anomaly detection baseline."),
    ("stage5", "U-Net synthetic-data validation", "Showed combined traditional + diffusion data helps real test segmentation."),
    ("stage6", "Expanded tile data and gray_stroke repair", "Found gray_stroke failure and repaired its synthetic distribution."),
    ("stage7", "Tile crack repair", "Improved crack generation and validated class-level tradeoff."),
    ("stage8", "Wood generalization", "Moved the pipeline from tile to wood and exposed scratch failure."),
    ("stage9", "Wood scratch repair", "Repaired wood scratch distribution and improved wood overall metrics."),
    ("stage10", "Leather generalization", "Moved to a third category and exposed severe over-segmentation."),
    ("stage11", "Leather precision / cut repair", "Added train/good negative masks and fixed cut generation."),
    ("stage12", "Leather fold tradeoff", "Improved fold recall while documenting the precision tradeoff."),
]


def read_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def round_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    return value


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path.as_posix()}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metric_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        metrics = read_metrics(experiment["metrics_path"])
        summary_rows.append({
            "stage": experiment["stage"],
            "category": experiment["category"],
            "experiment_name": experiment["name"],
            "role": experiment["role"],
            "metrics_path": experiment["metrics_path"].as_posix(),
            "pixel_precision": round_value(metrics.get("pixel_precision", "")),
            "pixel_recall": round_value(metrics.get("pixel_recall", "")),
            "pixel_f1": round_value(metrics.get("pixel_f1", "")),
            "best_pixel_f1": round_value(metrics.get("best_pixel_f1", "")),
            "image_f1": round_value(metrics.get("image_f1", "")),
            "train_samples": metrics.get("train_samples", ""),
            "synthetic_train_samples": metrics.get("synthetic_train_samples", ""),
            "good_negative_samples": metrics.get("good_negative_samples", 0),
            "note": experiment["note"],
        })
        class_metrics = metrics.get("class_metrics", {})
        if not isinstance(class_metrics, dict):
            continue
        for defect_type, values in class_metrics.items():
            if defect_type == "good" or not isinstance(values, dict):
                continue
            class_rows.append({
                "stage": experiment["stage"],
                "category": experiment["category"],
                "experiment_name": experiment["name"],
                "role": experiment["role"],
                "defect_type": defect_type,
                "count": round_value(values.get("count", "")),
                "mean_dice": round_value(values.get("mean_dice", "")),
                "mean_iou": round_value(values.get("mean_iou", "")),
                "mean_recall": round_value(values.get("mean_recall", "")),
            })
    return summary_rows, class_rows


def write_timeline(path: Path) -> None:
    lines = [
        "# Final Experiment Timeline",
        "",
        "This file is generated by `scripts/13_collect_final_results.py`.",
        "",
        "| stage | theme | result |",
        "| --- | --- | --- |",
    ]
    for stage, theme, result in TIMELINE:
        lines.append(f"| {stage} | {theme} | {result} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = Path("outputs") / "final_report"
    summary_rows, class_rows = build_metric_rows()
    write_csv(summary_rows, output_dir / "final_metrics_summary.csv")
    write_csv(class_rows, output_dir / "final_class_metrics.csv")
    write_timeline(output_dir / "final_experiment_timeline.md")

    print("Final report collected.")
    print(f"Output dir: {output_dir.as_posix()}")
    print(f"Metric rows: {len(summary_rows)}")
    print(f"Class rows: {len(class_rows)}")


if __name__ == "__main__":
    main()
