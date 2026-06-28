from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FINAL_DIR = Path("outputs") / "final_report"
FIGURE_DIR = FINAL_DIR / "figures"

THRESHOLD_EXPERIMENTS = [
    {
        "name": "leather stage11 precision cut fixed",
        "role": "overall_best",
        "threshold_path": Path("outputs/training/unet_segmentation_stage11_leather_precision_cut_fix/leather/combined/threshold_sweep.csv"),
        "postprocess_path": Path("outputs/training/unet_segmentation_stage11_leather_precision_cut_fix/leather/combined/postprocess_sweep.csv"),
    },
    {
        "name": "leather stage12 fold fixed",
        "role": "diagnostic_tradeoff",
        "threshold_path": Path("outputs/training/unet_segmentation_stage12_leather_fold_fix/leather/combined/threshold_sweep.csv"),
        "postprocess_path": Path("outputs/training/unet_segmentation_stage12_leather_fold_fix/leather/combined/postprocess_sweep.csv"),
    },
]

DISTRIBUTION_REPAIRS = [
    {
        "defect": "tile/gray_stroke",
        "path": Path("outputs/gray_stroke_fix/gray_stroke_distribution_summary.csv"),
        "old_source": "old_traditional",
        "new_source": "fixed_traditional",
        "before_experiment": "tile stage6 expanded combined",
        "after_experiment": "tile stage6 gray_stroke fixed",
        "class_name": "gray_stroke",
    },
    {
        "defect": "tile/crack",
        "path": Path("outputs/crack_fix/crack_distribution_summary.csv"),
        "old_source": "old_traditional",
        "new_source": "new_traditional",
        "before_experiment": "tile stage6 gray_stroke fixed",
        "after_experiment": "tile stage7 crack fixed",
        "class_name": "crack",
    },
    {
        "defect": "wood/scratch",
        "path": Path("outputs/stage9_wood_scratch_fix/analysis/wood_scratch_distribution_summary.csv"),
        "old_source": "old_traditional",
        "new_source": "new_traditional",
        "before_experiment": "wood stage8 generalization",
        "after_experiment": "wood stage9 scratch fixed",
        "class_name": "scratch",
    },
    {
        "defect": "leather/cut",
        "path": Path("outputs/stage11_leather_precision_cut_fix/analysis/leather_cut_distribution_summary.csv"),
        "old_source": "old_traditional",
        "new_source": "new_traditional",
        "before_experiment": "leather stage10 generalization",
        "after_experiment": "leather stage11 precision cut fixed",
        "class_name": "cut",
    },
    {
        "defect": "leather/fold",
        "path": Path("outputs/stage12_leather_fold_fix/analysis/leather_fold_distribution_summary.csv"),
        "old_source": "old_traditional",
        "new_source": "new_traditional",
        "before_experiment": "leather stage11 precision cut fixed",
        "after_experiment": "leather stage12 fold fixed",
        "class_name": "fold",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def fmt(value: float) -> str:
    return f"{value:.4f}"


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing row where {key}={value}")


def collect_threshold_postprocess() -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for experiment in THRESHOLD_EXPERIMENTS:
        threshold_rows = read_csv(experiment["threshold_path"])
        postprocess_rows = read_csv(experiment["postprocess_path"])
        default_row = min(threshold_rows, key=lambda row: abs(as_float(row, "threshold") - 0.5))
        best_threshold_row = max(threshold_rows, key=lambda row: as_float(row, "pixel_f1"))
        best_postprocess_row = max(postprocess_rows, key=lambda row: as_float(row, "pixel_f1"))
        output_rows.append({
            "experiment_name": experiment["name"],
            "role": experiment["role"],
            "default_threshold": as_float(default_row, "threshold"),
            "default_pixel_precision": as_float(default_row, "pixel_precision"),
            "default_pixel_recall": as_float(default_row, "pixel_recall"),
            "default_pixel_f1": as_float(default_row, "pixel_f1"),
            "best_threshold": as_float(best_threshold_row, "threshold"),
            "best_threshold_precision": as_float(best_threshold_row, "pixel_precision"),
            "best_threshold_recall": as_float(best_threshold_row, "pixel_recall"),
            "best_threshold_f1": as_float(best_threshold_row, "pixel_f1"),
            "best_postprocess_threshold": as_float(best_postprocess_row, "threshold"),
            "best_postprocess_min_area_ratio": as_float(best_postprocess_row, "min_component_area_ratio"),
            "best_postprocess_precision": as_float(best_postprocess_row, "pixel_precision"),
            "best_postprocess_recall": as_float(best_postprocess_row, "pixel_recall"),
            "best_postprocess_f1": as_float(best_postprocess_row, "pixel_f1"),
            "threshold_f1_gain": as_float(best_threshold_row, "pixel_f1") - as_float(default_row, "pixel_f1"),
            "postprocess_f1_gain": as_float(best_postprocess_row, "pixel_f1") - as_float(default_row, "pixel_f1"),
        })
    return output_rows


def class_dice(class_rows: list[dict[str, str]], experiment_name: str, defect_type: str) -> float:
    for row in class_rows:
        if row["experiment_name"] == experiment_name and row["defect_type"] == defect_type:
            return as_float(row, "mean_dice")
    raise KeyError(f"Missing class metric: {experiment_name} / {defect_type}")


def collect_distribution_repairs() -> list[dict[str, Any]]:
    class_rows = read_csv(FINAL_DIR / "final_class_metrics.csv")
    output_rows: list[dict[str, Any]] = []
    for repair in DISTRIBUTION_REPAIRS:
        rows = read_csv(repair["path"])
        real = find_row(rows, "source", "real")
        old = find_row(rows, "source", repair["old_source"])
        new = find_row(rows, "source", repair["new_source"])
        before_dice = class_dice(class_rows, repair["before_experiment"], repair["class_name"])
        after_dice = class_dice(class_rows, repair["after_experiment"], repair["class_name"])
        output_rows.append({
            "defect": repair["defect"],
            "real_mean_area_ratio": as_float(real, "mean_area_ratio"),
            "old_synthetic_mean_area_ratio": as_float(old, "mean_area_ratio"),
            "new_synthetic_mean_area_ratio": as_float(new, "mean_area_ratio"),
            "old_area_abs_error": abs(as_float(old, "mean_area_ratio") - as_float(real, "mean_area_ratio")),
            "new_area_abs_error": abs(as_float(new, "mean_area_ratio") - as_float(real, "mean_area_ratio")),
            "dice_before": before_dice,
            "dice_after": after_dice,
            "dice_gain": after_dice - before_dice,
        })
    return output_rows


def collect_baseline_comparison() -> list[dict[str, Any]]:
    patchcore = read_json(Path("outputs/baselines/patchcore/tile/metrics.json"))
    final_rows = read_csv(FINAL_DIR / "final_metrics_summary.csv")
    tile_best = find_row(final_rows, "experiment_name", "tile stage6 gray_stroke fixed")
    return [
        {
            "category": "tile",
            "method": "PatchCore-style ResNet18",
            "role": "unsupervised anomaly baseline",
            "pixel_precision": patchcore["pixel_precision"],
            "pixel_recall": patchcore["pixel_recall"],
            "pixel_f1": patchcore["pixel_f1"],
            "image_f1": patchcore["image_f1"],
            "note": "Strong image-level anomaly baseline; weaker pixel F1 than synthetic-supervised U-Net.",
        },
        {
            "category": "tile",
            "method": "Synthetic-data U-Net",
            "role": "supervised segmentation with generated masks",
            "pixel_precision": as_float(tile_best, "pixel_precision"),
            "pixel_recall": as_float(tile_best, "pixel_recall"),
            "pixel_f1": as_float(tile_best, "pixel_f1"),
            "image_f1": as_float(tile_best, "image_f1"),
            "note": "Higher pixel F1 after category-level synthetic distribution repair.",
        },
    ]


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def plot_threshold_sweep() -> None:
    plt.figure(figsize=(8.5, 5))
    for experiment in THRESHOLD_EXPERIMENTS:
        rows = read_csv(experiment["threshold_path"])
        thresholds = [as_float(row, "threshold") for row in rows]
        f1_values = [as_float(row, "pixel_f1") for row in rows]
        label = experiment["name"].replace("leather ", "")
        plt.plot(thresholds, f1_values, marker="o", markersize=3, linewidth=2, label=label)
    plt.xlabel("Pixel threshold")
    plt.ylabel("Pixel F1")
    plt.title("Leather Threshold Sweep")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "threshold_sweep_leather.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_distribution_repair(rows: list[dict[str, Any]]) -> None:
    labels = [row["defect"] for row in rows]
    old_error = [row["old_area_abs_error"] for row in rows]
    new_error = [row["new_area_abs_error"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(9.5, 5))
    plt.bar(x - width / 2, old_error, width, label="old area error", color="#9aa0a6")
    plt.bar(x + width / 2, new_error, width, label="new area error", color="#4f8a5f")
    plt.ylabel("Absolute error vs real mean area ratio")
    plt.title("Synthetic Distribution Repair: Area Match")
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "distribution_repair_summary.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_baseline_comparison(rows: list[dict[str, Any]]) -> None:
    labels = [row["method"] for row in rows]
    pixel_f1 = [row["pixel_f1"] for row in rows]
    image_f1 = [row["image_f1"] for row in rows]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7.8, 4.8))
    plt.bar(x - width / 2, pixel_f1, width, label="Pixel F1", color="#2f6f9f")
    plt.bar(x + width / 2, image_f1, width, label="Image F1", color="#d0d4d8")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(x, labels, rotation=10, ha="right")
    plt.title("Tile Baseline vs Synthetic-Supervised U-Net")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "baseline_vs_synthetic.png", dpi=180, bbox_inches="tight")
    plt.close()


def write_diagnostic_summary(
    threshold_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Diagnostic Evidence Summary",
        "",
        "Generated by `scripts/17_collect_diagnostics.py`.",
        "",
        "## Threshold And Postprocess Diagnostics",
        "",
        *markdown_table(
            threshold_rows,
            [
                "experiment_name",
                "role",
                "default_pixel_f1",
                "best_threshold",
                "best_threshold_f1",
                "threshold_f1_gain",
                "best_postprocess_min_area_ratio",
                "best_postprocess_f1",
                "postprocess_f1_gain",
            ],
        ),
        "",
        "## Distribution Repair Evidence",
        "",
        *markdown_table(
            distribution_rows,
            [
                "defect",
                "real_mean_area_ratio",
                "old_synthetic_mean_area_ratio",
                "new_synthetic_mean_area_ratio",
                "dice_before",
                "dice_after",
                "dice_gain",
            ],
        ),
        "",
        "## Baseline Comparison",
        "",
        *markdown_table(
            baseline_rows,
            ["category", "method", "role", "pixel_f1", "image_f1", "note"],
        ),
        "",
        "## Interpretation",
        "",
        "- Stage 11 remains the leather overall recommendation because threshold tuning improves Pixel F1 without changing the model or training data.",
        "- Stage 12 confirms fold recall can be recovered, but its lower default Pixel F1 and precision tradeoff keep it as a diagnostic specialist.",
        "- PatchCore is a strong image-level tile anomaly baseline, while synthetic-supervised U-Net is stronger for pixel-level segmentation after distribution repair.",
        "- The largest class-level gains align with synthetic area distribution moving closer to real defect statistics.",
        "",
    ]
    (FINAL_DIR / "diagnostic_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    threshold_rows = collect_threshold_postprocess()
    distribution_rows = collect_distribution_repairs()
    baseline_rows = collect_baseline_comparison()
    write_csv(threshold_rows, FINAL_DIR / "threshold_postprocess_summary.csv")
    write_csv(distribution_rows, FINAL_DIR / "distribution_repair_summary.csv")
    write_csv(baseline_rows, FINAL_DIR / "baseline_comparison_summary.csv")
    plot_threshold_sweep()
    plot_distribution_repair(distribution_rows)
    plot_baseline_comparison(baseline_rows)
    write_diagnostic_summary(threshold_rows, distribution_rows, baseline_rows)
    print("Diagnostic evidence collected.")
    print(f"Output dir: {FINAL_DIR.as_posix()}")


if __name__ == "__main__":
    main()

