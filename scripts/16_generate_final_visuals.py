from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs") / "final_report"
FIGURE_DIR = OUTPUT_DIR / "figures"
SUMMARY_CSV = OUTPUT_DIR / "final_metrics_summary.csv"
CLASS_CSV = OUTPUT_DIR / "final_class_metrics.csv"
PAPER_TABLES = OUTPUT_DIR / "final_paper_tables.md"

COLORS = {
    "tile": "#2f6f9f",
    "wood": "#4f8a5f",
    "leather": "#9a5b3f",
    "baseline": "#9aa0a6",
    "highlight": "#c44e52",
    "tradeoff": "#8172b3",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def fmt(value: float) -> str:
    return f"{value:.4f}"


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_recommended_models(summary_rows: list[dict[str, str]]) -> None:
    rows = [row for row in summary_rows if row["role"] in {"overall_best", "class_specialist", "diagnostic_tradeoff"}]
    labels = [
        "tile\nStage 6",
        "tile crack\nStage 7",
        "wood\nStage 9",
        "leather\nStage 11",
        "fold tradeoff\nStage 12",
    ]
    values = [as_float(row, "pixel_f1") for row in rows]
    image_values = [as_float(row, "image_f1") for row in rows]
    colors = [
        COLORS["tile"],
        COLORS["tile"],
        COLORS["wood"],
        COLORS["leather"],
        COLORS["tradeoff"],
    ]

    x = np.arange(len(rows))
    width = 0.36
    plt.figure(figsize=(9.5, 4.8))
    plt.bar(x - width / 2, values, width, label="Pixel F1", color=colors)
    plt.bar(x + width / 2, image_values, width, label="Image F1", color="#d0d4d8")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(x, labels)
    plt.title("Final Recommended Models")
    plt.legend(frameon=False, loc="upper right")
    for index, value in enumerate(values):
        plt.text(index - width / 2, value + 0.025, fmt(value), ha="center", fontsize=9)
    savefig(FIGURE_DIR / "final_recommended_models.png")


def plot_category_progression(summary_rows: list[dict[str, str]]) -> None:
    order = {
        "tile": ["tile stage5 combined", "tile stage6 expanded combined", "tile stage6 gray_stroke fixed", "tile stage7 crack fixed"],
        "wood": ["wood stage8 generalization", "wood stage9 scratch fixed"],
        "leather": ["leather stage10 generalization", "leather stage11 precision cut fixed", "leather stage12 fold fixed"],
    }
    by_name = {row["experiment_name"]: row for row in summary_rows}
    plt.figure(figsize=(10, 5))
    for category, names in order.items():
        y = [as_float(by_name[name], "pixel_f1") for name in names]
        x = np.arange(len(names))
        plt.plot(x, y, marker="o", linewidth=2.4, label=category, color=COLORS[category])
        for index, value in enumerate(y):
            plt.text(index, value + 0.025, fmt(value), ha="center", fontsize=8)
    plt.xticks([0, 1, 2, 3], ["baseline", "diagnostic", "repair", "specialist"])
    plt.ylabel("Pixel F1")
    plt.ylim(0, 0.95)
    plt.title("Metric Progression Across Project Stages")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.25)
    savefig(FIGURE_DIR / "category_progression.png")


def find_class(class_rows: list[dict[str, str]], experiment_name: str, defect_type: str) -> dict[str, str]:
    for row in class_rows:
        if row["experiment_name"] == experiment_name and row["defect_type"] == defect_type:
            return row
    raise KeyError(f"Missing class metric: {experiment_name} / {defect_type}")


def plot_class_repairs(class_rows: list[dict[str, str]]) -> None:
    repairs = [
        ("tile gray_stroke", "gray_stroke", "tile stage6 expanded combined", "tile stage6 gray_stroke fixed"),
        ("tile crack", "crack", "tile stage6 gray_stroke fixed", "tile stage7 crack fixed"),
        ("wood scratch", "scratch", "wood stage8 generalization", "wood stage9 scratch fixed"),
        ("leather cut", "cut", "leather stage10 generalization", "leather stage11 precision cut fixed"),
        ("leather fold", "fold", "leather stage11 precision cut fixed", "leather stage12 fold fixed"),
    ]
    before = [as_float(find_class(class_rows, old, defect), "mean_dice") for _, defect, old, _ in repairs]
    after = [as_float(find_class(class_rows, new, defect), "mean_dice") for _, defect, _, new in repairs]
    labels = [label for label, _, _, _ in repairs]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, before, width, label="before", color=COLORS["baseline"])
    plt.bar(x + width / 2, after, width, label="after", color=COLORS["highlight"])
    plt.ylabel("Mean Dice")
    plt.ylim(0, 0.95)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.title("Class-Level Repair Before / After")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.22)
    for index, value in enumerate(after):
        plt.text(index + width / 2, value + 0.025, fmt(value), ha="center", fontsize=8)
    savefig(FIGURE_DIR / "class_repair_before_after.png")


def plot_precision_recall(summary_rows: list[dict[str, str]]) -> None:
    rows = [row for row in summary_rows if row["category"] in {"tile", "wood", "leather"}]
    plt.figure(figsize=(7.8, 5.4))
    for row in rows:
        color = COLORS[row["category"]]
        marker = "X" if row["role"] in {"overall_best", "class_specialist", "diagnostic_tradeoff"} else "o"
        size = 110 if marker == "X" else 55
        plt.scatter(as_float(row, "pixel_recall"), as_float(row, "pixel_precision"), s=size, marker=marker, color=color, alpha=0.86)
        if row["role"] in {"overall_best", "diagnostic_tradeoff"}:
            label = row["experiment_name"].replace(" fixed", "").replace(" generalization", "")
            plt.text(as_float(row, "pixel_recall") + 0.015, as_float(row, "pixel_precision"), label, fontsize=8)
    plt.xlabel("Pixel Recall")
    plt.ylabel("Pixel Precision")
    plt.xlim(0, 0.9)
    plt.ylim(0, 1.02)
    plt.title("Pixel Precision / Recall Tradeoff")
    plt.grid(alpha=0.25)
    savefig(FIGURE_DIR / "precision_recall_tradeoff.png")


def markdown_table(rows: list[list[str]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_paper_tables(summary_rows: list[dict[str, str]], class_rows: list[dict[str, str]]) -> None:
    recommended = [row for row in summary_rows if row["role"] in {"overall_best", "class_specialist", "diagnostic_tradeoff"}]
    recommended_rows = [
        [
            row["category"],
            row["experiment_name"],
            row["role"],
            fmt(as_float(row, "pixel_f1")),
            fmt(as_float(row, "best_pixel_f1")),
            fmt(as_float(row, "image_f1")),
        ]
        for row in recommended
    ]

    repair_pairs = [
        ("tile", "gray_stroke", "tile stage6 expanded combined", "tile stage6 gray_stroke fixed"),
        ("tile", "crack", "tile stage6 gray_stroke fixed", "tile stage7 crack fixed"),
        ("wood", "scratch", "wood stage8 generalization", "wood stage9 scratch fixed"),
        ("leather", "cut", "leather stage10 generalization", "leather stage11 precision cut fixed"),
        ("leather", "fold", "leather stage11 precision cut fixed", "leather stage12 fold fixed"),
    ]
    repair_rows = []
    for category, defect, old, new in repair_pairs:
        old_row = find_class(class_rows, old, defect)
        new_row = find_class(class_rows, new, defect)
        repair_rows.append([
            category,
            defect,
            old.split(" ", 1)[1],
            fmt(as_float(old_row, "mean_dice")),
            new.split(" ", 1)[1],
            fmt(as_float(new_row, "mean_dice")),
        ])

    lines = [
        "# Final Paper Tables",
        "",
        "Generated by `scripts/16_generate_final_visuals.py`.",
        "",
        "## Table 1. Recommended Final Models",
        "",
        *markdown_table(recommended_rows, ["category", "experiment", "role", "pixel_f1", "best_pixel_f1", "image_f1"]),
        "",
        "## Table 2. Class Repair Before / After",
        "",
        *markdown_table(repair_rows, ["category", "defect", "before", "dice_before", "after", "dice_after"]),
        "",
        "## Table 3. Reproducibility Artifacts",
        "",
        *markdown_table(
            [
                ["final metrics", "`outputs/final_report/final_metrics_summary.csv`"],
                ["class metrics", "`outputs/final_report/final_class_metrics.csv`"],
                ["dashboard", "`outputs/final_report/final_results_dashboard.md`"],
                ["health check", "`outputs/final_report/project_health_check.md`"],
                ["configuration", "`configs/categories.json`, `configs/final_experiments.json`"],
                ["tests", "`tests/`"],
            ],
            ["artifact", "path"],
        ),
        "",
    ]
    PAPER_TABLES.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary_rows = read_csv(SUMMARY_CSV)
    class_rows = read_csv(CLASS_CSV)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_recommended_models(summary_rows)
    plot_category_progression(summary_rows)
    plot_class_repairs(class_rows)
    plot_precision_recall(summary_rows)
    write_paper_tables(summary_rows, class_rows)
    print(f"Final figures written to {FIGURE_DIR.as_posix()}")
    print(f"Paper tables written to {PAPER_TABLES.as_posix()}")


if __name__ == "__main__":
    main()

