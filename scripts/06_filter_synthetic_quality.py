from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFECT_TYPES = ["crack", "glue_strip", "gray_stroke", "oil", "rough"]
SOURCES = ["traditional", "diffusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter synthetic defect samples with simple quality rules.")
    parser.add_argument(
        "--traditional-summary",
        type=Path,
        default=Path("outputs") / "expanded_synthetic" / "traditional" / "tile" / "summary.csv",
    )
    parser.add_argument(
        "--diffusion-summary",
        type=Path,
        default=Path("outputs") / "expanded_synthetic" / "diffusion" / "tile" / "summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "quality_filter" / "tile",
    )
    parser.add_argument("--min-mask-area", type=float, default=0.001)
    parser.add_argument("--max-mask-area", type=float, default=0.25)
    parser.add_argument("--min-inside-diff", type=float, default=4.0)
    parser.add_argument("--max-outside-diff", type=float, default=40.0)
    return parser.parse_args()


def read_csv(path: Path, source: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{source} summary not found: {path.as_posix()}")
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["synthetic_source"] = source
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    return mask > 0


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    output = image.copy()
    output[mask] = (1.0 - alpha) * output[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def score_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    source_path = Path(row["source_image"])
    image_path = Path(row["output_image"])
    mask_path = Path(row["output_mask"])
    for path in [source_path, image_path, mask_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path.as_posix()}")

    source = load_rgb(source_path)
    image = load_rgb(image_path)
    mask = load_mask(mask_path)
    if source.shape != image.shape:
        raise ValueError(f"Source and synthetic image size mismatch: {image_path.as_posix()}")

    area_ratio = float(mask.mean())
    diff = np.abs(image - source).mean(axis=2)
    inside_diff = float(diff[mask].mean()) if mask.any() else 0.0
    outside_mask = ~mask
    outside_diff = float(diff[outside_mask].mean()) if outside_mask.any() else 0.0
    outside_p95 = float(np.percentile(diff[outside_mask], 95)) if outside_mask.any() else 0.0

    reject_reasons: list[str] = []
    if area_ratio < args.min_mask_area:
        reject_reasons.append("mask_too_small")
    if area_ratio > args.max_mask_area:
        reject_reasons.append("mask_too_large")
    if inside_diff < args.min_inside_diff:
        reject_reasons.append("defect_change_too_weak")
    if outside_diff > args.max_outside_diff:
        reject_reasons.append("background_changed_too_much")

    accepted = len(reject_reasons) == 0
    scored = dict(row)
    scored.update(
        {
            "accepted": int(accepted),
            "reject_reasons": ";".join(reject_reasons),
            "mask_area_ratio": area_ratio,
            "inside_mean_abs_diff": inside_diff,
            "outside_mean_abs_diff": outside_diff,
            "outside_p95_abs_diff": outside_p95,
        }
    )
    return scored


def select_preview_rows(rows: list[dict[str, object]], max_rows: int = 10) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for source in SOURCES:
        for defect_type in DEFECT_TYPES:
            key = (source, defect_type)
            for row in rows:
                if row["synthetic_source"] == source and row["defect_type"] == defect_type and key not in seen:
                    selected.append(row)
                    seen.add(key)
                    break
    return selected[:max_rows]


def save_preview(rows: list[dict[str, object]], output_path: Path, title: str) -> None:
    selected = select_preview_rows(rows)
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 5, figsize=(16, 3.0 * len(selected)))
    axes_array = np.array(axes).reshape(len(selected), 5)
    for row_index, row in enumerate(selected):
        source = load_rgb(Path(str(row["source_image"])))
        image = load_rgb(Path(str(row["output_image"])))
        mask = load_mask(Path(str(row["output_mask"])))
        diff = np.abs(image - source).mean(axis=2)
        panels = [
            source.astype(np.uint8),
            image.astype(np.uint8),
            mask.astype(np.uint8) * 255,
            diff,
            overlay_mask(image, mask),
        ]
        titles = [
            f"{row['synthetic_source']} {row['defect_type']}",
            "synthetic",
            "mask",
            "abs diff",
            "overlay",
        ]
        for col_index, (panel, panel_title) in enumerate(zip(panels, titles)):
            axes_array[row_index, col_index].imshow(panel, cmap="gray" if np.array(panel).ndim == 2 else None)
            axes_array[row_index, col_index].set_title(panel_title, fontsize=9)
            axes_array[row_index, col_index].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for source in SOURCES:
        for defect_type in DEFECT_TYPES:
            subset = [row for row in rows if row["synthetic_source"] == source and row["defect_type"] == defect_type]
            if not subset:
                continue
            accepted = [row for row in subset if int(row["accepted"]) == 1]
            summary_rows.append(
                {
                    "synthetic_source": source,
                    "defect_type": defect_type,
                    "total": len(subset),
                    "accepted": len(accepted),
                    "rejected": len(subset) - len(accepted),
                    "accept_rate": len(accepted) / len(subset),
                    "mean_mask_area_ratio": float(np.mean([float(row["mask_area_ratio"]) for row in subset])),
                    "mean_inside_abs_diff": float(np.mean([float(row["inside_mean_abs_diff"]) for row in subset])),
                    "mean_outside_abs_diff": float(np.mean([float(row["outside_mean_abs_diff"]) for row in subset])),
                }
            )
    return summary_rows


def write_summary_md(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Stage 6 Synthetic Quality Filter Summary",
        "",
        "This file is generated by scripts/06_filter_synthetic_quality.py.",
        "",
        "| source | defect_type | total | accepted | rejected | accept_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['synthetic_source']} | {row['defect_type']} | {row['total']} | "
            f"{row['accepted']} | {row['rejected']} | {float(row['accept_rate']):.3f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_csv(args.traditional_summary, "traditional") + read_csv(args.diffusion_summary, "diffusion")
    scored_rows = [score_row(row, args) for row in rows]
    accepted = [row for row in scored_rows if int(row["accepted"]) == 1]
    rejected = [row for row in scored_rows if int(row["accepted"]) == 0]
    accepted_traditional = [row for row in accepted if row["synthetic_source"] == "traditional"]
    accepted_diffusion = [row for row in accepted if row["synthetic_source"] == "diffusion"]
    summary_rows = summarize(scored_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(scored_rows, args.output_dir / "quality_report.csv")
    write_csv(accepted, args.output_dir / "accepted_summary.csv")
    write_csv(rejected, args.output_dir / "rejected_summary.csv")
    write_csv(accepted_traditional, args.output_dir / "accepted_traditional_summary.csv")
    write_csv(accepted_diffusion, args.output_dir / "accepted_diffusion_summary.csv")
    write_csv(summary_rows, args.output_dir / "filter_summary.csv")
    save_preview(accepted, args.output_dir / "preview_accepted.png", "accepted synthetic samples")
    save_preview(rejected, args.output_dir / "preview_rejected.png", "rejected synthetic samples")
    write_summary_md(summary_rows, args.output_dir / "summary.md")

    print("Synthetic quality filtering finished.")
    print(f"Output dir: {args.output_dir.as_posix()}")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")


if __name__ == "__main__":
    main()
