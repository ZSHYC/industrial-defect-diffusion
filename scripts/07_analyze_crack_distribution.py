from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_defect.config import IMAGE_EXTENSIONS  # noqa: E402
from industrial_defect.paths import resolve_data_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real and synthetic tile crack distributions.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to the MVTec AD dataset root. Defaults to DATA_ROOT.",
    )
    parser.add_argument("--category", default="tile")
    parser.add_argument(
        "--old-traditional-summary",
        type=Path,
        default=Path("outputs") / "expanded_synthetic" / "traditional" / "tile" / "summary.csv",
    )
    parser.add_argument(
        "--old-diffusion-summary",
        type=Path,
        default=Path("outputs") / "expanded_synthetic" / "diffusion" / "tile" / "summary.csv",
    )
    parser.add_argument(
        "--new-traditional-summary",
        type=Path,
        default=Path("outputs") / "stage7_synthetic" / "traditional" / "tile" / "summary.csv",
    )
    parser.add_argument(
        "--new-diffusion-summary",
        type=Path,
        default=Path("outputs") / "stage7_synthetic" / "diffusion" / "tile" / "summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs") / "crack_fix" / "crack_distribution_summary.csv",
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def bbox_stats(mask: np.ndarray) -> tuple[int, int, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0.0, 0.0
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    aspect_ratio = float(width / max(height, 1))
    fill_ratio = float(mask.sum() / max(width * height, 1))
    return width, height, aspect_ratio, fill_ratio


def sample_stats(source: str, image_path: Path, mask_path: Path) -> dict[str, object]:
    gray = read_gray(image_path)
    mask = read_mask(mask_path)
    outside = ~mask
    bbox_width, bbox_height, bbox_aspect_ratio, bbox_fill_ratio = bbox_stats(mask)
    inside_gray = float(gray[mask].mean()) if mask.any() else 0.0
    outside_gray = float(gray[outside].mean()) if outside.any() else 0.0
    return {
        "source": source,
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "area_ratio": float(mask.sum() / mask.size),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "bbox_fill_ratio": bbox_fill_ratio,
        "inside_gray": inside_gray,
        "outside_gray": outside_gray,
        "inside_minus_outside": inside_gray - outside_gray,
    }


def load_real_rows(data_root: Path, category: str) -> list[dict[str, object]]:
    image_dir = data_root / category / "test" / "crack"
    mask_dir = data_root / category / "ground_truth" / "crack"
    rows: list[dict[str, object]] = []
    for image_path in list_images(image_dir):
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path.as_posix()}")
        rows.append(sample_stats("real", image_path, mask_path))
    return rows


def load_synthetic_rows(summary_path: Path, source: str) -> list[dict[str, object]]:
    if not summary_path.exists():
        raise FileNotFoundError(f"{source} summary not found: {summary_path.as_posix()}")
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        summary_rows = list(csv.DictReader(file))
    rows: list[dict[str, object]] = []
    for row in summary_rows:
        if row["defect_type"] != "crack":
            continue
        rows.append(sample_stats(source, Path(row["output_image"]), Path(row["output_mask"])))
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "area_ratio",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "bbox_fill_ratio",
        "inside_gray",
        "outside_gray",
        "inside_minus_outside",
    ]
    summary_rows: list[dict[str, object]] = []
    sources = ["real", "old_traditional", "old_diffusion", "new_traditional", "new_diffusion"]
    for source in sources:
        source_rows = [row for row in rows if row["source"] == source]
        if not source_rows:
            continue
        summary: dict[str, object] = {"source": source, "count": len(source_rows)}
        for field in fields:
            values = np.array([float(row[field]) for row in source_rows], dtype=np.float64)
            summary[f"mean_{field}"] = float(values.mean())
            summary[f"min_{field}"] = float(values.min())
            summary[f"max_{field}"] = float(values.max())
        summary_rows.append(summary)
    return summary_rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.data_root = resolve_data_root(args.data_root)
    rows: list[dict[str, object]] = []
    rows.extend(load_real_rows(args.data_root, args.category))
    rows.extend(load_synthetic_rows(args.old_traditional_summary, "old_traditional"))
    rows.extend(load_synthetic_rows(args.old_diffusion_summary, "old_diffusion"))
    rows.extend(load_synthetic_rows(args.new_traditional_summary, "new_traditional"))
    rows.extend(load_synthetic_rows(args.new_diffusion_summary, "new_diffusion"))

    summary_rows = summarize(rows)
    write_csv(summary_rows, args.output_csv)
    write_csv(rows, args.output_csv.with_name("crack_distribution_samples.csv"))

    print("Crack distribution analysis finished.")
    print(f"Summary: {args.output_csv.as_posix()}")
    for row in summary_rows:
        print(
            f"  {row['source']}: count={row['count']}, "
            f"mean_area_ratio={float(row['mean_area_ratio']):.4f}, "
            f"mean_inside_minus_outside={float(row['mean_inside_minus_outside']):.2f}"
        )


if __name__ == "__main__":
    main()
