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

SOURCES = ["real", "old_traditional", "old_diffusion", "new_traditional", "new_diffusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real and synthetic wood scratch distributions.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to the MVTec AD dataset root. Defaults to DATA_ROOT.",
    )
    parser.add_argument("--category", default="wood")
    parser.add_argument(
        "--old-traditional-summary",
        type=Path,
        default=Path("outputs") / "stage8_wood_synthetic" / "traditional" / "wood" / "summary.csv",
    )
    parser.add_argument(
        "--old-diffusion-summary",
        type=Path,
        default=Path("outputs") / "stage8_wood_synthetic" / "diffusion" / "wood" / "summary.csv",
    )
    parser.add_argument(
        "--new-traditional-summary",
        type=Path,
        default=Path("outputs") / "stage9_wood_scratch_fix" / "traditional" / "wood" / "summary.csv",
    )
    parser.add_argument(
        "--new-diffusion-summary",
        type=Path,
        default=Path("outputs") / "stage9_wood_scratch_fix" / "diffusion" / "wood" / "summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "stage9_wood_scratch_fix" / "analysis",
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def bbox_stats(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    height, width = mask.shape
    bbox_width = float(xs.max() - xs.min() + 1)
    bbox_height = float(ys.max() - ys.min() + 1)
    bbox_area = max(bbox_width * bbox_height, 1.0)
    return (
        bbox_width / width,
        bbox_height / height,
        bbox_width / max(bbox_height, 1.0),
        float(mask.sum() / bbox_area),
    )


def sample_stats(source: str, image_path: Path, mask_path: Path) -> dict[str, object]:
    gray = read_gray(image_path)
    mask = read_mask(mask_path)
    outside = ~mask
    bbox_width_ratio, bbox_height_ratio, bbox_aspect_ratio, bbox_fill_ratio = bbox_stats(mask)
    inside_gray = float(gray[mask].mean()) if mask.any() else 0.0
    outside_gray = float(gray[outside].mean()) if outside.any() else 0.0
    return {
        "source": source,
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "area_ratio": float(mask.sum() / mask.size),
        "bbox_width_ratio": bbox_width_ratio,
        "bbox_height_ratio": bbox_height_ratio,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "bbox_fill_ratio": bbox_fill_ratio,
        "inside_gray": inside_gray,
        "outside_gray": outside_gray,
        "inside_minus_outside": inside_gray - outside_gray,
    }


def load_real_rows(data_root: Path, category: str) -> list[dict[str, object]]:
    image_dir = data_root / category / "test" / "scratch"
    mask_dir = data_root / category / "ground_truth" / "scratch"
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
        if row["defect_type"] != "scratch":
            continue
        rows.append(sample_stats(source, Path(row["output_image"]), Path(row["output_mask"])))
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "area_ratio",
        "bbox_width_ratio",
        "bbox_height_ratio",
        "bbox_aspect_ratio",
        "bbox_fill_ratio",
        "inside_gray",
        "outside_gray",
        "inside_minus_outside",
    ]
    summary_rows: list[dict[str, object]] = []
    for source in SOURCES:
        source_rows = [row for row in rows if row["source"] == source]
        if not source_rows:
            continue
        summary: dict[str, object] = {"source": source, "count": len(source_rows)}
        for field in fields:
            values = np.array([float(row[field]) for row in source_rows], dtype=np.float64)
            summary[f"mean_{field}"] = float(values.mean())
            summary[f"median_{field}"] = float(np.median(values))
            summary[f"min_{field}"] = float(values.min())
            summary[f"max_{field}"] = float(values.max())
        summary_rows.append(summary)
    return summary_rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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
    write_csv(summary_rows, args.output_dir / "wood_scratch_distribution_summary.csv")
    write_csv(rows, args.output_dir / "wood_scratch_distribution_samples.csv")

    print("Wood scratch distribution analysis finished.")
    print(f"Output dir: {args.output_dir.as_posix()}")
    for row in summary_rows:
        print(
            f"  {row['source']}: count={row['count']}, "
            f"mean_area_ratio={float(row['mean_area_ratio']):.4f}, "
            f"mean_bbox_w={float(row['mean_bbox_width_ratio']):.4f}, "
            f"mean_bbox_h={float(row['mean_bbox_height_ratio']):.4f}, "
            f"mean_inside_minus_outside={float(row['mean_inside_minus_outside']):.2f}"
        )


if __name__ == "__main__":
    main()
