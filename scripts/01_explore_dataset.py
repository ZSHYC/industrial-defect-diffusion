from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    return (mask > 0).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image and mask shapes do not match: {image.shape[:2]} vs {mask.shape}")

    overlay = image.astype(np.float32).copy()
    color_array = np.array(color, dtype=np.float32)
    defect_pixels = mask.astype(bool)
    overlay[defect_pixels] = (1.0 - alpha) * overlay[defect_pixels] + alpha * color_array
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_image_grid(images: list[np.ndarray], titles: list[str], output_path: Path, columns: int = 3) -> None:
    if not images:
        return
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.2 * rows))
    axes_array = np.array(axes).reshape(-1)

    for axis in axes_array:
        axis.axis("off")

    for axis, image, title in zip(axes_array, images, titles):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def expected_mask_path(ground_truth_root: Path, defect_type: str, image_path: Path) -> Path:
    return ground_truth_root / defect_type / f"{image_path.stem}_mask.png"


def collect_dataset_stats(category_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_good = list_images(category_root / "train" / "good")
    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"

    summary_rows: list[dict[str, object]] = [
        {
            "split": "train",
            "defect_type": "good",
            "image_count": len(train_good),
            "mask_count": 0,
            "missing_masks": 0,
        }
    ]

    defect_area_rows: list[dict[str, object]] = []

    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        defect_type = defect_dir.name
        images = list_images(defect_dir)
        mask_count = 0
        missing_masks = 0

        if defect_type == "good":
            summary_rows.append(
                {
                    "split": "test",
                    "defect_type": defect_type,
                    "image_count": len(images),
                    "mask_count": 0,
                    "missing_masks": 0,
                }
            )
            continue

        for image_path in images:
            mask_path = expected_mask_path(ground_truth_root, defect_type, image_path)
            if not mask_path.exists():
                missing_masks += 1
                continue

            mask_count += 1
            image = read_rgb(image_path)
            mask = read_mask(mask_path)
            if image.shape[:2] != mask.shape:
                raise ValueError(
                    f"Shape mismatch: image={image_path}, image_shape={image.shape[:2]}, "
                    f"mask={mask_path}, mask_shape={mask.shape}"
                )

            area_ratio = float(mask.sum() / mask.size)
            defect_area_rows.append(
                {
                    "defect_type": defect_type,
                    "image_name": image_path.name,
                    "mask_name": mask_path.name,
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "defect_pixels": int(mask.sum()),
                    "total_pixels": int(mask.size),
                    "defect_area_ratio": area_ratio,
                }
            )

        summary_rows.append(
            {
                "split": "test",
                "defect_type": defect_type,
                "image_count": len(images),
                "mask_count": mask_count,
                "missing_masks": missing_masks,
            }
        )

    return summary_rows, defect_area_rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_normal_samples(category_root: Path, output_dir: Path, max_samples: int) -> None:
    train_good = list_images(category_root / "train" / "good")[:max_samples]
    images = [read_rgb(path) for path in train_good]
    titles = [f"train/good/{path.name}" for path in train_good]
    save_image_grid(images, titles, output_dir / "normal_train_samples.png", columns=min(3, max_samples))


def create_defect_overlays(category_root: Path, output_dir: Path) -> None:
    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"

    images: list[np.ndarray] = []
    titles: list[str] = []

    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir() and path.name != "good"):
        defect_images = list_images(defect_dir)
        if not defect_images:
            continue
        image_path = defect_images[0]
        mask_path = expected_mask_path(ground_truth_root, defect_dir.name, image_path)
        if not mask_path.exists():
            continue

        image = read_rgb(image_path)
        mask = read_mask(mask_path)
        images.extend([image, mask * 255, overlay_mask(image, mask)])
        titles.extend([
            f"{defect_dir.name}/{image_path.name}",
            "mask",
            "overlay",
        ])

    save_image_grid(images, titles, output_dir / "defect_mask_overlays.png", columns=3)


def create_area_plot(defect_area_rows: list[dict[str, object]], output_dir: Path) -> None:
    if not defect_area_rows:
        return

    defect_types = sorted({str(row["defect_type"]) for row in defect_area_rows})
    data = [
        [100.0 * float(row["defect_area_ratio"]) for row in defect_area_rows if row["defect_type"] == defect_type]
        for defect_type in defect_types
    ]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(data, tick_labels=defect_types, showmeans=True)
    axis.set_title("Defect Area Ratio by Defect Type")
    axis.set_ylabel("Defect area ratio (%)")
    axis.set_xlabel("Defect type")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "defect_area_ratio_boxplot.png", dpi=160)
    plt.close(fig)


def validate_category_root(category_root: Path) -> None:
    required_paths = [
        category_root / "train" / "good",
        category_root / "test",
        category_root / "ground_truth",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required MVTec AD path not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore one category from the MVTec AD dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD"),
        help="Path to the MVTec AD dataset root.",
    )
    parser.add_argument("--category", default="tile", help="MVTec AD category name, for example: tile.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "eda",
        help="Directory for EDA outputs. Category name is appended automatically.",
    )
    parser.add_argument("--max-normal-samples", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_root = args.data_root / args.category
    output_dir = args.output_dir / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_category_root(category_root)

    summary_rows, defect_area_rows = collect_dataset_stats(category_root)
    write_csv(summary_rows, output_dir / "dataset_summary.csv")
    write_csv(defect_area_rows, output_dir / "defect_area_stats.csv")

    create_normal_samples(category_root, output_dir, args.max_normal_samples)
    create_defect_overlays(category_root, output_dir)
    create_area_plot(defect_area_rows, output_dir)

    print(f"Category: {args.category}")
    print(f"Dataset root: {args.data_root}")
    print(f"Output dir: {output_dir.resolve()}")
    print("")
    print("Dataset summary:")
    for row in summary_rows:
        print(
            f"  {row['split']:>5} | {row['defect_type']:<12} | "
            f"images={row['image_count']:<3} masks={row['mask_count']:<3} missing_masks={row['missing_masks']}"
        )

    if defect_area_rows:
        ratios = np.array([float(row["defect_area_ratio"]) for row in defect_area_rows])
        print("")
        print(
            "Defect area ratio: "
            f"min={100 * ratios.min():.3f}% "
            f"mean={100 * ratios.mean():.3f}% "
            f"max={100 * ratios.max():.3f}%"
        )


if __name__ == "__main__":
    main()
