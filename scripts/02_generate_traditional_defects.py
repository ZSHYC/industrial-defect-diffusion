from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFECT_TYPES = ["crack", "glue_strip", "gray_stroke", "oil", "rough"]


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    defect_pixels = mask.astype(bool)
    overlay[defect_pixels] = (1.0 - alpha) * overlay[defect_pixels] + alpha * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_soft_polygon(size: tuple[int, int], points: list[tuple[int, int]], blur_radius: float) -> np.ndarray:
    mask_img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(points, fill=255)
    if blur_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.array(mask_img, dtype=np.float32) / 255.0


def random_polyline(rng: np.random.Generator, width: int, height: int, point_count: int) -> list[tuple[int, int]]:
    margin = max(20, min(width, height) // 12)
    start_x = int(rng.integers(margin, width - margin))
    start_y = int(rng.integers(margin, height - margin))
    angle = float(rng.uniform(0, 2 * np.pi))
    step = float(rng.uniform(min(width, height) * 0.08, min(width, height) * 0.16))

    points = [(start_x, start_y)]
    x, y = float(start_x), float(start_y)
    for _ in range(point_count - 1):
        angle += float(rng.normal(0, 0.45))
        x += np.cos(angle) * step * float(rng.uniform(0.75, 1.25))
        y += np.sin(angle) * step * float(rng.uniform(0.75, 1.25))
        x = float(np.clip(x, margin, width - margin))
        y = float(np.clip(y, margin, height - margin))
        points.append((int(x), int(y)))
    return points


def generate_crack(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    height, width = image.shape[:2]
    line_width = int(rng.integers(5, 11))

    # Real tile cracks are usually long, dominant dark fractures with only small branches.
    center_x = float(rng.uniform(width * 0.25, width * 0.75))
    center_y = float(rng.uniform(height * 0.25, height * 0.75))
    angle = float(rng.uniform(0, np.pi))
    length = float(rng.uniform(min(width, height) * 0.55, min(width, height) * 0.85))
    point_count = int(rng.integers(5, 8))
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-np.sin(angle), np.cos(angle)])
    points: list[tuple[int, int]] = []
    for idx in range(point_count):
        t = -0.5 + idx / (point_count - 1)
        base = np.array([center_x, center_y]) + direction * length * t
        jitter = normal * float(rng.normal(0, min(width, height) * 0.025))
        x, y = base + jitter
        points.append((int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))))

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.line(points, fill=255, width=line_width, joint="curve")

    branch_count = int(rng.integers(1, 2))
    branches: list[list[tuple[int, int]]] = []
    for _ in range(branch_count):
        anchor = points[int(rng.integers(1, len(points) - 1))]
        branch = [anchor]
        branch_angle = angle + float(rng.choice([-1, 1])) * float(rng.uniform(0.65, 1.2))
        branch_length = float(rng.uniform(min(width, height) * 0.10, min(width, height) * 0.22))
        branch.append((
            int(np.clip(anchor[0] + np.cos(branch_angle) * branch_length, 0, width - 1)),
            int(np.clip(anchor[1] + np.sin(branch_angle) * branch_length, 0, height - 1)),
        ))
        draw.line(branch, fill=255, width=max(1, line_width - 2))
        branches.append(branch)

    soft_mask = np.array(mask_img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.2)))), dtype=np.float32) / 255.0
    mask = (np.array(mask_img) > 0).astype(np.uint8)

    dark_factor = float(rng.uniform(0.08, 0.24))
    output = image.astype(np.float32).copy()
    output = output * (1 - soft_mask[..., None]) + output * dark_factor * soft_mask[..., None]

    return output.astype(np.uint8), mask, {
        "line_width": line_width,
        "points": points,
        "branches": branches,
        "dark_factor": dark_factor,
    }


def generate_glue_strip(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    height, width = image.shape[:2]
    center_x = int(rng.integers(width * 0.25, width * 0.75))
    center_y = int(rng.integers(height * 0.25, height * 0.75))
    strip_w = float(rng.uniform(width * 0.055, width * 0.095))
    strip_h = float(rng.uniform(height * 0.24, height * 0.40))
    angle = float(rng.uniform(-0.9, 0.9))

    # Build a jagged strip instead of a perfect rectangle, closer to torn glue tape.
    segments = 7
    left_edge: list[list[float]] = []
    right_edge: list[list[float]] = []
    for idx in range(segments + 1):
        y = -strip_h / 2 + strip_h * idx / segments
        left_edge.append([
            -strip_w / 2 + float(rng.uniform(-strip_w * 0.25, strip_w * 0.15)),
            y + float(rng.uniform(-strip_h * 0.035, strip_h * 0.035)),
        ])
        right_edge.append([
            strip_w / 2 + float(rng.uniform(-strip_w * 0.15, strip_w * 0.25)),
            y + float(rng.uniform(-strip_h * 0.035, strip_h * 0.035)),
        ])
    corners = np.array(left_edge + right_edge[::-1])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points_arr = corners @ rotation.T + np.array([center_x, center_y])
    points = [(int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))) for x, y in points_arr]

    soft_mask = draw_soft_polygon((width, height), points, blur_radius=float(rng.uniform(1.5, 3.5)))
    mask = (soft_mask > 0.18).astype(np.uint8)

    tint = np.array([
        float(rng.uniform(185, 230)),
        float(rng.uniform(185, 230)),
        float(rng.uniform(175, 215)),
    ])
    alpha = float(rng.uniform(0.52, 0.72))
    output = image.astype(np.float32).copy()
    output = output * (1 - soft_mask[..., None] * alpha) + tint * (soft_mask[..., None] * alpha)

    highlight = np.array([235.0, 235.0, 225.0])
    output = output * (1 - soft_mask[..., None] * 0.08) + highlight * (soft_mask[..., None] * 0.08)

    return output.astype(np.uint8), mask, {
        "center": [center_x, center_y],
        "strip_width": strip_w,
        "strip_height": strip_h,
        "angle": angle,
        "alpha": alpha,
        "tint": tint.tolist(),
    }


def generate_gray_stroke(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    height, width = image.shape[:2]
    stroke_width = int(rng.integers(30, 58))
    points = random_polyline(rng, width, height, int(rng.integers(2, 4)))

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.line(points, fill=255, width=stroke_width, joint="curve")

    for point in points:
        radius_x = int(rng.integers(stroke_width // 2, stroke_width))
        radius_y = int(rng.integers(stroke_width // 3, max(stroke_width // 3 + 1, stroke_width * 2 // 3)))
        draw.ellipse(
            [
                point[0] - radius_x,
                point[1] - radius_y,
                point[0] + radius_x,
                point[1] + radius_y,
            ],
            fill=255,
        )

    soft_mask = np.array(mask_img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(3, 7)))), dtype=np.float32) / 255.0
    mask = (soft_mask > 0.12).astype(np.uint8)

    gray_value = float(rng.uniform(35, 75))
    alpha = float(rng.uniform(0.62, 0.85))
    gray = np.full_like(image.astype(np.float32), gray_value)
    output = image.astype(np.float32) * (1 - soft_mask[..., None] * alpha) + gray * (soft_mask[..., None] * alpha)

    return output.astype(np.uint8), mask, {
        "stroke_width": stroke_width,
        "points": points,
        "gray_value": gray_value,
        "alpha": alpha,
    }


def generate_oil(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    height, width = image.shape[:2]
    center_x = int(rng.integers(width * 0.25, width * 0.75))
    center_y = int(rng.integers(height * 0.25, height * 0.75))
    radius_x = float(rng.uniform(width * 0.10, width * 0.20))
    radius_y = float(rng.uniform(height * 0.08, height * 0.18))
    point_count = int(rng.integers(12, 20))

    points: list[tuple[int, int]] = []
    for index in range(point_count):
        angle = 2 * np.pi * index / point_count
        jitter = float(rng.uniform(0.45, 1.35))
        x = center_x + np.cos(angle) * radius_x * jitter
        y = center_y + np.sin(angle) * radius_y * jitter
        points.append((int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))))

    if rng.random() < 0.7:
        tail_anchor = points[int(rng.integers(0, len(points)))]
        tail_len = float(rng.uniform(min(width, height) * 0.05, min(width, height) * 0.12))
        tail_angle = float(rng.uniform(0, 2 * np.pi))
        points.append((
            int(np.clip(tail_anchor[0] + np.cos(tail_angle) * tail_len, 0, width - 1)),
            int(np.clip(tail_anchor[1] + np.sin(tail_angle) * tail_len, 0, height - 1)),
        ))

    soft_mask = draw_soft_polygon((width, height), points, blur_radius=float(rng.uniform(7, 14)))
    mask = (soft_mask > 0.16).astype(np.uint8)

    tint = np.array([
        float(rng.uniform(145, 190)),
        float(rng.uniform(135, 175)),
        float(rng.uniform(70, 110)),
    ])
    alpha = float(rng.uniform(0.32, 0.50))
    output = image.astype(np.float32) * (1 - soft_mask[..., None] * alpha) + tint * (soft_mask[..., None] * alpha)

    return output.astype(np.uint8), mask, {
        "center": [center_x, center_y],
        "radius_x": radius_x,
        "radius_y": radius_y,
        "point_count": point_count,
        "alpha": alpha,
        "tint": tint.tolist(),
    }


def generate_rough(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    height, width = image.shape[:2]
    center_x = int(rng.integers(width * 0.2, width * 0.8))
    center_y = int(rng.integers(height * 0.2, height * 0.8))
    radius_x = float(rng.uniform(width * 0.07, width * 0.15))
    radius_y = float(rng.uniform(height * 0.07, height * 0.16))
    point_count = int(rng.integers(10, 16))

    points: list[tuple[int, int]] = []
    for index in range(point_count):
        angle = 2 * np.pi * index / point_count
        jitter = float(rng.uniform(0.55, 1.35))
        x = center_x + np.cos(angle) * radius_x * jitter
        y = center_y + np.sin(angle) * radius_y * jitter
        points.append((int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))))

    soft_mask = draw_soft_polygon((width, height), points, blur_radius=float(rng.uniform(2, 5)))
    mask = (soft_mask > 0.18).astype(np.uint8)

    output = image.astype(np.float32).copy()
    noise = rng.normal(0, float(rng.uniform(22, 42)), size=image.shape).astype(np.float32)
    contrast = float(rng.uniform(1.35, 1.75))
    brightness_shift = float(rng.uniform(14, 36))
    local_mean = output.mean(axis=2, keepdims=True)
    rough_region = (output - local_mean) * contrast + local_mean + noise + brightness_shift
    desaturated = rough_region.mean(axis=2, keepdims=True)
    rough_region = rough_region * 0.55 + desaturated * 0.45
    output = output * (1 - soft_mask[..., None] * 0.9) + rough_region * (soft_mask[..., None] * 0.9)

    return np.clip(output, 0, 255).astype(np.uint8), mask, {
        "center": [center_x, center_y],
        "radius_x": radius_x,
        "radius_y": radius_y,
        "point_count": point_count,
        "contrast": contrast,
        "brightness_shift": brightness_shift,
    }


GENERATORS = {
    "crack": generate_crack,
    "glue_strip": generate_glue_strip,
    "gray_stroke": generate_gray_stroke,
    "oil": generate_oil,
    "rough": generate_rough,
}


def save_preview(rows: list[dict[str, object]], output_path: Path) -> None:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        defect_type = str(row["defect_type"])
        if defect_type not in seen:
            selected.append(row)
            seen.add(defect_type)
        if len(selected) == len(DEFECT_TYPES):
            break

    fig, axes = plt.subplots(len(selected), 4, figsize=(13, 3.4 * len(selected)))
    axes_array = np.array(axes).reshape(len(selected), 4)

    for row_index, row in enumerate(selected):
        source = read_rgb(Path(str(row["source_image"])))
        image = read_rgb(Path(str(row["output_image"])))
        mask = np.array(Image.open(str(row["output_mask"])).convert("L")) > 0
        overlay = overlay_mask(image, mask.astype(np.uint8))
        panels = [source, image, mask.astype(np.uint8) * 255, overlay]
        titles = [f"{row['defect_type']} source", "synthetic", "mask", "overlay"]
        for col_index, (panel, title) in enumerate(zip(panels, titles)):
            axes_array[row_index, col_index].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes_array[row_index, col_index].set_title(title, fontsize=10)
            axes_array[row_index, col_index].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_real_vs_traditional_preview(rows: list[dict[str, object]], data_root: Path, category: str, output_path: Path) -> None:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        defect_type = str(row["defect_type"])
        if defect_type not in seen:
            selected.append(row)
            seen.add(defect_type)
        if len(selected) == len(DEFECT_TYPES):
            break

    fig, axes = plt.subplots(len(selected), 6, figsize=(18, 3.2 * len(selected)))
    axes_array = np.array(axes).reshape(len(selected), 6)
    category_root = data_root / category

    for row_index, row in enumerate(selected):
        defect_type = str(row["defect_type"])
        real_image_path = category_root / "test" / defect_type / "000.png"
        real_mask_path = category_root / "ground_truth" / defect_type / "000_mask.png"
        synthetic_image_path = Path(str(row["output_image"]))
        synthetic_mask_path = Path(str(row["output_mask"]))

        if not real_image_path.exists() or not real_mask_path.exists():
            continue

        real_image = read_rgb(real_image_path)
        real_mask = (np.array(Image.open(real_mask_path).convert("L")) > 0).astype(np.uint8)
        synthetic_image = read_rgb(synthetic_image_path)
        synthetic_mask = (np.array(Image.open(synthetic_mask_path).convert("L")) > 0).astype(np.uint8)

        panels = [
            real_image,
            real_mask * 255,
            overlay_mask(real_image, real_mask),
            synthetic_image,
            synthetic_mask * 255,
            overlay_mask(synthetic_image, synthetic_mask),
        ]
        titles = [
            f"real {defect_type}",
            "real mask",
            "real overlay",
            f"traditional {defect_type}",
            "traditional mask",
            "traditional overlay",
        ]
        for col_index, (panel, title) in enumerate(zip(panels, titles)):
            axes_array[row_index, col_index].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes_array[row_index, col_index].set_title(title, fontsize=9)
            axes_array[row_index, col_index].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate traditional synthetic defects for MVTec AD tile images.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\zsh\Desktop\昂坤视觉\MVTec_AD"),
        help="Path to the MVTec AD dataset root.",
    )
    parser.add_argument("--category", default="tile", help="MVTec AD category name.")
    parser.add_argument("--samples-per-type", type=int, default=5, help="Number of samples per defect type.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "traditional_synthetic",
        help="Output directory. Category name is appended automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_root = args.data_root / args.category
    train_good_dir = category_root / "train" / "good"
    if not train_good_dir.exists():
        raise FileNotFoundError(f"Required normal training image directory not found: {train_good_dir}")

    normal_images = list_images(train_good_dir)
    if not normal_images:
        raise FileNotFoundError(f"No normal training images found in: {train_good_dir}")

    output_root = args.output_dir / args.category
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    metadata_dir = output_root / "metadata"
    for folder in [image_dir, mask_dir, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for defect_type in DEFECT_TYPES:
        generator = GENERATORS[defect_type]
        for index in range(args.samples_per_type):
            source_path = normal_images[int(rng.integers(0, len(normal_images)))]
            source = read_rgb(source_path)
            synthetic, mask, parameters = generator(source, rng)

            sample_name = f"{args.category}_trad_{defect_type}_{index:03d}"
            image_path = image_dir / f"{sample_name}.png"
            mask_path = mask_dir / f"{sample_name}_mask.png"
            metadata_path = metadata_dir / f"{sample_name}.json"

            save_rgb(synthetic, image_path)
            save_mask(mask, mask_path)

            metadata = {
                "source_image": str(source_path.resolve()),
                "defect_type": defect_type,
                "output_image": str(image_path.resolve()),
                "output_mask": str(mask_path.resolve()),
                "seed": args.seed,
                "generation_method": "traditional_rule_based",
                "parameters": parameters,
            }
            write_json(metadata, metadata_path)

            rows.append({
                "sample_name": sample_name,
                "defect_type": defect_type,
                "source_image": str(source_path.resolve()),
                "output_image": str(image_path.resolve()),
                "output_mask": str(mask_path.resolve()),
                "metadata": str(metadata_path.resolve()),
                "mask_pixels": int((mask > 0).sum()),
                "mask_area_ratio": float((mask > 0).sum() / mask.size),
            })

    write_csv(rows, output_root / "summary.csv")
    save_preview(rows, output_root / "preview.png")
    save_real_vs_traditional_preview(rows, args.data_root, args.category, output_root / "real_vs_traditional_preview.png")

    print(f"Generated {len(rows)} traditional synthetic defects.")
    print(f"Output dir: {output_root.as_posix()}")
    for defect_type in DEFECT_TYPES:
        count = sum(1 for row in rows if row["defect_type"] == defect_type)
        print(f"  {defect_type}: {count}")


if __name__ == "__main__":
    main()
