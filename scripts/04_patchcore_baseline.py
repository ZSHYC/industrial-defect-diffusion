from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


DEFECT_TYPES = ["crack", "glue_strip", "gray_stroke", "oil", "rough"]
TEST_TYPES = ["good", *DEFECT_TYPES]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class TestSample:
    image_path: Path
    mask_path: Path | None
    defect_type: str
    label: int


class ResNetPatchFeatureExtractor(torch.nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=None)
            backbone.load_state_dict(weights.get_state_dict(progress=False))
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pretrained ResNet18 weights. Check network access or torchvision cache."
            ) from exc

        self.stem = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([layer2, layer3], dim=1)
        return F.normalize(features, p=2, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight PatchCore-style anomaly detection baseline.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", default="tile")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-memory-patches", type=int, default=10000)
    parser.add_argument("--distance-chunk-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "baselines" / "patchcore")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear previous PatchCore outputs.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_dataset(data_root: Path, category: str) -> Path:
    category_root = data_root / category
    required_dirs = [
        category_root / "train" / "good",
        category_root / "test",
        category_root / "ground_truth",
    ]
    for folder in required_dirs:
        if not folder.exists():
            raise FileNotFoundError(f"Required directory not found: {folder.as_posix()}")
    for defect_type in DEFECT_TYPES:
        if not (category_root / "test" / defect_type).exists():
            raise FileNotFoundError(f"Test defect directory not found: {defect_type}")
        if not (category_root / "ground_truth" / defect_type).exists():
            raise FileNotFoundError(f"Ground truth defect directory not found: {defect_type}")
    if not (category_root / "test" / "good").exists():
        raise FileNotFoundError("Test good directory not found.")
    return category_root


def list_images(folder: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def collect_test_samples(category_root: Path) -> list[TestSample]:
    samples: list[TestSample] = []
    for defect_type in TEST_TYPES:
        image_paths = list_images(category_root / "test" / defect_type)
        if not image_paths:
            raise FileNotFoundError(f"No test images found for: {defect_type}")
        for image_path in image_paths:
            if defect_type == "good":
                samples.append(TestSample(image_path=image_path, mask_path=None, defect_type=defect_type, label=0))
                continue
            mask_path = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found for {defect_type}/{image_path.name}: {mask_path.as_posix()}")
            samples.append(TestSample(image_path=image_path, mask_path=mask_path, defect_type=defect_type, label=1))
    return samples


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_mask(sample: TestSample, image_size: int) -> Image.Image:
    if sample.mask_path is None:
        return Image.new("L", (image_size, image_size), 0)
    mask = Image.open(sample.mask_path).convert("L")
    mask = mask.point(lambda value: 255 if value > 0 else 0)
    return mask.resize((image_size, image_size), Image.Resampling.NEAREST)


def reset_output_dir(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)


def extract_patch_features(
    image_paths: list[Path],
    extractor: ResNetPatchFeatureExtractor,
    transform: transforms.Compose,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    patches: list[torch.Tensor] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = torch.stack([transform(read_rgb(path)) for path in batch_paths]).to(device)
        features = extractor(batch)
        features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu()
        patches.append(features)
    return torch.cat(patches, dim=0)


def subsample_memory_bank(memory_bank: torch.Tensor, max_patches: int, seed: int) -> torch.Tensor:
    if len(memory_bank) <= max_patches:
        return memory_bank
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(memory_bank), generator=generator)[:max_patches]
    return memory_bank[indices]


@torch.no_grad()
def score_image(
    image_path: Path,
    extractor: ResNetPatchFeatureExtractor,
    transform: transforms.Compose,
    memory_bank: torch.Tensor,
    device: str,
    image_size: int,
    distance_chunk_size: int,
) -> tuple[np.ndarray, float]:
    image_tensor = transform(read_rgb(image_path)).unsqueeze(0).to(device)
    features = extractor(image_tensor)
    _, _, feature_h, feature_w = features.shape
    patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])

    memory_bank = memory_bank.to(device)
    nearest_distances: list[torch.Tensor] = []
    for start in range(0, len(patches), distance_chunk_size):
        patch_chunk = patches[start : start + distance_chunk_size]
        distances = torch.cdist(patch_chunk, memory_bank)
        nearest_distances.append(distances.min(dim=1).values.cpu())

    scores = torch.cat(nearest_distances).reshape(feature_h, feature_w)
    score_map = F.interpolate(
        scores.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    score_map_np = score_map.numpy().astype(np.float32)
    return score_map_np, float(score_map_np.max())


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def save_heatmap(score_map: np.ndarray, output_path: Path) -> None:
    normalized = min_max_normalize(score_map)
    cmap = plt.get_cmap("jet")
    heatmap = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(heatmap).save(output_path)


def save_overlay(image_path: Path, score_map: np.ndarray, output_path: Path, alpha: float = 0.45) -> None:
    image = read_rgb(image_path).resize(score_map.shape[::-1], Image.Resampling.LANCZOS)
    image_arr = np.array(image).astype(np.float32)
    normalized = min_max_normalize(score_map)
    heatmap = (plt.get_cmap("jet")(normalized)[..., :3] * 255).astype(np.float32)
    overlay = (1.0 - alpha) * image_arr + alpha * heatmap
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    dice = f1
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
    }


def auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    positive_count = int(y_true.sum())
    negative_count = int(len(y_true) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[y_true == 1].sum())
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(auc)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    if y_true.sum() == 0:
        threshold = float(y_score.max()) if len(y_score) else 0.0
        return threshold, binary_metrics(y_true, y_score >= threshold)

    candidate_thresholds = np.quantile(y_score, np.linspace(0.01, 0.99, 99))
    best_threshold = float(candidate_thresholds[0])
    best_metrics = binary_metrics(y_true, y_score >= best_threshold)
    for threshold in candidate_thresholds:
        metrics = binary_metrics(y_true, y_score >= threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_preview(rows: list[dict[str, object]], output_path: Path) -> None:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for defect_type in TEST_TYPES:
        for row in rows:
            if row["defect_type"] == defect_type and defect_type not in seen:
                selected.append(row)
                seen.add(defect_type)
                break

    fig, axes = plt.subplots(len(selected), 4, figsize=(13, 3.2 * len(selected)))
    axes_array = np.array(axes).reshape(len(selected), 4)
    for row_index, row in enumerate(selected):
        image = read_rgb(Path(str(row["image_path"]))).resize((224, 224), Image.Resampling.LANCZOS)
        mask = Image.open(str(row["mask_preview"])).convert("L")
        heatmap = read_rgb(Path(str(row["anomaly_map"])))
        overlay = read_rgb(Path(str(row["overlay"])))
        panels = [image, np.array(mask), heatmap, overlay]
        titles = [f"{row['defect_type']} image", "gt mask", "anomaly map", "overlay"]
        for col_index, (panel, title) in enumerate(zip(panels, titles)):
            axes_array[row_index, col_index].imshow(panel, cmap="gray" if np.array(panel).ndim == 2 else None)
            axes_array[row_index, col_index].set_title(title, fontsize=9)
            axes_array[row_index, col_index].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary_md(metrics: dict[str, object], output_path: Path) -> None:
    lines = [
        "# Stage 4 PatchCore Baseline Summary",
        "",
        "This file is generated by scripts/04_patchcore_baseline.py.",
        "",
        "## Metrics",
        "",
        f"- Image AUROC: {metrics['image_auroc']:.4f}",
        f"- Pixel AUROC: {metrics['pixel_auroc']:.4f}",
        f"- Image F1: {metrics['image_f1']:.4f}",
        f"- Pixel F1: {metrics['pixel_f1']:.4f}",
        f"- Pixel Dice: {metrics['pixel_dice']:.4f}",
        f"- Pixel IoU: {metrics['pixel_iou']:.4f}",
        "",
        "## Notes",
        "",
        "- Training uses only train/good images.",
        "- Synthetic traditional and diffusion images are not used in this stage.",
        "- F1/Dice/IoU use a best-F1 diagnostic threshold on the current test set.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    category_root = validate_dataset(args.data_root, args.category)
    train_paths = list_images(category_root / "train" / "good")
    if not train_paths:
        raise FileNotFoundError("No train/good images found.")
    test_samples = collect_test_samples(category_root)

    output_root = args.output_dir / args.category
    if not args.keep_existing:
        reset_output_dir(output_root)
    anomaly_map_dir = output_root / "anomaly_maps"
    overlay_dir = output_root / "overlays"
    metadata_dir = output_root / "metadata"
    mask_preview_dir = output_root / "mask_previews"
    for folder in [anomaly_map_dir, overlay_dir, metadata_dir, mask_preview_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    transform = build_transform(args.image_size)
    extractor = ResNetPatchFeatureExtractor(args.device)
    memory_bank = extract_patch_features(train_paths, extractor, transform, args.device, args.batch_size)
    original_memory_count = len(memory_bank)
    memory_bank = subsample_memory_bank(memory_bank, args.max_memory_patches, args.seed)

    result_rows: list[dict[str, object]] = []
    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []

    for index, sample in enumerate(test_samples):
        score_map, image_score = score_image(
            image_path=sample.image_path,
            extractor=extractor,
            transform=transform,
            memory_bank=memory_bank,
            device=args.device,
            image_size=args.image_size,
            distance_chunk_size=args.distance_chunk_size,
        )
        mask = read_mask(sample, args.image_size)
        mask_arr = (np.array(mask) > 0).astype(np.uint8)

        sample_name = f"{args.category}_patchcore_{sample.defect_type}_{sample.image_path.stem}"
        anomaly_map_path = anomaly_map_dir / f"{sample_name}.png"
        overlay_path = overlay_dir / f"{sample_name}.png"
        mask_preview_path = mask_preview_dir / f"{sample_name}_mask.png"
        metadata_path = metadata_dir / f"{sample_name}.json"

        save_heatmap(score_map, anomaly_map_path)
        save_overlay(sample.image_path, score_map, overlay_path)
        mask.save(mask_preview_path)

        metadata = {
            "image_path": str(sample.image_path.resolve()),
            "mask_path": str(sample.mask_path.resolve()) if sample.mask_path else None,
            "defect_type": sample.defect_type,
            "label": sample.label,
            "image_score": image_score,
            "anomaly_map": str(anomaly_map_path.resolve()),
            "overlay": str(overlay_path.resolve()),
            "seed": args.seed,
            "backbone": "resnet18",
            "feature_layers": ["layer2", "layer3"],
            "image_size": args.image_size,
            "max_memory_patches": args.max_memory_patches,
            "memory_patches_before_subsample": original_memory_count,
            "memory_patches_used": len(memory_bank),
        }
        write_json(metadata, metadata_path)

        image_labels.append(sample.label)
        image_scores.append(image_score)
        pixel_labels.append(mask_arr.reshape(-1))
        pixel_scores.append(score_map.reshape(-1))

        result_rows.append(
            {
                "sample_name": sample_name,
                "defect_type": sample.defect_type,
                "label": sample.label,
                "image_path": str(sample.image_path.resolve()),
                "mask_path": str(sample.mask_path.resolve()) if sample.mask_path else "",
                "image_score": image_score,
                "anomaly_map": str(anomaly_map_path.resolve()),
                "overlay": str(overlay_path.resolve()),
                "mask_preview": str(mask_preview_path.resolve()),
                "metadata": str(metadata_path.resolve()),
            }
        )
        if (index + 1) % 10 == 0:
            print(f"Scored {index + 1}/{len(test_samples)} test images.")

    image_labels_np = np.array(image_labels, dtype=np.uint8)
    image_scores_np = np.array(image_scores, dtype=np.float64)
    pixel_labels_np = np.concatenate(pixel_labels).astype(np.uint8)
    pixel_scores_np = np.concatenate(pixel_scores).astype(np.float64)

    image_threshold, image_binary = best_f1_threshold(image_labels_np, image_scores_np)
    pixel_threshold, pixel_binary = best_f1_threshold(pixel_labels_np, pixel_scores_np)
    metrics = {
        "category": args.category,
        "method": "patchcore_style_resnet18",
        "train_images": len(train_paths),
        "test_images": len(test_samples),
        "image_size": args.image_size,
        "seed": args.seed,
        "device": args.device,
        "backbone": "resnet18",
        "feature_layers": ["layer2", "layer3"],
        "memory_patches_before_subsample": original_memory_count,
        "memory_patches_used": len(memory_bank),
        "image_auroc": auroc_score(image_labels_np, image_scores_np),
        "pixel_auroc": auroc_score(pixel_labels_np, pixel_scores_np),
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "threshold_method": "best_f1_on_test_for_stage_diagnostics",
        "image_precision": image_binary["precision"],
        "image_recall": image_binary["recall"],
        "image_f1": image_binary["f1"],
        "pixel_precision": pixel_binary["precision"],
        "pixel_recall": pixel_binary["recall"],
        "pixel_f1": pixel_binary["f1"],
        "pixel_dice": pixel_binary["dice"],
        "pixel_iou": pixel_binary["iou"],
    }

    write_csv(result_rows, output_root / "image_scores.csv")
    write_json(metrics, output_root / "metrics.json")
    write_summary_md(metrics, output_root / "summary.md")
    save_preview(result_rows, output_root / "preview.png")

    print("PatchCore baseline finished.")
    print(f"Output dir: {(Path('outputs') / 'baselines' / 'patchcore' / args.category).as_posix()}")
    print(f"Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"Pixel AUROC: {metrics['pixel_auroc']:.4f}")


if __name__ == "__main__":
    main()
