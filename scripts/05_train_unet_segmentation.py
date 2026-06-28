from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_defect.config import EVALUATION_DEFECT_TYPES as CATEGORY_DEFECT_TYPES  # noqa: E402

EXPERIMENTS = ["traditional", "diffusion", "combined"]
EXPERIMENT_SEED_OFFSETS = {
    "traditional": 0,
    "diffusion": 1000,
    "combined": 2000,
}


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path | None
    defect_type: str
    label: int
    source: str


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, samples: list[SegmentationSample], image_size: int) -> None:
        self.samples = samples
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if sample.mask_path is None:
            mask = Image.new("L", image.size, 0)
        else:
            mask = Image.open(sample.mask_path).convert("L")
            mask = mask.point(lambda value: 255 if value > 0 else 0)
        return self.image_transform(image), self.mask_transform(mask)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        b = self.bottleneck(self.pool(c3))
        x = self.up3(b)
        x = self.conv3(torch.cat([x, c3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, c2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, c1], dim=1))
        return self.out(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight U-Net segmentation models on synthetic defects.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", default="tile")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--traditional-summary", type=Path, default=Path("outputs") / "traditional_synthetic" / "tile" / "summary.csv")
    parser.add_argument("--diffusion-summary", type=Path, default=Path("outputs") / "diffusion_synthetic" / "tile" / "summary.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "training" / "unet_segmentation")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear previous U-Net outputs.")
    parser.add_argument(
        "--good-negative-samples",
        type=int,
        default=0,
        help="Number of real train/good images to add as empty-mask negative training samples.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=EXPERIMENTS,
        help="Subset of experiments to run.",
    )
    return parser.parse_args()


def defect_types_for_category(category: str) -> list[str]:
    if category not in CATEGORY_DEFECT_TYPES:
        supported = ", ".join(sorted(CATEGORY_DEFECT_TYPES))
        raise ValueError(f"Unsupported category '{category}'. Supported categories: {supported}.")
    return CATEGORY_DEFECT_TYPES[category]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_dataset(data_root: Path, category: str, defect_types: list[str]) -> Path:
    category_root = data_root / category
    for folder in [category_root / "test", category_root / "ground_truth"]:
        if not folder.exists():
            raise FileNotFoundError(f"Required directory not found: {folder.as_posix()}")
    for defect_type in ["good", *defect_types]:
        if not (category_root / "test" / defect_type).exists():
            raise FileNotFoundError(f"Test directory not found: {defect_type}")
    for defect_type in defect_types:
        if not (category_root / "ground_truth" / defect_type).exists():
            raise FileNotFoundError(f"Ground truth directory not found: {defect_type}")
    return category_root


def list_images(folder: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def load_synthetic_samples(summary_path: Path, source: str) -> list[SegmentationSample]:
    if not summary_path.exists():
        raise FileNotFoundError(f"{source} summary not found: {summary_path.as_posix()}")
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    samples: list[SegmentationSample] = []
    for row in rows:
        image_path = Path(row["output_image"])
        mask_path = Path(row["output_mask"])
        if not image_path.exists():
            raise FileNotFoundError(f"{source} image not found: {image_path.as_posix()}")
        if not mask_path.exists():
            raise FileNotFoundError(f"{source} mask not found: {mask_path.as_posix()}")
        samples.append(
            SegmentationSample(
                image_path=image_path,
                mask_path=mask_path,
                defect_type=row["defect_type"],
                label=1,
                source=source,
            )
        )
    return samples


def collect_good_negative_samples(category_root: Path, count: int, seed: int) -> list[SegmentationSample]:
    if count <= 0:
        return []
    good_images = list_images(category_root / "train" / "good")
    if not good_images:
        raise FileNotFoundError(f"No train/good images found in: {(category_root / 'train' / 'good').as_posix()}")
    rng = np.random.default_rng(seed)
    if count <= len(good_images):
        indices = rng.choice(len(good_images), size=count, replace=False)
    else:
        indices = rng.choice(len(good_images), size=count, replace=True)
    return [
        SegmentationSample(good_images[int(index)], None, "good_negative", 0, "real_train_good")
        for index in indices
    ]


def collect_test_samples(category_root: Path, defect_types: list[str]) -> list[SegmentationSample]:
    samples: list[SegmentationSample] = []
    for defect_type in ["good", *defect_types]:
        for image_path in list_images(category_root / "test" / defect_type):
            if defect_type == "good":
                samples.append(SegmentationSample(image_path, None, defect_type, 0, "real_test"))
            else:
                mask_path = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask not found for {defect_type}/{image_path.name}: {mask_path.as_posix()}")
                samples.append(SegmentationSample(image_path, mask_path, defect_type, 1, "real_test"))
    return samples


def reset_output_dir(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


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
    return {"precision": precision, "recall": recall, "f1": f1, "dice": f1, "iou": iou}


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, y_score >= best_threshold)
    for threshold in thresholds:
        metrics = binary_metrics(y_true, y_score >= threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in np.arange(0.05, 0.951, 0.01):
        metrics = binary_metrics(y_true, y_score >= float(threshold))
        rows.append({
            "threshold": float(threshold),
            "pixel_precision": metrics["precision"],
            "pixel_recall": metrics["recall"],
            "pixel_f1": metrics["f1"],
            "pixel_iou": metrics["iou"],
        })
    return rows


def remove_small_components(mask: np.ndarray, min_area_pixels: int) -> np.ndarray:
    if min_area_pixels <= 0:
        return mask.astype(np.uint8)
    mask_bool = mask.astype(bool)
    height, width = mask_bool.shape
    visited = np.zeros_like(mask_bool, dtype=bool)
    output = np.zeros_like(mask_bool, dtype=bool)
    for y in range(height):
        for x in range(width):
            if not mask_bool[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask_bool[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(component) >= min_area_pixels:
                ys, xs = zip(*component)
                output[np.array(ys), np.array(xs)] = True
    return output.astype(np.uint8)


def postprocess_sweep(
    y_true_images: list[np.ndarray],
    y_prob_images: list[np.ndarray],
    thresholds: list[float],
    min_area_ratios: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    image_area = int(y_prob_images[0].size) if y_prob_images else 0
    labels_flat = np.concatenate([mask.reshape(-1) for mask in y_true_images])
    for threshold in thresholds:
        for min_area_ratio in min_area_ratios:
            min_area_pixels = int(round(min_area_ratio * image_area))
            preds = [
                remove_small_components((prob >= threshold).astype(np.uint8), min_area_pixels).reshape(-1)
                for prob in y_prob_images
            ]
            pred_flat = np.concatenate(preds)
            metrics = binary_metrics(labels_flat, pred_flat)
            rows.append({
                "threshold": float(threshold),
                "min_component_area_ratio": float(min_area_ratio),
                "min_component_area_pixels": min_area_pixels,
                "pixel_precision": metrics["precision"],
                "pixel_recall": metrics["recall"],
                "pixel_f1": metrics["f1"],
                "pixel_iou": metrics["iou"],
            })
    return rows


def read_mask(sample: SegmentationSample, image_size: int) -> Image.Image:
    if sample.mask_path is None:
        return Image.new("L", (image_size, image_size), 0)
    mask = Image.open(sample.mask_path).convert("L")
    mask = mask.point(lambda value: 255 if value > 0 else 0)
    return mask.resize((image_size, image_size), Image.Resampling.NEAREST)


def save_prediction_mask(prob_map: np.ndarray, output_path: Path, threshold: float = 0.5) -> None:
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(output_path)


def save_overlay(image_path: Path, pred_mask: np.ndarray, output_path: Path, image_size: int) -> None:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    image_arr = np.array(image).astype(np.float32)
    mask = pred_mask.astype(bool)
    color = np.array([255, 0, 0], dtype=np.float32)
    image_arr[mask] = 0.55 * image_arr[mask] + 0.45 * color
    Image.fromarray(np.clip(image_arr, 0, 255).astype(np.uint8)).save(output_path)


def write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_model(
    experiment: str,
    samples: list[SegmentationSample],
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[LightUNet, list[float]]:
    experiment_seed = args.seed + EXPERIMENT_SEED_OFFSETS[experiment]
    seed_everything(experiment_seed)
    dataset = SyntheticSegmentationDataset(samples, args.image_size)
    generator = torch.Generator().manual_seed(experiment_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = LightUNet().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    best_loss = float("inf")
    checkpoint_dir = output_root / experiment / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_losses: list[float] = []
        for images, masks in loader:
            images = images.to(args.device)
            masks = masks.to(args.device)
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        print(f"{experiment} epoch {epoch + 1}/{args.epochs} loss={mean_loss:.4f}")

    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=args.device))
    return model, losses


def save_loss_curve(losses: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def predict_probability(model: LightUNet, image_path: Path, image_size: int, device: str) -> np.ndarray:
    transform = transforms.Compose([transforms.Resize((image_size, image_size), antialias=True), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy().astype(np.float32)
    return prob


def evaluate_model(
    experiment: str,
    model: LightUNet,
    test_samples: list[SegmentationSample],
    train_sample_count: int,
    args: argparse.Namespace,
    output_root: Path,
    defect_types: list[str],
) -> dict[str, object]:
    experiment_root = output_root / experiment
    prediction_dir = experiment_root / "predictions"
    overlay_dir = experiment_root / "overlays"
    metadata_dir = experiment_root / "metadata"
    for folder in [prediction_dir, overlay_dir, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    pixel_labels: list[np.ndarray] = []
    pixel_probs: list[np.ndarray] = []
    pixel_label_images: list[np.ndarray] = []
    pixel_prob_images: list[np.ndarray] = []
    image_labels: list[int] = []
    image_scores: list[float] = []

    for sample in test_samples:
        prob = predict_probability(model, sample.image_path, args.image_size, args.device)
        gt_mask = read_mask(sample, args.image_size)
        gt_arr = (np.array(gt_mask) > 0).astype(np.uint8)
        pred_arr = (prob >= 0.5).astype(np.uint8)
        sample_name = f"{args.category}_{experiment}_{sample.defect_type}_{sample.image_path.stem}"
        prediction_path = prediction_dir / f"{sample_name}_pred.png"
        overlay_path = overlay_dir / f"{sample_name}_overlay.png"
        metadata_path = metadata_dir / f"{sample_name}.json"

        save_prediction_mask(prob, prediction_path, threshold=0.5)
        save_overlay(sample.image_path, pred_arr, overlay_path, args.image_size)
        per_image = binary_metrics(gt_arr.reshape(-1), pred_arr.reshape(-1))
        image_score = float(prob.max())

        metadata = {
            "image_path": str(sample.image_path.resolve()),
            "mask_path": str(sample.mask_path.resolve()) if sample.mask_path else None,
            "defect_type": sample.defect_type,
            "label": sample.label,
            "image_score": image_score,
            "prediction_path": str(prediction_path.resolve()),
            "overlay_path": str(overlay_path.resolve()),
            "dice": per_image["dice"],
            "iou": per_image["iou"],
            "precision": per_image["precision"],
            "recall": per_image["recall"],
        }
        write_json(metadata, metadata_path)

        rows.append(
            {
                "experiment": experiment,
                "image_path": str(sample.image_path.resolve()),
                "defect_type": sample.defect_type,
                "label": sample.label,
                "mask_path": str(sample.mask_path.resolve()) if sample.mask_path else "",
                "prediction_path": str(prediction_path.resolve()),
                "overlay_path": str(overlay_path.resolve()),
                "image_score": image_score,
                "dice": per_image["dice"],
                "iou": per_image["iou"],
                "precision": per_image["precision"],
                "recall": per_image["recall"],
            }
        )
        pixel_labels.append(gt_arr.reshape(-1))
        pixel_probs.append(prob.reshape(-1))
        pixel_label_images.append(gt_arr)
        pixel_prob_images.append(prob)
        image_labels.append(sample.label)
        image_scores.append(image_score)

    pixel_labels_np = np.concatenate(pixel_labels)
    pixel_probs_np = np.concatenate(pixel_probs)
    pixel_pred_05 = (pixel_probs_np >= 0.5).astype(np.uint8)
    image_labels_np = np.array(image_labels, dtype=np.uint8)
    image_scores_np = np.array(image_scores, dtype=np.float32)
    image_pred_05 = (image_scores_np >= 0.5).astype(np.uint8)
    pixel_05 = binary_metrics(pixel_labels_np, pixel_pred_05)
    image_05 = binary_metrics(image_labels_np, image_pred_05)
    best_threshold, best_pixel = best_f1_threshold(pixel_labels_np, pixel_probs_np)
    threshold_rows = threshold_sweep(pixel_labels_np, pixel_probs_np)
    postprocess_rows = postprocess_sweep(
        pixel_label_images,
        pixel_prob_images,
        thresholds=[0.5, best_threshold],
        min_area_ratios=[0.0, 0.0002, 0.0005, 0.001, 0.002],
    )
    best_threshold_row = max(threshold_rows, key=lambda row: float(row["pixel_f1"]))
    best_postprocess_row = max(postprocess_rows, key=lambda row: float(row["pixel_f1"]))

    class_metrics: dict[str, dict[str, float]] = {}
    for defect_type in ["good", *defect_types]:
        class_rows = [row for row in rows if row["defect_type"] == defect_type]
        class_metrics[defect_type] = {
            "count": float(len(class_rows)),
            "mean_dice": float(np.mean([float(row["dice"]) for row in class_rows])) if class_rows else 0.0,
            "mean_iou": float(np.mean([float(row["iou"]) for row in class_rows])) if class_rows else 0.0,
            "mean_recall": float(np.mean([float(row["recall"]) for row in class_rows])) if class_rows else 0.0,
        }

    metrics: dict[str, object] = {
        "experiment": experiment,
        "train_samples": train_sample_count,
        "synthetic_train_samples": train_sample_count - args.good_negative_samples,
        "good_negative_samples": args.good_negative_samples,
        "test_samples": len(test_samples),
        "image_size": args.image_size,
        "threshold": 0.5,
        "best_f1_diagnostic_threshold": best_threshold,
        "pixel_precision": pixel_05["precision"],
        "pixel_recall": pixel_05["recall"],
        "pixel_f1": pixel_05["f1"],
        "pixel_dice": pixel_05["dice"],
        "pixel_iou": pixel_05["iou"],
        "best_pixel_precision": best_pixel["precision"],
        "best_pixel_recall": best_pixel["recall"],
        "best_pixel_f1": best_pixel["f1"],
        "best_pixel_iou": best_pixel["iou"],
        "image_precision": image_05["precision"],
        "image_recall": image_05["recall"],
        "image_f1": image_05["f1"],
        "threshold_sweep_best": best_threshold_row,
        "postprocess_sweep_best": best_postprocess_row,
        "class_metrics": class_metrics,
    }
    write_csv(rows, experiment_root / "test_predictions.csv")
    write_csv(threshold_rows, experiment_root / "threshold_sweep.csv")
    write_csv(postprocess_rows, experiment_root / "postprocess_sweep.csv")
    write_json(metrics, experiment_root / "metrics.json")
    return metrics


def save_preview(experiment: str, test_rows_path: Path, output_path: Path, image_size: int, defect_types: list[str]) -> None:
    rows: list[dict[str, str]]
    with test_rows_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    selected: list[dict[str, str]] = []
    for defect_type in ["good", *defect_types]:
        for row in rows:
            if row["defect_type"] == defect_type:
                selected.append(row)
                break

    fig, axes = plt.subplots(len(selected), 4, figsize=(13, 3.2 * len(selected)))
    axes_array = np.array(axes).reshape(len(selected), 4)
    for row_index, row in enumerate(selected):
        image = Image.open(row["image_path"]).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
        if row["mask_path"]:
            gt_mask = read_mask(
                SegmentationSample(Path(row["image_path"]), Path(row["mask_path"]), row["defect_type"], int(row["label"]), "real_test"),
                image_size,
            )
        else:
            gt_mask = Image.new("L", (image_size, image_size), 0)
        pred = Image.open(row["prediction_path"]).convert("L")
        overlay = Image.open(row["overlay_path"]).convert("RGB")
        panels = [image, np.array(gt_mask), np.array(pred), overlay]
        titles = [f"{row['defect_type']} image", "gt mask", "pred mask", "overlay"]
        for col_index, (panel, title) in enumerate(zip(panels, titles)):
            axes_array[row_index, col_index].imshow(panel, cmap="gray" if np.array(panel).ndim == 2 else None)
            axes_array[row_index, col_index].set_title(title, fontsize=9)
            axes_array[row_index, col_index].axis("off")
    fig.suptitle(experiment, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_comparison_preview(output_root: Path, output_path: Path, experiments: list[str]) -> None:
    previews = []
    for experiment in experiments:
        preview_path = output_root / experiment / "preview.png"
        if preview_path.exists():
            previews.append((experiment, Image.open(preview_path).convert("RGB")))
    if not previews:
        return
    max_width = max(image.width for _, image in previews)
    total_height = sum(image.height for _, image in previews)
    canvas = Image.new("RGB", (max_width, total_height), "white")
    y_offset = 0
    for _, image in previews:
        canvas.paste(image, (0, y_offset))
        y_offset += image.height
    canvas.save(output_path)


def write_comparison_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    write_csv(rows, output_path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    defect_types = defect_types_for_category(args.category)
    category_root = validate_dataset(args.data_root, args.category, defect_types)
    traditional_samples = load_synthetic_samples(args.traditional_summary, "traditional")
    diffusion_samples = load_synthetic_samples(args.diffusion_summary, "diffusion")
    good_negative_samples = collect_good_negative_samples(category_root, args.good_negative_samples, args.seed)
    test_samples = collect_test_samples(category_root, defect_types)

    output_root = args.output_dir / args.category
    if not args.keep_existing:
        reset_output_dir(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_samples = {
        "traditional": traditional_samples + good_negative_samples,
        "diffusion": diffusion_samples + good_negative_samples,
        "combined": traditional_samples + diffusion_samples + good_negative_samples,
    }
    comparison_rows: list[dict[str, object]] = []
    for experiment in args.experiments:
        samples = experiment_samples[experiment]
        experiment_root = output_root / experiment
        experiment_root.mkdir(parents=True, exist_ok=True)
        model, losses = train_model(experiment, samples, args, output_root)
        save_loss_curve(losses, experiment_root / "loss_curve.png")
        metrics = evaluate_model(experiment, model, test_samples, len(samples), args, output_root, defect_types)
        save_preview(experiment, experiment_root / "test_predictions.csv", experiment_root / "preview.png", args.image_size, defect_types)
        comparison_rows.append(
            {
                "experiment": experiment,
                "train_samples": len(samples),
                "test_samples": len(test_samples),
                "pixel_precision": metrics["pixel_precision"],
                "pixel_recall": metrics["pixel_recall"],
                "pixel_f1": metrics["pixel_f1"],
                "pixel_dice": metrics["pixel_dice"],
                "pixel_iou": metrics["pixel_iou"],
                "best_pixel_f1": metrics["best_pixel_f1"],
                "image_precision": metrics["image_precision"],
                "image_recall": metrics["image_recall"],
                "image_f1": metrics["image_f1"],
            }
        )

    write_comparison_summary(comparison_rows, output_root / "comparison_summary.csv")
    save_comparison_preview(output_root, output_root / "comparison_preview.png", args.experiments)

    print("U-Net segmentation experiments finished.")
    print(f"Output dir: {output_root.as_posix()}")
    for row in comparison_rows:
        print(
            f"{row['experiment']}: pixel_f1={float(row['pixel_f1']):.4f}, "
            f"pixel_iou={float(row['pixel_iou']):.4f}, image_f1={float(row['image_f1']):.4f}"
        )


if __name__ == "__main__":
    main()
