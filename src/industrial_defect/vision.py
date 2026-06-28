from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .models import LightUNet


def load_rgb_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def probability_to_image(prob_map: np.ndarray) -> Image.Image:
    return Image.fromarray((prob_map * 255).clip(0, 255).astype(np.uint8))


def binary_mask_from_probability(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prob_map >= threshold).astype(np.uint8)


def save_probability_mask(prob_map: np.ndarray, output_path: Path, *, size: tuple[int, int] | None = None) -> None:
    image = probability_to_image(prob_map)
    if size is not None:
        image = image.resize(size, Image.Resampling.BILINEAR)
    image.save(output_path)


def save_binary_mask(mask: np.ndarray, output_path: Path, *, size: tuple[int, int] | None = None) -> None:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    if size is not None:
        image = image.resize(size, Image.Resampling.NEAREST)
    image.save(output_path)


def save_overlay(
    image_path: Path,
    pred_mask: np.ndarray,
    output_path: Path,
    *,
    image_size: int | None = None,
    output_size: tuple[int, int] | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> None:
    image = load_rgb_image(image_path)
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    if output_size is not None:
        image = image.resize(output_size, Image.Resampling.LANCZOS)
        mask_image = Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(output_size, Image.Resampling.NEAREST)
        mask = np.array(mask_image) > 0
    else:
        mask = pred_mask.astype(bool)
    image_arr = np.array(image).astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    image_arr[mask] = (1.0 - alpha) * image_arr[mask] + alpha * color_arr
    Image.fromarray(np.clip(image_arr, 0, 255).astype(np.uint8)).save(output_path)


@torch.no_grad()
def predict_probability(model: LightUNet, image_path: Path, image_size: int, device: str) -> np.ndarray:
    transform = transforms.Compose([transforms.Resize((image_size, image_size), antialias=True), transforms.ToTensor()])
    image = load_rgb_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy().astype(np.float32)
    return prob
