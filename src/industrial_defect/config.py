from __future__ import annotations

import json
from dataclasses import dataclass

from .paths import config_path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

NEGATIVE_PROMPT = "cartoon, painting, text, watermark, logo, unrealistic, oversaturated, blurry, distorted, object, people"
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"


@dataclass(frozen=True)
class CategoryConfig:
    name: str
    defect_types: list[str]
    evaluation_defect_types: list[str]
    prompts: dict[str, str]


def load_categories() -> dict[str, dict[str, object]]:
    return json.loads(config_path("categories.json").read_text(encoding="utf-8"))


CATEGORIES = load_categories()
CATEGORY_DEFECT_TYPES = {
    category: list(config["defect_types"])
    for category, config in CATEGORIES.items()
}
EVALUATION_DEFECT_TYPES = {
    category: list(config["evaluation_defect_types"])
    for category, config in CATEGORIES.items()
}
CATEGORY_PROMPTS = {
    category: dict(config["prompts"])
    for category, config in CATEGORIES.items()
}


def supported_categories() -> list[str]:
    return sorted(CATEGORY_DEFECT_TYPES)


def get_category_config(category: str) -> CategoryConfig:
    if category not in CATEGORY_DEFECT_TYPES:
        supported = ", ".join(supported_categories())
        raise ValueError(f"Unsupported category '{category}'. Supported categories: {supported}.")
    return CategoryConfig(
        name=category,
        defect_types=list(CATEGORY_DEFECT_TYPES[category]),
        evaluation_defect_types=list(EVALUATION_DEFECT_TYPES[category]),
        prompts=dict(CATEGORY_PROMPTS[category]),
    )


def defect_types_for_category(category: str, *, evaluation_order: bool = False) -> list[str]:
    config = get_category_config(category)
    return config.evaluation_defect_types if evaluation_order else config.defect_types
