from __future__ import annotations

from dataclasses import dataclass


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CATEGORY_DEFECT_TYPES = {
    "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
    "wood": ["color", "hole", "liquid", "scratch", "combined"],
    "leather": ["color", "cut", "fold", "glue", "poke"],
}

EVALUATION_DEFECT_TYPES = {
    "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
    "wood": ["color", "combined", "hole", "liquid", "scratch"],
    "leather": ["color", "cut", "fold", "glue", "poke"],
}

CATEGORY_PROMPTS = {
    "tile": {
        "crack": "a realistic long branching dark hairline crack fracture across industrial ceramic tile surface, thin irregular fracture lines, inspection image, natural texture",
        "glue_strip": "a realistic pale translucent glue strip defect on industrial ceramic tile surface, inspection image, natural texture",
        "gray_stroke": "a realistic dark gray black irregular smudge stain defect on industrial ceramic tile surface, rough local contamination, inspection image, visible defect region",
        "oil": "a realistic yellow brown translucent oil stain defect on industrial ceramic tile surface, inspection image, visible contamination",
        "rough": "a realistic rough damaged texture defect on industrial ceramic tile surface, inspection image, visible local abnormal area",
    },
    "wood": {
        "color": "a realistic dark color stain defect on industrial wood surface, natural grain texture, inspection image, visible local discoloration",
        "hole": "a realistic dark irregular hole defect on industrial wood surface, damaged wood grain, inspection image, visible local defect",
        "liquid": "a realistic translucent liquid stain defect on industrial wood surface, natural grain texture, inspection image, visible wet contamination",
        "scratch": "realistic long bright scratch marks and large shallow abrasion on industrial wood surface, follows natural wood grain texture, inspection image, visible wide scratched defect",
        "combined": "realistic combined defects on industrial wood surface, local stain and scratch damage, natural grain texture, inspection image",
    },
    "leather": {
        "color": "a realistic dark color stain defect on industrial leather surface, fine leather grain texture, inspection image, visible local defect",
        "cut": "a realistic narrow cut defect on industrial leather surface, fine leather grain texture, inspection image, visible local defect",
        "fold": "a realistic shallow elongated fold crease defect on industrial leather surface, subtle raised ridge and shadow, follows fine leather grain texture, inspection image, visible local defect",
        "glue": "a realistic pale translucent glue smear defect on industrial leather surface, fine leather grain texture, inspection image, visible local defect",
        "poke": "a realistic small poke hole defect on industrial leather surface, fine leather grain texture, inspection image, visible local defect",
    },
}

NEGATIVE_PROMPT = "cartoon, painting, text, watermark, logo, unrealistic, oversaturated, blurry, distorted, object, people"
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"


@dataclass(frozen=True)
class CategoryConfig:
    name: str
    defect_types: list[str]
    evaluation_defect_types: list[str]
    prompts: dict[str, str]


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

