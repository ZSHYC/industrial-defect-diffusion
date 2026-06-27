from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


CATEGORY_DEFECT_TYPES = {
    "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
    "wood": ["color", "hole", "liquid", "scratch", "combined"],
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
}

NEGATIVE_PROMPT = "cartoon, painting, text, watermark, logo, unrealistic, oversaturated, blurry, distorted, object, people"
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_mask(path: Path) -> Image.Image:
    mask = Image.open(path).convert("L")
    return mask.point(lambda value: 255 if value > 0 else 0)


def resize_for_inpaint(image: Image.Image, mask: Image.Image, image_size: int) -> tuple[Image.Image, Image.Image]:
    return (
        image.resize((image_size, image_size), Image.Resampling.LANCZOS),
        mask.resize((image_size, image_size), Image.Resampling.NEAREST),
    )


def overlay_mask(image: Image.Image, mask: Image.Image, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    image_arr = np.array(image.convert("RGB")).astype(np.float32)
    mask_arr = np.array(mask.convert("L")) > 0
    image_arr[mask_arr] = (1.0 - alpha) * image_arr[mask_arr] + alpha * np.array(color, dtype=np.float32)
    return np.clip(image_arr, 0, 255).astype(np.uint8)


def write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_traditional_rows(summary_path: Path, samples_per_type: int, defect_types: list[str]) -> list[dict[str, str]]:
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    selected: list[dict[str, str]] = []
    for defect_type in defect_types:
        type_rows = [row for row in rows if row["defect_type"] == defect_type]
        if len(type_rows) < samples_per_type:
            raise ValueError(f"Need {samples_per_type} masks for {defect_type}, found {len(type_rows)}.")
        selected.extend(type_rows[:samples_per_type])
    return selected


def defect_types_for_category(category: str) -> list[str]:
    if category not in CATEGORY_DEFECT_TYPES:
        supported = ", ".join(sorted(CATEGORY_DEFECT_TYPES))
        raise ValueError(f"Unsupported category '{category}'. Supported categories: {supported}.")
    return CATEGORY_DEFECT_TYPES[category]


def save_preview(rows: list[dict[str, object]], output_path: Path, defect_types: list[str]) -> None:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        defect_type = str(row["defect_type"])
        if defect_type not in seen:
            selected.append(row)
            seen.add(defect_type)
        if len(selected) == len(defect_types):
            break

    fig, axes = plt.subplots(len(selected), 4, figsize=(13, 3.4 * len(selected)))
    axes_array = np.array(axes).reshape(len(selected), 4)

    for row_index, row in enumerate(selected):
        source = read_rgb(Path(str(row["source_image"])))
        mask = read_mask(Path(str(row["output_mask"])))
        diffusion = read_rgb(Path(str(row["output_image"])))
        overlay = overlay_mask(diffusion, mask)
        panels = [source, diffusion, np.array(mask), overlay]
        titles = [f"{row['defect_type']} source", "diffusion", "mask", "overlay"]
        for col_index, (panel, title) in enumerate(zip(panels, titles)):
            axes_array[row_index, col_index].imshow(panel, cmap="gray" if np.array(panel).ndim == 2 else None)
            axes_array[row_index, col_index].set_title(title, fontsize=10)
            axes_array[row_index, col_index].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_traditional_vs_diffusion_preview(rows: list[dict[str, object]], output_path: Path, defect_types: list[str]) -> None:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        defect_type = str(row["defect_type"])
        if defect_type not in seen:
            selected.append(row)
            seen.add(defect_type)
        if len(selected) == len(defect_types):
            break

    fig, axes = plt.subplots(len(selected), 5, figsize=(16, 3.3 * len(selected)))
    axes_array = np.array(axes).reshape(len(selected), 5)

    for row_index, row in enumerate(selected):
        source = read_rgb(Path(str(row["source_image"])))
        mask = read_mask(Path(str(row["output_mask"])))
        traditional = read_rgb(Path(str(row["traditional_image"])))
        diffusion = read_rgb(Path(str(row["output_image"])))
        panels = [source, np.array(mask), traditional, diffusion, overlay_mask(diffusion, mask)]
        titles = [f"{row['defect_type']} source", "mask", "traditional", "diffusion", "diffusion overlay"]
        for col_index, (panel, title) in enumerate(zip(panels, titles)):
            axes_array[row_index, col_index].imshow(panel, cmap="gray" if np.array(panel).ndim == 2 else None)
            axes_array[row_index, col_index].set_title(title, fontsize=9)
            axes_array[row_index, col_index].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_pipeline(
    model_id: str,
    device: str,
    variant: str | None,
    use_safetensors: bool,
    local_files_only: bool,
) -> StableDiffusionInpaintPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )
    except OSError as exc:
        raise RuntimeError(
            "Failed to load the inpainting model. Check the model id, Hugging Face access, "
            "network connection, or local cache. Current model id: "
            f"{model_id}"
        ) from exc
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=False)
    return pipe


def reset_output_dir(output_root: Path) -> None:
    for name in ["images", "masks", "metadata"]:
        folder = output_root / name
        if folder.exists():
            shutil.rmtree(folder)
    for filename in ["preview.png", "traditional_vs_diffusion_preview.png", "summary.csv"]:
        path = output_root / filename
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diffusion inpainting defects using traditional masks.")
    parser.add_argument("--category", default="tile")
    parser.add_argument("--samples-per-type", type=int, default=3)
    parser.add_argument(
        "--defect-types",
        nargs="+",
        default=None,
        help="Defect types to generate. Defaults to all configured defects for the category.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--no-safetensors", action="store_true", help="Allow loading non-safetensors model files.")
    parser.add_argument("--local-files-only", action="store_true", help="Only use already downloaded model files.")
    parser.add_argument(
        "--conditioning-image",
        choices=["source", "traditional"],
        default="traditional",
        help="Use the clean source image or the traditional synthetic image as the inpainting condition.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear previous diffusion outputs.")
    parser.add_argument(
        "--traditional-summary",
        type=Path,
        default=Path("outputs") / "traditional_synthetic" / "tile" / "summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "diffusion_synthetic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configured_defect_types = defect_types_for_category(args.category)
    if args.defect_types is None:
        args.defect_types = configured_defect_types
    else:
        unknown = sorted(set(args.defect_types) - set(configured_defect_types))
        if unknown:
            raise ValueError(f"Unsupported defect types for {args.category}: {', '.join(unknown)}")
    prompts = CATEGORY_PROMPTS[args.category]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if not 0.0 < args.strength <= 1.0:
        raise ValueError("--strength must be in (0, 1].")
    if int(args.num_inference_steps * args.strength) < 1:
        raise ValueError("num_inference_steps * strength must be at least 1.")
    if not args.traditional_summary.exists():
        raise FileNotFoundError(f"Traditional summary not found: {args.traditional_summary.as_posix()}")

    output_root = args.output_dir / args.category
    if not args.keep_existing:
        reset_output_dir(output_root)

    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    metadata_dir = output_root / "metadata"
    for folder in [image_dir, mask_dir, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    selected_rows = load_traditional_rows(args.traditional_summary, args.samples_per_type, args.defect_types)
    pipe = build_pipeline(
        model_id=args.model_id,
        device=args.device,
        variant=args.variant,
        use_safetensors=not args.no_safetensors,
        local_files_only=args.local_files_only,
    )

    result_rows: list[dict[str, object]] = []
    for global_index, row in enumerate(selected_rows):
        defect_type = row["defect_type"]
        prompt = prompts[defect_type]
        source_path = Path(row["source_image"])
        mask_path = Path(row["output_mask"])
        traditional_image_path = Path(row["output_image"])

        source_image = read_rgb(source_path)
        traditional_image = read_rgb(traditional_image_path)
        mask = read_mask(mask_path)
        original_size = source_image.size
        conditioning_image = traditional_image if args.conditioning_image == "traditional" else source_image
        resized_conditioning, resized_mask = resize_for_inpaint(conditioning_image, mask, args.image_size)

        generator = torch.Generator(device=args.device).manual_seed(args.seed + global_index)
        output = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=resized_conditioning,
            mask_image=resized_mask,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            generator=generator,
        ).images[0]

        output = output.resize(original_size, Image.Resampling.LANCZOS)
        sample_name = f"{args.category}_diff_{defect_type}_{global_index:03d}"
        output_image_path = image_dir / f"{sample_name}.png"
        output_mask_path = mask_dir / f"{sample_name}_mask.png"
        metadata_path = metadata_dir / f"{sample_name}.json"

        output.save(output_image_path)
        mask.save(output_mask_path)

        metadata = {
            "source_image": str(source_path.resolve()),
            "source_mask": str(mask_path.resolve()),
            "traditional_image": str(traditional_image_path.resolve()),
            "conditioning_image": args.conditioning_image,
            "defect_type": defect_type,
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "model_id": args.model_id,
            "variant": args.variant,
            "use_safetensors": not args.no_safetensors,
            "seed": args.seed + global_index,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "image_size": args.image_size,
            "device": args.device,
            "output_image": str(output_image_path.resolve()),
            "output_mask": str(output_mask_path.resolve()),
        }
        write_json(metadata, metadata_path)

        result_rows.append({
            "sample_name": sample_name,
            "defect_type": defect_type,
            "source_image": str(source_path.resolve()),
            "source_mask": str(mask_path.resolve()),
            "traditional_image": str(traditional_image_path.resolve()),
            "conditioning_image": args.conditioning_image,
            "output_image": str(output_image_path.resolve()),
            "output_mask": str(output_mask_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "seed": args.seed + global_index,
            "model_id": args.model_id,
            "variant": args.variant,
            "prompt": prompt,
        })

    write_csv(result_rows, output_root / "summary.csv")
    save_preview(result_rows, output_root / "preview.png", args.defect_types)
    save_traditional_vs_diffusion_preview(result_rows, output_root / "traditional_vs_diffusion_preview.png", args.defect_types)

    print(f"Generated {len(result_rows)} diffusion synthetic defects.")
    print(f"Output dir: {output_root.as_posix()}")
    for defect_type in args.defect_types:
        count = sum(1 for row in result_rows if row["defect_type"] == defect_type)
        print(f"  {defect_type}: {count}")


if __name__ == "__main__":
    main()
