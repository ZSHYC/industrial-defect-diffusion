from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_defect.paths import resolve_data_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stage 8 wood generalization experiment.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to the MVTec AD dataset root. Defaults to DATA_ROOT.",
    )
    parser.add_argument("--category", default="wood")
    parser.add_argument("--samples-per-type", type=int, default=20)
    parser.add_argument("--diffusion-samples-per-type", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=304)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-diffusion", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("\n>>> " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    args.data_root = resolve_data_root(args.data_root)
    python = sys.executable
    category = args.category
    traditional_summary = Path("outputs") / "stage8_wood_synthetic" / "traditional" / category / "summary.csv"
    diffusion_summary = Path("outputs") / "stage8_wood_synthetic" / "diffusion" / category / "summary.csv"
    accepted_traditional = Path("outputs") / "stage8_wood_quality_filter" / category / "accepted_traditional_summary.csv"
    accepted_diffusion = Path("outputs") / "stage8_wood_quality_filter" / category / "accepted_diffusion_summary.csv"

    run_step([
        python,
        "scripts/01_explore_dataset.py",
        "--data-root",
        str(args.data_root),
        "--category",
        category,
        "--output-dir",
        "outputs/eda",
    ])
    run_step([
        python,
        "scripts/02_generate_traditional_defects.py",
        "--data-root",
        str(args.data_root),
        "--category",
        category,
        "--samples-per-type",
        str(args.samples_per_type),
        "--seed",
        str(args.seed),
        "--output-dir",
        "outputs/stage8_wood_synthetic/traditional",
    ])

    if not args.skip_diffusion:
        diffusion_command = [
            python,
            "scripts/03_generate_diffusion_defects.py",
            "--category",
            category,
            "--traditional-summary",
            str(traditional_summary),
            "--samples-per-type",
            str(args.diffusion_samples_per_type),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--seed",
            str(args.seed),
            "--output-dir",
            "outputs/stage8_wood_synthetic/diffusion",
        ]
        if args.local_files_only:
            diffusion_command.append("--local-files-only")
        run_step(diffusion_command)

    run_step([
        python,
        "scripts/06_filter_synthetic_quality.py",
        "--traditional-summary",
        str(traditional_summary),
        "--diffusion-summary",
        str(diffusion_summary),
        "--output-dir",
        str(Path("outputs") / "stage8_wood_quality_filter" / category),
    ])

    if not args.skip_training:
        run_step([
            python,
            "scripts/05_train_unet_segmentation.py",
            "--data-root",
            str(args.data_root),
            "--category",
            category,
            "--image-size",
            str(args.image_size),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--traditional-summary",
            str(accepted_traditional),
            "--diffusion-summary",
            str(accepted_diffusion),
            "--output-dir",
            "outputs/training/unet_segmentation_stage8_wood",
            "--experiments",
            "combined",
        ])


if __name__ == "__main__":
    main()
