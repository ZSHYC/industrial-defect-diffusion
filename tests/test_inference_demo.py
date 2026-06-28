import importlib.util
from pathlib import Path

import pytest


def load_script_module():
    path = Path("scripts") / "18_inference_demo.py"
    spec = importlib.util.spec_from_file_location("inference_demo", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dry_run_writes_demo_requirements(tmp_path, monkeypatch) -> None:
    module = load_script_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    args = module.parse_args(["--category", "leather", "--dry-run", "--output-dir", str(tmp_path)])
    module.write_dry_run_outputs(args)
    report = tmp_path / "demo_requirements.md"
    readiness = tmp_path / "outputs" / "final_report" / "deployment_readiness.md"
    assert report.exists()
    assert readiness.exists()
    text = report.read_text(encoding="utf-8")
    assert "Inference Demo Dry Run" in text
    assert "leather" in text
    assert "path/to/best_model.pt" in text


def test_missing_checkpoint_requires_local_model(tmp_path) -> None:
    module = load_script_module()
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not a real image but validation only checks existence")
    args = module.parse_args([
        "--category",
        "tile",
        "--image",
        str(image_path),
        "--checkpoint",
        str(tmp_path / "missing.pt"),
    ])
    with pytest.raises(FileNotFoundError, match="intentionally not committed"):
        module.validate_args(args)


def test_threshold_must_be_between_zero_and_one() -> None:
    module = load_script_module()
    args = module.parse_args(["--category", "wood", "--dry-run", "--threshold", "1.5"])
    with pytest.raises(ValueError, match="threshold"):
        module.validate_args(args)
