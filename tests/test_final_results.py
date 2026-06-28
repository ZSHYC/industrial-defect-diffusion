from pathlib import Path

from industrial_defect.final_results import FINAL_EXPERIMENTS, FINAL_TIMELINE
from industrial_defect.io import read_json


def test_final_experiment_manifest() -> None:
    assert len(FINAL_EXPERIMENTS) == 9
    roles = {str(row["role"]) for row in FINAL_EXPERIMENTS}
    assert {"baseline", "overall_best", "class_specialist", "diagnostic_tradeoff"}.issubset(roles)


def test_final_metrics_files_exist_and_have_required_keys() -> None:
    required = {"pixel_precision", "pixel_recall", "pixel_f1", "best_pixel_f1", "image_f1", "class_metrics"}
    for experiment in FINAL_EXPERIMENTS:
        path = Path(experiment["metrics_path"])
        assert path.exists(), path.as_posix()
        metrics = read_json(path)
        assert required.issubset(metrics), path.as_posix()


def test_timeline_reaches_stage15() -> None:
    assert FINAL_TIMELINE[-1][0] == "stage15"

