from __future__ import annotations

import json
from pathlib import Path

from .paths import config_path


def load_final_results_config() -> dict[str, object]:
    return json.loads(config_path("final_experiments.json").read_text(encoding="utf-8"))


def load_final_experiments() -> list[dict[str, object]]:
    payload = load_final_results_config()
    experiments = []
    for experiment in payload["experiments"]:
        row = dict(experiment)
        row["metrics_path"] = Path(str(row["metrics_path"]))
        experiments.append(row)
    return experiments


def load_final_timeline() -> list[tuple[str, str, str]]:
    payload = load_final_results_config()
    return [tuple(row) for row in payload["timeline"]]


FINAL_EXPERIMENTS = load_final_experiments()
FINAL_TIMELINE = load_final_timeline()
