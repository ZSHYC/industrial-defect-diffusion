from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def config_path(name: str) -> Path:
    return PROJECT_ROOT / "configs" / name


def resolve_data_root(data_root: Path | str | None = None) -> Path:
    value = Path(data_root) if data_root is not None else None
    if value is None:
        env_value = os.environ.get("DATA_ROOT")
        if env_value:
            value = Path(env_value)
    if value is None:
        raise ValueError("MVTec AD data root is required. Pass --data-root or set the DATA_ROOT environment variable.")
    return value


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
