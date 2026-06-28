import pytest

from industrial_defect.paths import resolve_data_root


def test_resolve_data_root_prefers_argument(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert resolve_data_root(tmp_path) == tmp_path


def test_resolve_data_root_uses_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert resolve_data_root() == tmp_path


def test_resolve_data_root_requires_value(monkeypatch) -> None:
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="DATA_ROOT"):
        resolve_data_root()

