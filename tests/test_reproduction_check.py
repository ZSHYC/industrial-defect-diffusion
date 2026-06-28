import importlib.util
from pathlib import Path


def load_script_module():
    path = Path("scripts") / "14_reproduction_check.py"
    spec = importlib.util.spec_from_file_location("reproduction_check", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reproduction_check_non_strict_allows_missing_data_root(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DATA_ROOT", raising=False)
    module = load_script_module()
    args = module.parse_args([])
    args.output_dir = tmp_path
    report, ok = module.build_report(args)
    assert ok
    assert "DATA_ROOT" in report
    assert "Metrics JSON Files" in report
