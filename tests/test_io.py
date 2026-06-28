from industrial_defect.io import read_csv, read_json, write_csv, write_json


def test_csv_roundtrip(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    write_csv([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], path)
    rows = read_csv(path)
    assert rows == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "payload.json"
    write_json({"hello": "world", "value": 3}, path)
    assert read_json(path) == {"hello": "world", "value": 3}

