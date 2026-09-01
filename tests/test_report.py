import json

from robustdim.report import Tee, save_json


def test_tee_prints_and_appends(tmp_path, capsys) -> None:
    log = tmp_path / "logs" / "run.log"
    tee = Tee(log)
    tee("line 1")
    tee("line 2")
    assert capsys.readouterr().out == "line 1\nline 2\n"
    assert log.read_text(encoding="utf-8") == "line 1\nline 2\n"


def test_tee_truncates_previous_log(tmp_path) -> None:
    log = tmp_path / "run.log"
    log.write_text("stale\n", encoding="utf-8")
    Tee(log)("fresh")
    assert log.read_text(encoding="utf-8") == "fresh\n"


def test_save_json_rewrites_incrementally(tmp_path) -> None:
    path = tmp_path / "logs" / "scores.json"
    save_json(path, {"a": 1})
    save_json(path, {"a": 1, "b": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
