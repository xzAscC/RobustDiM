import json
import re

from robustdim.report import Tee, save_json

STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ")


def test_tee_prints_and_appends_with_timestamp(tmp_path, capsys) -> None:
    log = tmp_path / "logs" / "run.log"
    tee = Tee(log)
    tee("line 1")
    tee("line 2")
    out_lines = capsys.readouterr().out.splitlines()
    log_lines = log.read_text(encoding="utf-8").splitlines()
    assert out_lines == log_lines
    assert len(log_lines) == 2
    assert all(STAMP.match(line) for line in log_lines)
    assert log_lines[0].endswith("line 1") and log_lines[1].endswith("line 2")


def test_tee_truncates_previous_log(tmp_path) -> None:
    log = tmp_path / "run.log"
    log.write_text("stale\n", encoding="utf-8")
    Tee(log)("fresh")
    (line,) = log.read_text(encoding="utf-8").splitlines()
    assert STAMP.match(line) and line.endswith("fresh")


def test_save_json_rewrites_incrementally(tmp_path) -> None:
    path = tmp_path / "logs" / "scores.json"
    save_json(path, {"a": 1})
    save_json(path, {"a": 1, "b": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
