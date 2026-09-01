import os
from pathlib import Path

import pytest

from robustdim.config import load_env_file
from robustdim.evaluate import (
    SafetyJudge,
    harmbench_safety,
    parse_choice,
    parse_yes_no,
)


def test_parse_yes_no() -> None:
    assert parse_yes_no("Yes") is True
    assert parse_yes_no("yes.") is True
    assert parse_yes_no("No") is False
    assert parse_yes_no("n") is False


def test_safety_is_one_minus_attack_success() -> None:
    assert harmbench_safety([False, False, True, False]) == 0.75


def test_parse_mmlu_choice() -> None:
    assert parse_choice("the answer is (C)") == "C"
    assert parse_choice("The answer is B.") == "B"
    assert parse_choice("I refuse") is None


def test_judge_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAMBANOVA_API_KEY", raising=False)
    monkeypatch.setattr("robustdim.config.load_env_file", lambda path=".env": None)
    monkeypatch.setattr("robustdim.evaluate.load_env_file", lambda path=".env": None)
    with pytest.raises(RuntimeError, match="SAMBANOVA_API_KEY"):
        SafetyJudge()


def test_load_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAMBANOVA_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("SAMBANOVA_API_KEY=from-file\n")
    load_env_file(env)
    assert os.environ["SAMBANOVA_API_KEY"] == "from-file"


def test_judge_posts_to_sambanova(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAMBANOVA_API_KEY", "test-key")
    captured: dict = {}

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "Yes"}}]}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> Resp:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return Resp()

    monkeypatch.setattr("robustdim.evaluate.requests.post", fake_post)
    judge = SafetyJudge()
    assert judge.is_harmful("steal", "here is the exploit", "bank") is True
    assert captured["url"] == "https://api.sambanova.ai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "DeepSeek-V3.2"
    assert captured["json"]["temperature"] == 0
