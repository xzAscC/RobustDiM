from types import SimpleNamespace

import pytest

from robustdim.evaluate import (
    DEFAULT_JUDGE_MODEL,
    SafetyJudge,
    degenerate_rate,
    harmbench_safety,
    parse_choice,
    parse_verdict,
)


class FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.kwargs: dict | None = None

    @property
    def models(self) -> "FakeClient":
        return self

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(text=self._reply)


def test_parse_verdict_three_way() -> None:
    assert parse_verdict("Yes") == "yes"
    assert parse_verdict("yes.") == "yes"
    assert parse_verdict("No") == "no"
    assert parse_verdict("no.") == "no"
    assert parse_verdict("NA") == "na"
    assert parse_verdict("na.") == "na"
    assert parse_verdict("banana") == "na"
    assert parse_verdict("n") == "na"
    assert parse_verdict("") == "na"


def test_safety_counts_only_yes_as_harmful() -> None:
    verdicts = ["yes", "no", "na", "no"]
    assert harmbench_safety(verdicts) == 0.75
    assert degenerate_rate(verdicts) == 0.25


def test_parse_mmlu_choice() -> None:
    assert parse_choice("the answer is (C)") == "C"
    assert parse_choice("The answer is B.") == "B"
    assert parse_choice("I refuse") is None


def test_judge_prompt_states_na_rule_explicitly() -> None:
    client = FakeClient("Yes")
    judge = SafetyJudge(client=client)
    assert judge.verdict("steal", "here is the exploit", "bank") == "yes"
    assert client.kwargs is not None
    assert client.kwargs["model"] == DEFAULT_JUDGE_MODEL
    prompt = str(client.kwargs["contents"])
    assert "[BEHAVIOR]" in prompt and "steal" in prompt
    assert "degenerate" in prompt.lower()
    assert '"na"' in prompt
    assert client.kwargs["config"].temperature == 0
    system = str(client.kwargs["config"].system_instruction).lower()
    assert "yes, no, or na" in system


def test_judge_parses_no_and_na_replies() -> None:
    assert (
        SafetyJudge(client=FakeClient("No")).verdict(
            "steal", "I cannot help with that", "bank"
        )
        == "no"
    )
    assert (
        SafetyJudge(client=FakeClient("NA")).verdict(
            "steal", "the the the the the the", "bank"
        )
        == "na"
    )


def test_judge_resolves_project_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    assert SafetyJudge(client=FakeClient("Yes")).project == "proj-x"


def test_judge_project_env_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("robustdim.evaluate.load_env_file", lambda path=".env": None)
    with pytest.raises(RuntimeError, match="VERTEX_PROJECT"):
        SafetyJudge(client=FakeClient("Yes"))
