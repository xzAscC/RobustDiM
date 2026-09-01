from types import SimpleNamespace

from robustdim.evaluate import (
    DEFAULT_JUDGE_MODEL,
    SafetyJudge,
    harmbench_safety,
    parse_choice,
    parse_yes_no,
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


def test_judge_calls_gemini_with_rules() -> None:
    client = FakeClient("Yes")
    judge = SafetyJudge(client=client)
    assert judge.is_harmful("steal", "here is the exploit", "bank") is True
    assert client.kwargs is not None
    assert client.kwargs["model"] == DEFAULT_JUDGE_MODEL
    prompt = str(client.kwargs["contents"])
    assert "[BEHAVIOR]" in prompt and "steal" in prompt
    assert client.kwargs["config"].temperature == 0


def test_judge_parses_no_reply() -> None:
    judge = SafetyJudge(client=FakeClient("No"))
    assert judge.is_harmful("steal", "I cannot help with that", "bank") is False
