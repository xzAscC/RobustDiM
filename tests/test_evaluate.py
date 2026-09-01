from robustdim.evaluate import harmbench_safety, parse_choice, parse_yes_no


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
