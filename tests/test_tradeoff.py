from typing import Any

from robustdim.tradeoff import generate_all


def test_settings_signature_distinguishes_operating_point() -> None:
    from robustdim.tradeoff import _settings_signature

    base = {
        "model": {"id": "m", "layer": 17},
        "steering": {
            "alpha": 20.0,
            "max_new_tokens_harmbench": 256,
            "max_new_tokens_mmlu": 2048,
        },
        "methods": {"moment_k_frac": 0.01, "lowvar_frac": 0.01},
        "eval": {"harmbench_n": 100, "mmlu_n": 50, "mmlu_split": "test"},
        "stability": {"seed": 0},
    }
    other = {**base, "steering": {**base["steering"], "alpha": 30.0}}
    assert _settings_signature(base) == _settings_signature(base)
    assert _settings_signature(base) != _settings_signature(other)


def test_reusable_gens_requires_complete_rows() -> None:
    from robustdim.tradeoff import reusable_gens

    prior = {"dim": {"hb": ["a"] * 3, "mmlu": ["b"] * 2}}
    reused = reusable_gens(prior, "dim", 3, 2)
    assert reused == {"hb": ["a"] * 3, "mmlu": ["b"] * 2}
    assert reusable_gens(prior, "dim", 3, 3) is None
    assert reusable_gens(prior, "missing", 3, 2) is None


def test_gen_all_uses_batched_generation() -> None:
    class FakeLM:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], int, Any, float, int, int]] = []

        def generate_batch(
            self,
            users: list[str],
            layer: int,
            direction: Any,
            scale: float,
            tokens: int,
            batch_size: int,
        ) -> list[str]:
            self.calls.append((users, layer, direction, scale, tokens, batch_size))
            return [f"output:{user}" for user in users]

    lm = FakeLM()
    prompts = [f"prompt-{i}" for i in range(7)]
    messages = []
    direction: Any = "direction"

    outputs = generate_all(
        lm,
        prompts,
        20,
        direction,
        10,
        256,
        4,
        messages.append,
        "hb",
    )

    assert outputs == [f"output:{prompt}" for prompt in prompts]
    assert lm.calls == [
        (prompts[:4], 20, direction, 10, 256, 4),
        (prompts[4:], 20, direction, 10, 256, 4),
    ]
    assert messages == ["hb: 4/7 generated", "hb: 7/7 generated"]
