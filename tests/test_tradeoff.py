from typing import Any

from robustdim.tradeoff import generate_all


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
