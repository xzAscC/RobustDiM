import torch
import pytest
from typing import Any, cast

from robustdim.model import HookedLM, last_prompt_hook, prefill_hook


def test_hook_steers_only_last_prompt_position() -> None:
    direction = torch.tensor([1.0, -1.0])
    hook = last_prompt_hook(direction, alpha=10.0, pos=3)
    out = hook(None, None, torch.ones(1, 4, 2))
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out[0, 3], torch.tensor([11.0, -9.0]))
    torch.testing.assert_close(out[0, 0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(out[0, 1], torch.tensor([1.0, 1.0]))


def test_hook_skips_decode_steps() -> None:
    hook = last_prompt_hook(torch.ones(2), alpha=10.0, pos=3)
    decode = torch.ones(1, 1, 2)
    out = hook(None, None, decode)
    torch.testing.assert_close(out, decode)


def test_hook_handles_tuple_output() -> None:
    hook = last_prompt_hook(torch.ones(2), alpha=1.0, pos=0)
    out = hook(None, None, (torch.zeros(1, 2, 2), "past"))
    assert isinstance(out, tuple) and out[1] == "past"
    torch.testing.assert_close(out[0][0, 0], torch.ones(2))
    torch.testing.assert_close(out[0][0, 1], torch.zeros(2))


def test_prefill_hook_applies_only_to_full_prompt() -> None:
    direction = torch.ones(2)
    hook = prefill_hook(direction, alpha=2.0, pos=3, width=4)
    prompt = torch.ones(1, 4, 2)
    out = hook(None, None, prompt)
    torch.testing.assert_close(out[0, 3], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(hook(None, None, prompt), prompt)
    decode = torch.ones(1, 1, 2)
    torch.testing.assert_close(hook(None, None, decode), decode)


def test_prefill_wrapper_skips_one_token_decode_when_pos_is_zero() -> None:
    direction = torch.ones(2)
    hook = prefill_hook(direction, alpha=2.0, pos=0, width=1)
    prompt = torch.ones(1, 1, 2)
    torch.testing.assert_close(hook(None, None, prompt)[0, 0], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(hook(None, None, prompt), prompt)
    raw = last_prompt_hook(direction, alpha=2.0, pos=0)
    raw_first = raw(None, None, prompt)
    raw_second = raw(None, None, raw_first)
    torch.testing.assert_close(raw_first[0, 0], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(raw_second[0, 0], torch.tensor([5.0, 5.0]))


def test_prefill_hook_handles_tuple_output() -> None:
    hook = prefill_hook(torch.ones(2), alpha=1.0, pos=3, width=4)
    out = hook(None, None, (torch.zeros(1, 4, 2), "past"))
    assert isinstance(out, tuple) and out[1] == "past"
    torch.testing.assert_close(out[0][0, 3], torch.ones(2))


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.padding_side = "right"
        self.decoded: list[torch.Tensor] = []

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        width = max(len(text) for text in texts)
        ids = torch.tensor([[1] * width for _ in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: torch.Tensor, **_: object) -> str:
        self.decoded.append(ids.clone())
        return "generated"


class _FakeHandle:
    def __init__(self, layer: "_FakeLayer") -> None:
        self.layer = layer

    def remove(self) -> None:
        self.layer.handles_removed += 1


class _FakeLayer:
    def __init__(self) -> None:
        self.hooks: list[object] = []
        self.handles_removed = 0
        self.prefill_hidden = torch.empty(0)
        self.decode_hidden = torch.empty(0)

    def register_forward_hook(self, hook: object) -> _FakeHandle:
        self.hooks.append(hook)
        return _FakeHandle(self)


class _FakeModel:
    def __init__(self, layers: list[_FakeLayer], fail: bool = False) -> None:
        self.model: Any = type("Inner", (), {"layers": layers})()
        self.fail = fail

    def parameters(self) -> object:
        return iter([torch.empty(0)])

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: object
    ) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("generation failed")
        hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
        decode_hidden = torch.zeros(input_ids.shape[0], 1, 2)
        for layer in self.model.layers:
            if layer.hooks:
                layer.prefill_hidden = layer.hooks[0](layer, None, hidden)
                layer.decode_hidden = layer.hooks[0](layer, None, decode_hidden)
        return torch.cat(
            [input_ids, torch.tensor([[8, 9]]).expand(input_ids.shape[0], -1)], dim=1
        )


def _fake_lm(
    monkeypatch: pytest.MonkeyPatch, *, fail: bool = False
) -> tuple[HookedLM, list[_FakeLayer]]:
    monkeypatch.setattr("robustdim.model.chat_text", lambda _tokenizer, user: user)
    layers = [_FakeLayer(), _FakeLayer()]
    lm = cast(Any, object.__new__(HookedLM))
    lm.tokenizer = _FakeTokenizer()
    lm.model = _FakeModel(layers, fail=fail)
    lm.layers = layers
    return lm, layers


def test_generate_batch_steers_last_padded_position_and_restores_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lm, layers = _fake_lm(monkeypatch)
    out = lm.generate_batch(
        ["a", "abcd"], layer=1, direction=torch.ones(2), alpha=2.0, batch_size=2
    )
    assert out == ["generated", "generated"]
    assert lm.tokenizer.padding_side == "right"
    assert len(layers[0].hooks) == 0 and len(layers[1].hooks) == 1
    torch.testing.assert_close(layers[1].prefill_hidden[:, -1], torch.full((2, 2), 2.0))
    torch.testing.assert_close(layers[1].decode_hidden, torch.zeros(2, 1, 2))
    torch.testing.assert_close(lm.tokenizer.decoded[0], torch.tensor([8, 9]))


def test_generate_batch_no_hook_without_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lm, layers = _fake_lm(monkeypatch)
    assert lm.generate_batch(["abc"], layer=1, direction=None) == ["generated"]
    assert all(not layer.hooks for layer in layers)


def test_generate_restores_padding_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    lm, _ = _fake_lm(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="generation failed"):
        lm.generate_batch(["abc"], layer=0)
    assert lm.tokenizer.padding_side == "right"
