import torch

from robustdim.model import last_prompt_hook


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
