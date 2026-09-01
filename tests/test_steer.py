import torch

from robustdim.model import apply_steer


def test_apply_steer_adds_scaled_direction() -> None:
    hidden = torch.zeros(2, 3, 4)
    direction = torch.tensor([1.0, 0.0, 0.0, -1.0])
    got = apply_steer(hidden, direction, alpha=2.0)
    expected = torch.tensor([2.0, 0.0, 0.0, -2.0]).view(1, 1, 4).expand_as(hidden)
    torch.testing.assert_close(got, expected)
