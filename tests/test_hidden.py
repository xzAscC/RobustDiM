import torch

from robustdim.hidden import last_token


def test_last_token_uses_attention_mask_not_padding() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [9.0, 9.0]],
            [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    got = last_token(hidden, mask)
    torch.testing.assert_close(got, torch.tensor([[2.0, 0.0], [5.0, 0.0]]))
