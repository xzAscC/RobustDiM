import pytest
import torch

from robustdim.directions import (
    construct,
    dim,
    lda,
    lowvar,
    moment,
    moment_proj,
    spectrum_k,
    unit,
)


def _cov(h: torch.Tensor) -> torch.Tensor:
    c = h - h.mean(0)
    return c.T @ c / h.shape[0]


def test_spectrum_k_is_one_percent_of_hidden_size() -> None:
    assert spectrum_k(2560, 0.01) == 26
    assert spectrum_k(10, 0.01) == 1


def test_dim_is_mean_difference() -> None:
    pos = torch.tensor([[2.0, 0.0], [4.0, 2.0]])
    neg = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
    torch.testing.assert_close(dim(pos, neg), torch.tensor([2.0, 0.0]))


def test_lda_recovers_mean_difference_axis() -> None:
    torch.manual_seed(0)
    pos = torch.randn(200, 2)
    pos[:, 0] += 3
    pos[:, 1] *= 0.2
    neg = torch.randn(200, 2)
    neg[:, 0] -= 3
    neg[:, 1] *= 0.2
    v = unit(lda(pos, neg))
    assert abs(v[0]).item() > 0.95


def test_large_ridge_matches_dim_direction() -> None:
    torch.manual_seed(1)
    pos = torch.randn(50, 4) + 0.5
    neg = torch.randn(50, 4) - 0.5
    v_ridge = unit(lda(pos, neg, gamma=1e6))
    v_dim = unit(dim(pos, neg))
    assert torch.dot(v_ridge, v_dim).abs().item() > 0.999


def test_lowvar_drops_high_variance_axis() -> None:
    torch.manual_seed(2)
    n = 80
    pos = torch.zeros(n, 2)
    pos[:, 0] = torch.randn(n) * 3
    pos[:, 1] = 1.0
    d = torch.tensor([0.0, 1.0])
    v = lowvar(pos, d, frac=0.5)
    assert v[0].abs().item() < 1e-5
    assert v[1].item() > 0.5


def test_moment_is_top_eigenvector_aligned_with_dim() -> None:
    torch.manual_seed(3)
    n = 60
    pos = torch.zeros(n, 2)
    pos[:, 0] = 2.0
    pos[:, 1] = 0.1 * torch.randn(n)
    d = dim(pos, torch.zeros(n, 2))
    v = moment(pos, d)
    evals, evecs = torch.linalg.eigh(_cov(pos) + torch.outer(pos.mean(0), pos.mean(0)))
    top = evecs[:, -1]
    if torch.dot(top, d) < 0:
        top = -top
    torch.testing.assert_close(v, top, atol=1e-5, rtol=1e-5)
    assert torch.dot(v, d).item() > 0


def test_moment_proj_lives_in_top_eigenspace() -> None:
    torch.manual_seed(4)
    pos = torch.randn(40, 3) + torch.tensor([2.0, 0.0, 0.0])
    d = dim(pos, torch.randn(40, 3))
    v = moment_proj(pos, d, frac=1 / 3)
    m = pos.T @ pos / pos.shape[0]
    evals, evecs = torch.linalg.eigh(m)
    p = evecs[:, -1:]
    recon = p @ (p.T @ v)
    torch.testing.assert_close(v, recon, atol=1e-5, rtol=1e-5)


def test_construct_dim_returns_supplied_direction() -> None:
    d = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(construct("dim", torch.zeros(3, 2), None, d), d)


def test_construct_lda_requires_negative_class() -> None:
    with pytest.raises(ValueError, match="negative"):
        construct("lda", torch.zeros(3, 2), None, torch.zeros(2))
