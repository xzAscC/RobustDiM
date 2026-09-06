import pytest
import torch

from robustdim.metrics import (
    pairwise_stability,
    positive_spectrum,
    spectral_k,
    subspace_similarity,
    truncated_subsim,
)


def test_identical_directions_have_stability_one() -> None:
    v = torch.tensor([1.0, 2.0, 3.0])
    assert pairwise_stability([v, 2 * v, -3 * v]) == 1.0


def test_orthogonal_pair_has_stability_zero() -> None:
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert pairwise_stability([a, b]) == 0.0


def test_stability_is_mean_absolute_cosine() -> None:
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([1.0, 1.0])
    c = torch.tensor([0.0, 1.0])
    got = pairwise_stability([a, b, c])
    ab = abs(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    ac = abs(torch.nn.functional.cosine_similarity(a, c, dim=0).item())
    bc = abs(torch.nn.functional.cosine_similarity(b, c, dim=0).item())
    assert abs(got - (ab + ac + bc) / 3) < 1e-6


def test_subspace_similarity_identical_basis_is_one() -> None:
    basis = torch.eye(3, 2)
    assert subspace_similarity(basis, basis) == 1.0


def test_subspace_similarity_orthogonal_basis_is_zero() -> None:
    a = torch.tensor([[1.0], [0.0]])
    b = torch.tensor([[0.0], [1.0]])
    assert subspace_similarity(a, b) == 0.0


def test_subspace_similarity_requires_equal_width() -> None:
    with pytest.raises(ValueError):
        subspace_similarity(torch.eye(3, 1), torch.eye(3, 2))


def test_subspace_similarity_is_sign_invariant() -> None:
    basis = torch.eye(3, 2)
    flipped = basis * torch.tensor([[-1.0, 1.0, 1.0]]).T
    assert subspace_similarity(basis, flipped) == 1.0


def test_positive_spectrum_is_descending_and_filters_nonpositive() -> None:
    q, _ = torch.linalg.qr(
        torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
    )
    eigenvalues = torch.tensor([3.0, -2.0, 1.0])
    sigma = q @ torch.diag(eigenvalues) @ q.T
    values, vectors = positive_spectrum(sigma)
    torch.testing.assert_close(values, torch.tensor([3.0, 1.0], dtype=torch.float64))
    assert vectors.shape == (3, 2)
    assert vectors.dtype == torch.float64


def test_spectral_k_selects_smallest_squared_mass_prefix() -> None:
    assert spectral_k(torch.tensor([10.0, 3.0, 1.0])) == 2
    assert spectral_k(torch.tensor([10.0, 1.0, 1.0])) == 1


def test_spectral_k_raises_without_positive_values() -> None:
    with pytest.raises(ValueError):
        spectral_k(torch.tensor([0.0, -1.0]))
    with pytest.raises(ValueError):
        spectral_k(torch.tensor([]))


def test_truncated_subsim_uses_common_prefix() -> None:
    a = torch.eye(3, 2)
    b = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    expected = subspace_similarity(a[:, :2], b[:, :2])
    assert truncated_subsim(a, b) == expected
