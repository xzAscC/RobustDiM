import torch

from robustdim.metrics import pairwise_stability


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
