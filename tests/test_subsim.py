import torch
import pytest

from robustdim.metrics import subspace_similarity
from robustdim.metrics import positive_spectrum
from robustdim.subsim import (
    bottom_energy,
    delta_covariance,
    family_report,
    lda_attribution,
    pairwise_matrix,
    population_covariance,
    rank_curve,
    tau_pairwise,
)


def test_population_covariance_uses_divisor_n() -> None:
    h = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(population_covariance(h), expected)


def test_rank_curve_uses_squared_overlap() -> None:
    a = torch.eye(2, 1)
    b = torch.tensor([[1.0], [1.0]]) / 2**0.5
    assert rank_curve([a, b], [1])[1] == pytest.approx(0.5)


def test_rank_curve_marks_unavailable_width_as_none() -> None:
    bases = [torch.eye(3, 1), torch.eye(3, 2)]
    assert rank_curve(bases, [1, 2]) == {1: 1.0, 2: None}


def test_tau_subsim_uses_each_pair_common_k() -> None:
    bases = [torch.eye(3, 2), torch.eye(3, 1)]
    mean, matrix, widths = tau_pairwise(
        [torch.tensor([4.0, 0.1]), torch.tensor([4.0])], bases, tau=0.95
    )
    expected = subspace_similarity(bases[0][:, :1], bases[1][:, :1])
    assert widths == [1, 1]
    assert mean == expected
    assert matrix[0][1] == expected


def test_delta_spectrum_keeps_only_positive_eigenvectors() -> None:
    pos = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    neg = torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    delta = delta_covariance(population_covariance(pos), population_covariance(neg))
    values, vectors = positive_spectrum(delta)
    assert torch.all(values > 0)
    assert vectors.shape[1] == values.numel()


def test_delta_covariance_subtracts_covariances_exactly() -> None:
    a = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    b = torch.tensor([[0.5, 0.2], [0.2, 0.3]])
    delta = delta_covariance(a, b)
    torch.testing.assert_close(delta, (a - b).double())


def test_pooled_covariance_curve_uses_bottom_eigenvectors() -> None:
    pooled = torch.diag(torch.tensor([1.0, 4.0]))
    _, vectors = torch.linalg.eigh(pooled)
    curve = rank_curve([vectors, vectors], [1], position="bottom")
    assert curve[1] == 1.0
    assert torch.equal(vectors[:, :1], torch.tensor([[1.0], [0.0]]))


def test_bottom_energy_measures_unit_vector_mass() -> None:
    basis = torch.eye(3)
    assert bottom_energy(torch.tensor([2.0, 0.0, 0.0]), basis, 1) == pytest.approx(1.0)
    assert bottom_energy(torch.tensor([0.0, 5.0, 0.0]), basis, 1) == pytest.approx(0.0)
    assert bottom_energy(torch.ones(3), basis, 2) == pytest.approx(2 / 3)


def test_lda_attribution_reports_stability_and_energy() -> None:
    torch.manual_seed(5)
    neg = torch.randn(60, 2)
    pos_list = [torch.randn(60, 2) + 3.0 for _ in range(2)]
    covs = [population_covariance(p) for p in pos_list]
    bases = [torch.linalg.eigh(c + population_covariance(neg))[1] for c in covs]
    report = lda_attribution(pos_list, neg, bases, [1])
    assert 0.0 <= report["stability"] <= 1.0
    assert set(report["bottom_energy"]) == {"1"}
    assert 0.0 <= report["bottom_energy"]["1"] <= 1.0
    assert 0.0 <= report["dim_bottom_energy"]["1"] <= 1.0
    assert 0.0 <= report["dim_stability"] <= 1.0


def test_positive_covariance_curve_uses_bottom_eigenvectors() -> None:
    cov = torch.diag(torch.tensor([1.0, 4.0]))
    _, vectors = torch.linalg.eigh(cov)
    curve = rank_curve([vectors, vectors], [1], position="bottom")
    assert curve[1] == 1.0
    assert torch.equal(vectors[:, :1], torch.tensor([[1.0], [0.0]]))


def test_second_moment_curve_uses_top_eigenvectors() -> None:
    moment = torch.diag(torch.tensor([1.0, 4.0]))
    _, vectors = torch.linalg.eigh(moment)
    curve = rank_curve([vectors, vectors], [1], position="top")
    assert curve[1] == 1.0
    assert torch.equal(vectors[:, -1:], torch.tensor([[0.0], [1.0]]))


def test_subsim_pairwise_matrix_is_symmetric() -> None:
    bases = [torch.eye(3, 2), torch.eye(3, 2)[:, [1, 0]], torch.eye(3, 2)]
    matrix = pairwise_matrix(bases)
    assert matrix == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_pairwise_matrix_at_explicit_k_uses_that_width() -> None:
    a = torch.eye(3, 2)
    b = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert pairwise_matrix([a, b], k=1)[0][1] == pytest.approx(0.0)


def test_family_report_records_pairwise_width() -> None:
    _, vectors = torch.linalg.eigh(torch.diag(torch.tensor([1.0, 4.0])))
    report = family_report([vectors, vectors], [1], position="bottom", pairwise_k=1)
    assert report["pairwise_k"] == 1
    assert report["curve"]["1"] == 1.0
    assert report["pairwise"][0][1] == 1.0
