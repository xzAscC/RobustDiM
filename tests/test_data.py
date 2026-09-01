import pytest

from robustdim.data import disjoint_subsets, sample_indices


def test_sample_indices_is_deterministic_and_unique() -> None:
    a = sample_indices(20, 5, seed=3)
    b = sample_indices(20, 5, seed=3)
    c = sample_indices(20, 5, seed=4)
    assert a == b
    assert a != c
    assert len(set(a)) == 5
    assert max(a) < 20


def test_disjoint_subsets_have_no_overlap() -> None:
    subs = disjoint_subsets(pool=30, size=5, count=4, seed=0)
    assert len(subs) == 4
    assert all(len(s) == 5 for s in subs)
    flat = [i for sub in subs for i in sub]
    assert len(set(flat)) == 20


def test_disjoint_subsets_are_deterministic() -> None:
    assert disjoint_subsets(20, 4, 3, seed=1) == disjoint_subsets(20, 4, 3, seed=1)


def test_disjoint_subsets_reject_infeasible_request() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        disjoint_subsets(pool=10, size=5, count=3, seed=0)
