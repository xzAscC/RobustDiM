from robustdim.data import sample_indices


def test_sample_indices_is_deterministic_and_unique() -> None:
    a = sample_indices(20, 5, seed=3)
    b = sample_indices(20, 5, seed=3)
    c = sample_indices(20, 5, seed=4)
    assert a == b
    assert a != c
    assert len(set(a)) == 5
    assert max(a) < 20
