import pytest
import torch

from robustdim.metrics import subspace_similarity
from robustdim.sweep import (
    choose_shared,
    local_alphas,
    screen_grid,
    select_fracs,
    select_top_per_method,
    stage_counts,
)


def test_screen_grid_contains_expected_layer_alpha_pairs() -> None:
    records = screen_grid(
        [14, 17, 20, 23], [10, 20, 30, 40], ["dim", "moment", "moment_proj"]
    )
    assert len(records) == 48
    assert {(r["layer"], r["alpha"]) for r in records} == {
        (layer, alpha) for layer in [14, 17, 20, 23] for alpha in [10, 20, 30, 40]
    }


def test_screen_grid_contains_only_requested_methods() -> None:
    records = screen_grid([1], [2], ["dim", "moment_proj"])
    assert {r["method"] for r in records} == {"dim", "moment_proj"}


def test_top_k_selection_is_grouped_by_method() -> None:
    records = [
        {"method": "dim", "score": 0.8, "safety": 0.8, "degenerate": 0.2},
        {"method": "dim", "score": 0.8, "safety": 0.9, "degenerate": 0.1},
        {"method": "dim", "score": 0.7, "safety": 0.7, "degenerate": 0.0},
        {"method": "moment", "score": 0.6, "safety": 0.6, "degenerate": 0.0},
    ]
    selected = select_top_per_method(records, 2)
    assert [r["safety"] for r in selected["dim"]] == [0.9, 0.8]
    assert len(selected["moment"]) == 1


def test_local_alpha_neighborhood_is_bounded() -> None:
    grid = [10, 20, 30]
    assert local_alphas(grid, 0, 1) == [10, 20]
    assert local_alphas(grid, 2, 1) == [20, 30]


def test_fraction_grid_is_read_from_config() -> None:
    assert (
        stage_counts(
            {
                "layers": [1],
                "alphas": [2],
                "screen_methods": ["dim"],
                "fractions": [0.2, 0.4],
                "frac_methods": ["lowvar"],
            }
        )["fraction"]
        == 2
    )


def test_one_dimensional_subsim_matches_squared_cosine() -> None:
    a = torch.tensor([[1.0], [0.0]])
    b = torch.tensor([[3.0], [4.0]]) / 5
    assert subspace_similarity(a, b) == pytest.approx((3 / 5) ** 2)


def test_shared_final_operating_point_is_distinct_from_method_best() -> None:
    records = [
        {"method": "dim", "layer": 14, "alpha": 10, "final_score": 0.95},
        {"method": "moment", "layer": 17, "alpha": 20, "final_score": 0.90},
    ]
    assert choose_shared(records) == (14, 10, 0.95)


def test_sweep_dry_run_reports_stage_counts() -> None:
    counts = stage_counts(
        {
            "layers": [14, 17],
            "alphas": [10, 20, 30],
            "screen_methods": ["dim", "moment_proj"],
            "fractions": [0.1, 0.2, 0.3],
            "frac_methods": ["moment_proj", "lowvar"],
            "top_per_method": 2,
        }
    )
    assert counts == {
        "screen": 12,
        "verify_candidates": 4,
        "baseline": 1,
        "fraction": 6,
    }


def test_select_fracs_penalizes_degenerate() -> None:
    records = [
        {
            "method": "moment_proj",
            "frac": 0.01,
            "mmlu": 0.4,
            "local_alphas": {20: {"safety": 0.9, "degenerate": 0.0}},
        },
        {
            "method": "moment_proj",
            "frac": 0.05,
            "mmlu": 0.4,
            "local_alphas": {20: {"safety": 0.95, "degenerate": 0.3}},
        },
    ]
    selected = select_fracs(records, baseline=0.4, penalty=1.0, shared_alpha=20)
    assert selected["moment_proj"]["frac"] == 0.01


def test_select_fracs_falls_back_when_shared_alpha_missing() -> None:
    records = [
        {
            "method": "moment_proj",
            "frac": 0.01,
            "mmlu": 0.4,
            "local_alphas": {10: {"safety": 0.9, "degenerate": 0.0}},
        }
    ]
    selected = select_fracs(records, baseline=0.4, penalty=1.0, shared_alpha=20)
    assert selected["moment_proj"]["frac"] == 0.01


def test_choose_shared_requires_candidates() -> None:
    with pytest.raises(ValueError, match="verified"):
        choose_shared([])


def test_sweep_cfg_requires_valid_batch_size() -> None:
    from robustdim.sweep import _sweep_cfg

    with pytest.raises(ValueError, match="batch_size"):
        _sweep_cfg({"sweep": {"batch_size": 0}})


def test_sweep_cfg_requires_required_frac_methods() -> None:
    from robustdim.sweep import _sweep_cfg

    with pytest.raises(ValueError, match="moment_proj"):
        _sweep_cfg({"sweep": {"frac_methods": ["lowvar"]}})
