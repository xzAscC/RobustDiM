import pytest
import torch
from pathlib import Path

from robustdim.metrics import subspace_similarity
from robustdim.sweep import (
    choose_shared,
    cross_fraction_matrix,
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


def test_cross_fraction_matrix_uses_absolute_cosine() -> None:
    directions = {
        0.01: torch.tensor([1.0, 0.0]),
        0.05: torch.tensor([0.0, 3.0]),
        0.1: torch.tensor([-2.0, 0.0]),
    }
    report = cross_fraction_matrix(directions)
    assert report["fracs"] == [0.01, 0.05, 0.1]
    assert report["cos_matrix"][0][1] == pytest.approx(0.0)
    assert report["cos_matrix"][0][2] == pytest.approx(1.0)
    assert all(
        m[i][i] == pytest.approx(1.0) for i, m in enumerate([report["cos_matrix"]] * 3)
    )


def test_metrics_collects_all_verdicts_in_order() -> None:
    import re
    from types import SimpleNamespace

    from robustdim.evaluate import SafetyJudge
    from robustdim.sweep import _metrics

    class BehaviorIndexClient:
        @property
        def models(self) -> "BehaviorIndexClient":
            return self

        def generate_content(self, **kwargs: object) -> SimpleNamespace:
            match = re.search(r"\bb(\d+)\b", str(kwargs["contents"]))
            index = int(match.group(1)) if match else 0
            return SimpleNamespace(text=["yes", "no", "na"][index % 3])

    rows = [{"behavior": f"b{i}", "context": f"c{i}"} for i in range(7)]
    outputs = [f"out{i}" for i in range(7)]
    safety, degenerate = _metrics(
        SafetyJudge(client=BehaviorIndexClient(), project="unit-test"), rows, outputs
    )
    assert safety == pytest.approx(1 - 3 / 7)
    assert degenerate == pytest.approx(2 / 7)


def test_signature_captures_sweep_grid() -> None:
    from robustdim.sweep import _signature

    base = {
        "model": {"id": "m"},
        "eval": {"mmlu_n": 50},
        "sweep": {"layers": [14], "alphas": [10], "screen_methods": ["dim"]},
    }
    other = {
        "model": {"id": "m"},
        "eval": {"mmlu_n": 50},
        "sweep": {"layers": [17], "alphas": [10], "screen_methods": ["dim"]},
    }
    assert _signature(base) == _signature(base)
    assert _signature(base) != _signature(other)


def test_load_prior_rejects_missing_or_mismatched(tmp_path: Path) -> None:
    import json

    from robustdim.sweep import load_prior

    log = tmp_path / "sweep.json"
    assert load_prior(log, {"a": 1}) is None
    log.write_text(json.dumps({"screen": [], "signature": {"a": 2}}))
    assert load_prior(log, {"a": 1}) is None
    log.write_text(json.dumps({"screen": [1], "signature": {"a": 1}}))
    assert load_prior(log, {"a": 1}) == {"screen": [1], "signature": {"a": 1}}
    log.write_text("{not json")
    assert load_prior(log, {"a": 1}) is None


def test_pending_conditions_skip_completed() -> None:
    from robustdim.sweep import pending_conditions

    grid = [
        {"method": "dim", "layer": 14, "alpha": 10},
        {"method": "dim", "layer": 14, "alpha": 20},
        {"method": "moment", "layer": 17, "alpha": 10},
    ]
    done = [{"method": "dim", "layer": 14, "alpha": 10}]
    pending = pending_conditions(grid, done)
    assert pending == [
        {"method": "dim", "layer": 14, "alpha": 20},
        {"method": "moment", "layer": 17, "alpha": 10},
    ]


def test_variance_row_complete_requires_all_fields() -> None:
    from robustdim.sweep import variance_row_complete

    complete = {
        "method": "moment_proj",
        "frac": 0.01,
        "local_alphas": {20: {"safety": 0.8, "degenerate": 0.0}},
        "mmlu": 0.4,
        "stability": 0.9,
        "cos_dim": 0.5,
        "subsim_dim": 0.25,
    }
    assert variance_row_complete(complete)
    assert not variance_row_complete({**complete, "mmlu": None})
    assert not variance_row_complete({**complete, "local_alphas": {}})
    partial = dict(complete)
    del partial["stability"]
    assert not variance_row_complete(partial)


def test_find_verify_row_matches_candidate() -> None:
    from robustdim.sweep import find_verify_row

    prior = [
        {"method": "dim", "layer": 14, "alpha": 10, "mmlu": 0.4},
        {"method": "dim", "layer": 17, "alpha": 20, "mmlu": 0.38},
    ]
    hit = find_verify_row(prior, {"method": "dim", "layer": 17, "alpha": 20})
    assert hit is not None and hit["mmlu"] == 0.38
    assert (
        find_verify_row(prior, {"method": "moment", "layer": 17, "alpha": 20}) is None
    )


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
