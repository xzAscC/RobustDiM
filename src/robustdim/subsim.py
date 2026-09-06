import argparse
import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from robustdim.config import load_config
from robustdim.data import disjoint_subsets, load_class_prompts, sample_indices
from robustdim.directions import centered_cov, second_moment, spectrum_k
from robustdim.metrics import (
    positive_spectrum,
    spectral_k,
    subspace_similarity,
    truncated_subsim,
)
from robustdim.model import HookedLM
from robustdim.plots import save_subsim
from robustdim.report import Tee, save_json


def population_covariance(h: Tensor) -> Tensor:
    return centered_cov(h.double())


def delta_covariance(pos_cov: Tensor, neg_cov: Tensor) -> Tensor:
    """Difference of two population covariance matrices."""
    return pos_cov.to(torch.float64) - neg_cov.to(torch.float64)


def _oriented_basis(basis: Tensor, k: int, position: str) -> Tensor:
    if position == "top":
        return basis[:, -k:]
    return basis[:, :k]


def rank_curve(
    bases_per_replicate: list[Tensor],
    ks: list[int],
    position: Literal["top", "bottom", "positive"] = "positive",
) -> dict[int, float | None]:
    curve: dict[int, float | None] = {}
    for k in ks:
        if any(basis.shape[1] < k for basis in bases_per_replicate):
            curve[k] = None
            continue
        oriented = [
            _oriented_basis(basis, k, position)
            if position != "positive"
            else basis[:, :k]
            for basis in bases_per_replicate
        ]
        scores = [
            subspace_similarity(a, b)
            for i, a in enumerate(oriented)
            for b in oriented[i + 1 :]
        ]
        curve[k] = sum(scores) / len(scores) if scores else 1.0
    return curve


def pairwise_matrix(
    bases_per_replicate: list[Tensor], k: int | None = None, position: str = "positive"
) -> list[list[float]]:
    matrix = [[1.0 for _ in bases_per_replicate] for _ in bases_per_replicate]
    for i, first in enumerate(bases_per_replicate):
        for j in range(i + 1, len(bases_per_replicate)):
            second = bases_per_replicate[j]
            width = min(first.shape[1], second.shape[1]) if k is None else k
            if width == 0:
                value = 0.0
            else:
                a = (
                    _oriented_basis(first, width, position)
                    if position != "positive"
                    else first[:, :width]
                )
                b = (
                    _oriented_basis(second, width, position)
                    if position != "positive"
                    else second[:, :width]
                )
                value = subspace_similarity(a, b)
            matrix[i][j] = matrix[j][i] = value
    return matrix


def tau_pairwise(
    positive_values: list[Tensor],
    bases_per_replicate: list[Tensor],
    tau: float,
) -> tuple[float, list[list[float]], list[int]]:
    widths = [spectral_k(values, tau) for values in positive_values]
    matrix = [[1.0 for _ in bases_per_replicate] for _ in bases_per_replicate]
    scores: list[float] = []
    for i, first in enumerate(bases_per_replicate):
        for j in range(i + 1, len(bases_per_replicate)):
            width = min(widths[i], widths[j])
            value = truncated_subsim(
                first[:, :width], bases_per_replicate[j][:, :width]
            )
            matrix[i][j] = matrix[j][i] = value
            scores.append(value)
    return (sum(scores) / len(scores) if scores else 1.0), matrix, widths


def family_report(
    bases: list[Tensor],
    ks: list[int],
    position: Literal["top", "bottom", "positive"],
    pairwise_k: int | None = None,
) -> dict[str, Any]:
    return {
        "pairwise_k": pairwise_k,
        "curve": {
            str(k): value for k, value in rank_curve(bases, ks, position).items()
        },
        "pairwise": pairwise_matrix(bases, k=pairwise_k, position=position),
    }


def run(cfg: dict[str, Any], tee: Tee) -> dict[str, object]:
    pos_texts = load_class_prompts(cfg["data"]["benign"])
    neg_texts = load_class_prompts(cfg["data"]["harmful"])
    replicates = cfg["stability"]["replicates"]
    cov_n = cfg["data"]["cov_n"]
    seed = cfg["stability"]["seed"]
    pos_pool = min(len(pos_texts), cfg["stability"]["pool_n"])
    neg_pool = min(len(neg_texts), cov_n)
    if pos_pool < replicates * cov_n:
        raise ValueError(
            f"need {replicates}x{cov_n} disjoint positives, pool is {pos_pool}"
        )
    if neg_pool < cov_n:
        raise ValueError(f"need {cov_n} negatives, pool is {neg_pool}")
    subsim_cfg = cfg.get("subsim", {})
    tau = subsim_cfg.get("tau", 0.95)
    ks = subsim_cfg.get("ks", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    lm = HookedLM(cfg["model"]["id"], dtype=cfg["model"]["dtype"])
    layer = cfg["model"]["layer"]
    batch = cfg["model"]["batch_size"]
    pos_h = lm.extract(pos_texts[: replicates * cov_n], layer, batch)
    neg_h = lm.extract(neg_texts[:neg_pool], layer, batch)
    pos_blocks = disjoint_subsets(replicates * cov_n, cov_n, replicates, seed)
    delta_values: list[Tensor] = []
    delta_bases: list[Tensor] = []
    bottom_bases: list[Tensor] = []
    pooled_bases: list[Tensor] = []
    top_bases: list[Tensor] = []
    for r in range(replicates):
        pos = pos_h[pos_blocks[r]]
        neg = neg_h[sample_indices(neg_pool, cov_n, seed + 31 * r + 1)]
        pos_cov, neg_cov = population_covariance(pos), population_covariance(neg)
        delta = delta_covariance(pos_cov, neg_cov)
        values, vectors = positive_spectrum(delta)
        delta_values.append(values)
        delta_bases.append(vectors)
        _, bottom_bases_r = torch.linalg.eigh(pos_cov)
        _, pooled_bases_r = torch.linalg.eigh(pos_cov + neg_cov)
        _, top_bases_r = torch.linalg.eigh(second_moment(pos.double()))
        bottom_bases.append(bottom_bases_r)
        pooled_bases.append(pooled_bases_r)
        top_bases.append(top_bases_r)
    out: dict[str, object] = {"tau": tau, "ks": ks}
    mean, matrix, widths = tau_pairwise(delta_values, delta_bases, tau)
    out["delta"] = {
        "K": widths,
        "n_positive": [int(v.numel()) for v in delta_values],
        "tau_subsim_mean": mean,
        **family_report(delta_bases, ks, "positive"),
    }
    log_path = Path(cfg["io"]["logs"]) / "subsim.json"
    save_json(log_path, out)
    tee("delta: completed")
    method_k = spectrum_k(cfg["model"]["hidden_size"], cfg["methods"]["lowvar_frac"])
    family_specs: tuple[
        tuple[str, list[Tensor], Literal["top", "bottom", "positive"]], ...
    ] = (
        ("cov_bottom", bottom_bases, "bottom"),
        ("pooled_bottom", pooled_bases, "bottom"),
        ("moment_top", top_bases, "top"),
    )
    for name, bases, position in family_specs:
        out[name] = family_report(bases, ks, position, pairwise_k=method_k)
        save_json(log_path, out)
        tee(f"{name}: completed (pairwise at k={method_k})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    subsim_cfg = cfg.get("subsim", {})
    tau = subsim_cfg.get("tau", 0.95)
    ks = subsim_cfg.get("ks", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    tee = Tee(Path(cfg["io"]["logs"]) / "subsim.log")
    if args.dry_run:
        tee(
            json.dumps(
                {
                    "experiment": "subsim",
                    "tau": tau,
                    "ks": ks,
                    "replicates": cfg["stability"]["replicates"],
                    "families": [
                        "delta",
                        "cov_bottom",
                        "pooled_bottom",
                        "moment_top",
                    ],
                },
                indent=2,
            )
        )
        return
    result = run(cfg, tee)
    save_subsim({**result, "ks": ks}, Path(cfg["io"]["figs"]) / "subsim.pdf")
    tee(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
