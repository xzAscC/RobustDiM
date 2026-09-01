import argparse
import json
from pathlib import Path

import torch

from robustdim.config import load_config
from robustdim.data import disjoint_subsets, load_class_prompts, sample_indices
from robustdim.directions import METHODS, construct, dim, unit
from robustdim.metrics import pairwise_stability
from robustdim.model import HookedLM
from robustdim.plots import save_stability_bar
from robustdim.report import Tee, save_json


def run(cfg: dict, tee: Tee) -> dict[str, float]:
    pos_texts = load_class_prompts(cfg["data"]["benign"])
    neg_texts = load_class_prompts(cfg["data"]["harmful"])
    replicates = cfg["stability"]["replicates"]
    dim_n = cfg["data"]["dim_n"]
    cov_n = cfg["data"]["cov_n"]
    seed = cfg["stability"]["seed"]
    pos_pool = min(len(pos_texts), cfg["stability"]["pool_n"])
    neg_pool = min(len(neg_texts), max(cov_n, replicates * dim_n))
    if pos_pool < replicates * cov_n:
        raise ValueError(
            f"need {replicates}x{cov_n} disjoint positives, pool is {pos_pool}"
        )
    if neg_pool < max(cov_n, replicates * dim_n):
        raise ValueError(
            f"need max({cov_n}, {replicates}x{dim_n}) negatives, pool is {neg_pool}"
        )

    lm = HookedLM(cfg["model"]["id"], dtype=cfg["model"]["dtype"])
    layer = cfg["model"]["layer"]
    pos_h = lm.extract(pos_texts[:pos_pool], layer, cfg["model"]["batch_size"])
    neg_h = lm.extract(neg_texts[:neg_pool], layer, cfg["model"]["batch_size"])

    pos_blocks = disjoint_subsets(pos_pool, cov_n, replicates, seed)
    neg_dim_blocks = disjoint_subsets(neg_pool, dim_n, replicates, seed + 2)

    scores: dict[str, float] = {}
    out_dir = Path(cfg["io"]["checkpoints"])
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = cfg["methods"]
    scores_log = Path(cfg["io"]["logs"]) / "stability.json"
    for name in METHODS:
        vecs = []
        for r in range(replicates):
            d = dim(pos_h[pos_blocks[r][:dim_n]], neg_h[neg_dim_blocks[r]])
            neg_idx = (
                sample_indices(neg_pool, cov_n, seed + 31 * r + 1)
                if name in ("lda", "ridge")
                else None
            )
            v = unit(
                construct(
                    name,
                    pos_h[pos_blocks[r]],
                    None if neg_idx is None else neg_h[neg_idx],
                    d,
                    ridge_gamma=methods["ridge_gamma"],
                    lowvar_frac=methods["lowvar_frac"],
                    moment_k_frac=methods["moment_k_frac"],
                )
            )
            torch.save(v, out_dir / f"{name}_r{r}.pt")
            vecs.append(v)
        scores[name] = pairwise_stability(vecs)
        tee(f"{name}: stability={scores[name]:.4f}")
        save_json(scores_log, scores)
    return scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    tee = Tee(Path(cfg["io"]["logs"]) / "stability.log")
    if args.dry_run:
        tee(json.dumps({"experiment": "stability", "methods": list(METHODS)}, indent=2))
        return
    scores = run(cfg, tee)
    save_stability_bar(scores, Path(cfg["io"]["figs"]) / "stability.pdf")
    tee(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
