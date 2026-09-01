import argparse
import json
from pathlib import Path

import torch

from robustdim.config import load_config
from robustdim.data import load_class_prompts, sample_indices
from robustdim.directions import METHODS, construct, unit
from robustdim.metrics import pairwise_stability
from robustdim.model import HookedLM
from robustdim.plots import save_stability_bar


def _n_for(name: str, cfg: dict) -> int:
    return cfg["data"]["dim_n"] if name == "dim" else cfg["data"]["cov_n"]


def run(cfg: dict) -> dict[str, float]:
    pos_texts = load_class_prompts(cfg["data"]["benign"])
    neg_texts = load_class_prompts(cfg["data"]["harmful"])
    pool = min(len(pos_texts), len(neg_texts), cfg["stability"]["pool_n"])
    if pool < cfg["data"]["cov_n"]:
        raise ValueError(f"need {cfg['data']['cov_n']} examples, got {pool}")
    lm = HookedLM(cfg["model"]["id"], dtype=cfg["model"]["dtype"])
    layer = cfg["model"]["layer"]
    pos_h = lm.extract(pos_texts[:pool], layer, cfg["model"]["batch_size"])
    neg_h = lm.extract(neg_texts[:pool], layer, cfg["model"]["batch_size"])

    scores: dict[str, float] = {}
    out_dir = Path(cfg["io"]["checkpoints"])
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = cfg["methods"]
    for name in METHODS:
        vecs = []
        n = _n_for(name, cfg)
        for r in range(cfg["stability"]["replicates"]):
            idx_p = sample_indices(pool, n, seed=cfg["stability"]["seed"] + 17 * r)
            idx_n = sample_indices(pool, n, seed=cfg["stability"]["seed"] + 31 * r + 1)
            v = unit(
                construct(
                    name,
                    pos_h[idx_p],
                    neg_h[idx_n],
                    ridge_gamma=methods["ridge_gamma"],
                    lowvar_frac=methods["lowvar_frac"],
                    moment_k_frac=methods["moment_k_frac"],
                )
            )
            torch.save(v, out_dir / f"{name}_r{r}.pt")
            vecs.append(v)
        scores[name] = pairwise_stability(vecs)
    return scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        print(
            json.dumps({"experiment": "stability", "methods": list(METHODS)}, indent=2)
        )
        return
    scores = run(cfg)
    log = Path(cfg["io"]["logs"]) / "stability.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(scores, indent=2))
    save_stability_bar(scores, Path(cfg["io"]["figs"]) / "stability.pdf")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
