import argparse
import gc
import json
from pathlib import Path

import torch

from robustdim.config import load_config
from robustdim.data import (
    load_class_prompts,
    load_harmbench_contextual,
    load_mmlu_pro,
    sample_indices,
)
from robustdim.directions import METHODS, construct, unit
from robustdim.evaluate import (
    SafetyJudge,
    format_contextual,
    format_mmlu,
    harmbench_safety,
    mmlu_accuracy,
)
from robustdim.model import HookedLM
from robustdim.plots import save_tradeoff


def _directions(cfg: dict, lm: HookedLM) -> dict[str, torch.Tensor]:
    pos_texts = load_class_prompts(cfg["data"]["benign"])
    neg_texts = load_class_prompts(cfg["data"]["harmful"])
    cov_n = cfg["data"]["cov_n"]
    dim_n = cfg["data"]["dim_n"]
    layer = cfg["model"]["layer"]
    pos_h = lm.extract(pos_texts[:cov_n], layer, cfg["model"]["batch_size"])
    neg_h = lm.extract(neg_texts[:cov_n], layer, cfg["model"]["batch_size"])
    dim_idx = sample_indices(cov_n, dim_n, seed=cfg["stability"]["seed"])
    methods = cfg["methods"]
    out: dict[str, torch.Tensor] = {}
    for name in METHODS:
        pos = pos_h[dim_idx] if name == "dim" else pos_h
        neg = neg_h[dim_idx] if name == "dim" else neg_h
        out[name] = unit(
            construct(
                name,
                pos,
                neg,
                ridge_gamma=methods["ridge_gamma"],
                lowvar_frac=methods["lowvar_frac"],
                moment_k_frac=methods["moment_k_frac"],
            )
        )
    return out


def run(cfg: dict) -> dict[str, dict[str, float]]:
    lm = HookedLM(cfg["model"]["id"], dtype=cfg["model"]["dtype"])
    dirs = _directions(cfg, lm)
    ckpt = Path(cfg["io"]["checkpoints"])
    ckpt.mkdir(parents=True, exist_ok=True)
    for name, v in dirs.items():
        torch.save(v, ckpt / f"{name}.pt")

    hb = load_harmbench_contextual(cfg["eval"]["harmbench_n"])
    mmlu = load_mmlu_pro(
        cfg["eval"]["mmlu_split"], cfg["eval"].get("mmlu_n"), cfg["stability"]["seed"]
    )
    layer = cfg["model"]["layer"]
    alpha = cfg["steering"]["alpha"]
    hb_tokens = cfg["steering"]["max_new_tokens_harmbench"]
    mmlu_tokens = cfg["steering"]["max_new_tokens_mmlu"]
    settings = [
        ("baseline", None, 0.0),
        *((name, dirs[name], alpha) for name in METHODS),
    ]

    gens: dict[str, dict[str, list[str]]] = {}
    for label, direction, scale in settings:
        gens[label] = {
            "hb": [
                lm.generate(
                    format_contextual(row["behavior"], row["context"]),
                    layer,
                    direction,
                    scale,
                    hb_tokens,
                )
                for row in hb
            ],
            "mmlu": [
                lm.generate(
                    format_mmlu(row["question"], row["options"]),
                    layer,
                    direction,
                    scale,
                    mmlu_tokens,
                )
                for row in mmlu
            ],
        }

    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    judge = SafetyJudge(
        cfg["eval"]["judge_model"],
        project=cfg["eval"]["judge_project"],
        location=cfg["eval"]["judge_location"],
    )
    results: dict[str, dict[str, float]] = {}
    for label, g in gens.items():
        harmful = [
            judge.is_harmful(row["behavior"], text, row["context"])
            for row, text in zip(hb, g["hb"], strict=True)
        ]
        results[label] = {
            "safety": harmbench_safety(harmful),
            "mmlu": mmlu_accuracy(mmlu, g["mmlu"]),
        }
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        print(
            json.dumps({"experiment": "tradeoff", "methods": list(METHODS)}, indent=2)
        )
        return
    results = run(cfg)
    log = Path(cfg["io"]["logs"]) / "tradeoff.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(results, indent=2))
    points = {k: (v["safety"], v["mmlu"]) for k, v in results.items()}
    save_tradeoff(points, Path(cfg["io"]["figs"]) / "tradeoff.pdf")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
