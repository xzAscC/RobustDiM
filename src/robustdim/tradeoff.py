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
from robustdim.report import Tee, save_json


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


def run(cfg: dict, tee: Tee) -> dict[str, dict[str, float]]:
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

    def gen_all(label: str, prompts: list[str], direction, scale: int, tokens: int):
        outs = []
        for i, user in enumerate(prompts):
            outs.append(lm.generate(user, layer, direction, scale, tokens))
            if (i + 1) % 10 == 0:
                tee(f"{label}: {i + 1}/{len(prompts)} generated")
        return outs

    gens: dict[str, dict[str, list[str]]] = {}
    gens_log = Path(cfg["io"]["logs"]) / "tradeoff_generations.json"
    for label, direction, scale in settings:
        gens[label] = {
            "hb": gen_all(
                label,
                [format_contextual(r["behavior"], r["context"]) for r in hb],
                direction,
                scale,
                hb_tokens,
            ),
            "mmlu": gen_all(
                label,
                [format_mmlu(r["question"], r["options"]) for r in mmlu],
                direction,
                scale,
                mmlu_tokens,
            ),
        }
        tee(f"{label}: generation done")
        save_json(gens_log, gens)

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
    results_log = Path(cfg["io"]["logs"]) / "tradeoff.json"
    for label, g in gens.items():
        harmful = [
            judge.is_harmful(row["behavior"], text, row["context"])
            for row, text in zip(hb, g["hb"], strict=True)
        ]
        results[label] = {
            "safety": harmbench_safety(harmful),
            "mmlu": mmlu_accuracy(mmlu, g["mmlu"]),
        }
        tee(
            f"{label}: safety={results[label]['safety']:.3f} mmlu={results[label]['mmlu']:.3f}"
        )
        save_json(results_log, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    tee = Tee(Path(cfg["io"]["logs"]) / "tradeoff.log")
    if args.dry_run:
        tee(json.dumps({"experiment": "tradeoff", "methods": list(METHODS)}, indent=2))
        return
    results = run(cfg, tee)
    points = {k: (v["safety"], v["mmlu"]) for k, v in results.items()}
    save_tradeoff(points, Path(cfg["io"]["figs"]) / "tradeoff.pdf")
    tee(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
