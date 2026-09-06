import argparse
import copy
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch

from robustdim.data import (
    disjoint_subsets,
    load_class_prompts,
    load_harmbench_contextual,
    load_mmlu_pro,
    sample_indices,
)
from robustdim.directions import construct, dim, unit
from robustdim.evaluate import (
    SafetyJudge,
    degenerate_rate,
    format_contextual,
    format_mmlu,
    harmbench_safety,
    mmlu_accuracy,
)
from robustdim.metrics import pairwise_stability
from robustdim.model import HookedLM
from robustdim.plots import save_sweep, save_tradeoff, save_variance
from robustdim.report import Tee, save_json
from robustdim import tradeoff
from robustdim.config import load_config


DEFAULTS = {
    "layers": [14, 17, 20, 23],
    "alphas": [10, 20, 30, 40],
    "screen_methods": ["dim", "moment_proj", "moment"],
    "hb_screen_n": 50,
    "top_per_method": 2,
    "mmlu_penalty": 1.0,
    "fractions": [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "frac_methods": ["moment_proj", "lowvar"],
    "alpha_neighbors": 1,
    "batch_size": 4,
}


def _sweep_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULTS, **cfg.get("sweep", {})}


def screen_grid(
    layers: list[int], alphas: list[float], methods: list[str]
) -> list[dict[str, Any]]:
    return [
        {"method": method, "layer": layer, "alpha": alpha}
        for method in methods
        for layer in layers
        for alpha in alphas
    ]


def select_top_per_method(
    records: list[dict[str, Any]], k: int
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["method"]), []).append(record)
    return {
        method: sorted(
            rows,
            key=lambda r: (
                r.get("score", -float("inf")),
                r.get("safety", -float("inf")),
                -r.get("degenerate", float("inf")),
            ),
            reverse=True,
        )[:k]
        for method, rows in groups.items()
    }


def local_alphas(
    alphas: Sequence[int | float], idx: int, neighbors: int
) -> list[int | float]:
    lo = max(0, idx - neighbors)
    hi = min(len(alphas), idx + neighbors + 1)
    return list(alphas[lo:hi])


def stage_counts(config: dict[str, Any]) -> dict[str, int]:
    sw: dict[str, Any] = {**DEFAULTS, **config}
    methods = sw["screen_methods"]
    screen = len(sw["layers"]) * len(sw["alphas"]) * len(methods)
    verify = (
        min(sw["top_per_method"], len(sw["layers"]) * len(sw["alphas"])) * len(methods)
        + 1
    )
    return {
        "screen": screen,
        "verify": verify,
        "fraction": len(sw["fractions"]) * len(sw["frac_methods"]),
    }


def variance_record(
    method: str,
    frac: float,
    local: dict[Any, dict[str, float]],
    mmlu: float,
    stability_score: float,
    cos_dim: float,
    subsim_dim: float | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "frac": frac,
        "local_alphas": local,
        "mmlu": mmlu,
        "stability": stability_score,
        "cos_dim": cos_dim,
        "subsim_dim": cos_dim * cos_dim if subsim_dim is None else subsim_dim,
    }


def choose_shared(verify_records: list[dict[str, Any]]) -> tuple[Any, Any, float]:
    best = max(
        verify_records,
        key=lambda r: (r["final_score"], r.get("safety", 0), -r.get("degenerate", 0)),
    )
    return best["layer"], best["alpha"], best["final_score"]


def select_fracs(
    records: list[dict[str, Any]],
    baseline: float,
    penalty: float,
    shared_alpha: Any | None = None,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        keys = list(row["local_alphas"])
        alpha = shared_alpha if shared_alpha is not None else keys[len(keys) // 2]
        safety = row["local_alphas"][alpha]["safety"]
        row = {**row, "selection_score": safety - penalty * abs(row["mmlu"] - baseline)}
        groups.setdefault(row["method"], []).append(row)
    return {
        method: max(rows, key=lambda r: r["selection_score"])
        for method, rows in groups.items()
    }


def _direction(
    cfg: dict[str, Any],
    pos_h: torch.Tensor,
    neg_h: torch.Tensor,
    method: str,
    frac: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cov_n, dim_n = cfg["data"]["cov_n"], cfg["data"]["dim_n"]
    seed = cfg["stability"]["seed"]
    d = dim(
        pos_h[sample_indices(cov_n, dim_n, seed)],
        neg_h[sample_indices(cov_n, dim_n, seed + 1)],
    )
    methods = cfg["methods"]
    f = frac if frac is not None else methods.get("lowvar_frac", 0.01)
    raw = construct(
        method,
        pos_h,
        neg_h if method in ("lda", "ridge") else None,
        d,
        ridge_gamma=methods.get("ridge_gamma", 1.0),
        lowvar_frac=f,
        moment_k_frac=frac if frac is not None else methods.get("moment_k_frac", 0.01),
    )
    return unit(raw), pos_h.norm(dim=1).mean()


def _metrics(
    judge: SafetyJudge, rows: list[dict[str, str]], outputs: list[str]
) -> tuple[float, float]:
    verdicts = [
        judge.verdict(r["behavior"], text, r["context"])
        for r, text in zip(rows, outputs, strict=True)
    ]
    return harmbench_safety(verdicts), degenerate_rate(verdicts)


def run(cfg: dict[str, Any], tee: Tee) -> dict[str, Any]:
    sw = _sweep_cfg(cfg)
    logs = Path(cfg["io"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    sweep_log = logs / "sweep.json"
    hb = load_harmbench_contextual(sw["hb_screen_n"])
    hb_prompts = [format_contextual(r["behavior"], r["context"]) for r in hb]
    judge = SafetyJudge(
        cfg["eval"]["judge_model"],
        project=cfg["eval"].get("judge_project"),
        location=cfg["eval"].get("judge_location", "global"),
    )
    lm = HookedLM(cfg["model"]["id"], dtype=cfg["model"]["dtype"])
    pos_texts = load_class_prompts(cfg["data"]["benign"])
    neg_texts = load_class_prompts(cfg["data"]["harmful"])
    result: dict[str, Any] = {
        "experiment": "sweep",
        "screen": [],
        "verify": [],
        "method_best": {},
        "shared": {},
        "variance": [],
        "selected_fracs": {},
    }
    directions: dict[int, dict[str, torch.Tensor]] = {}
    norms: dict[int, torch.Tensor] = {}
    for layer in sw["layers"]:
        pos = lm.extract(
            pos_texts[: cfg["data"]["cov_n"]],
            layer,
            cfg["model"].get("batch_size", sw["batch_size"]),
        )
        neg = lm.extract(
            neg_texts[: cfg["data"]["cov_n"]],
            layer,
            cfg["model"].get("batch_size", sw["batch_size"]),
        )
        directions[layer] = {}
        norms[layer] = pos.norm(dim=1).mean()
        for method in sw["screen_methods"]:
            directions[layer][method], _ = _direction(cfg, pos, neg, method)
        for condition in [
            r for r in screen_grid([layer], sw["alphas"], sw["screen_methods"])
        ]:
            out = lm.generate_batch(
                hb_prompts,
                layer,
                directions[layer][condition["method"]] * norms[layer],
                condition["alpha"],
                cfg["steering"]["max_new_tokens_harmbench"],
                sw["batch_size"],
            )
            safety, deg = _metrics(judge, hb, out)
            result["screen"].append(
                {
                    **condition,
                    "safety": safety,
                    "degenerate": deg,
                    "score": safety - deg,
                }
            )
            save_json(sweep_log, result)
            tee(
                f"screen {condition['method']} layer={layer} alpha={condition['alpha']}"
            )
    top = select_top_per_method(result["screen"], sw["top_per_method"])
    mmlu = load_mmlu_pro(
        cfg["eval"]["mmlu_split"], cfg["eval"].get("mmlu_n"), cfg["stability"]["seed"]
    )
    mmlu_prompts = [format_mmlu(r["question"], r["options"]) for r in mmlu]
    baseline = mmlu_accuracy(
        mmlu,
        lm.generate_batch(
            mmlu_prompts,
            cfg["model"]["layer"],
            None,
            0,
            cfg["steering"]["max_new_tokens_mmlu"],
            sw["batch_size"],
        ),
    )
    for method, candidates in top.items():
        result["method_best"][method] = None
        for candidate in candidates:
            mmlu_out = lm.generate_batch(
                mmlu_prompts,
                candidate["layer"],
                directions[candidate["layer"]][method] * norms[candidate["layer"]],
                candidate["alpha"],
                cfg["steering"]["max_new_tokens_mmlu"],
                sw["batch_size"],
            )
            row = {
                **candidate,
                "mmlu": mmlu_accuracy(mmlu, mmlu_out),
                "final_score": candidate["score"]
                - sw["mmlu_penalty"] * abs(mmlu_accuracy(mmlu, mmlu_out) - baseline),
            }
            result["verify"].append(row)
            result["method_best"][method] = max(
                result["method_best"][method] or row,
                row,
                key=lambda r: r["final_score"],
            )
            save_json(sweep_log, result)
    shared_layer, shared_alpha, shared_score = choose_shared(result["verify"])
    result["shared"] = {
        "layer": shared_layer,
        "alpha": shared_alpha,
        "final_score": shared_score,
        "baseline_mmlu": baseline,
    }
    save_json(sweep_log, result)
    rep = cfg["stability"]["replicates"]
    cov_n = cfg["data"]["cov_n"]
    dim_n = cfg["data"]["dim_n"]
    seed = cfg["stability"]["seed"]
    pool = min(len(pos_texts), cfg["stability"]["pool_n"])
    if pool < rep * cov_n:
        raise ValueError(f"need {rep}x{cov_n} disjoint positives, pool is {pool}")
    pos_big = lm.extract(pos_texts[:pool], shared_layer, sw["batch_size"])
    neg_pool = min(len(neg_texts), max(cov_n, rep * dim_n))
    neg_h = lm.extract(neg_texts[:neg_pool], shared_layer, sw["batch_size"])
    blocks = disjoint_subsets(pool, cov_n, rep, seed)
    neg_blocks = disjoint_subsets(neg_pool, dim_n, rep, seed + 2)
    dim_vecs = [
        unit(dim(pos_big[blocks[r][:dim_n]], neg_h[neg_blocks[r]])) for r in range(rep)
    ]
    alpha_idx = sw["alphas"].index(shared_alpha)
    alphas = local_alphas(sw["alphas"], alpha_idx, sw["alpha_neighbors"])
    variance_log = logs / "variance.json"
    variance_tee = Tee(logs / "variance.log")
    for method in sw["frac_methods"]:
        for frac in sw["fractions"]:
            vecs = [
                unit(
                    construct(
                        method,
                        pos_big[blocks[r]],
                        None,
                        dim(pos_big[blocks[r][:dim_n]], neg_h[neg_blocks[r]]),
                        lowvar_frac=frac,
                        moment_k_frac=frac,
                    )
                )
                for r in range(rep)
            ]
            stability_score = pairwise_stability(vecs)
            cos_score = (
                sum(
                    abs(float(unit(a.double()) @ unit(b.double())))
                    for a, b in zip(vecs, dim_vecs, strict=True)
                )
                / rep
            )
            subsim_score = (
                sum(
                    float(unit(a.double()) @ unit(b.double())) ** 2
                    for a, b in zip(vecs, dim_vecs, strict=True)
                )
                / rep
            )
            d, norm = _direction(cfg, pos_big[:cov_n], neg_h[:cov_n], method, frac)
            local = {}
            for alpha in alphas:
                safety, deg = _metrics(
                    judge,
                    hb,
                    lm.generate_batch(
                        hb_prompts,
                        shared_layer,
                        d * norm,
                        alpha,
                        cfg["steering"]["max_new_tokens_harmbench"],
                        sw["batch_size"],
                    ),
                )
                local[alpha] = {"safety": safety, "degenerate": deg}
            mm = mmlu_accuracy(
                mmlu,
                lm.generate_batch(
                    mmlu_prompts,
                    shared_layer,
                    d * norm,
                    shared_alpha,
                    cfg["steering"]["max_new_tokens_mmlu"],
                    sw["batch_size"],
                ),
            )
            row = variance_record(
                method,
                frac,
                local,
                mm,
                stability_score,
                cos_score,
                subsim_score,
            )
            result["variance"].append(row)
            save_json(variance_log, result["variance"])
            save_json(sweep_log, result)
            variance_tee(f"{method} frac={frac}")
    result["selected_fracs"] = select_fracs(
        result["variance"], baseline, sw["mmlu_penalty"], shared_alpha
    )
    save_json(sweep_log, result)
    cfg2 = copy.deepcopy(cfg)
    cfg2["model"]["layer"] = shared_layer
    cfg2["steering"]["alpha"] = shared_alpha
    cfg2["methods"]["moment_k_frac"] = result["selected_fracs"]["moment_proj"]["frac"]
    cfg2["methods"]["lowvar_frac"] = result["selected_fracs"]["lowvar"]["frac"]
    final = tradeoff.run(cfg2, tee)
    save_tradeoff(
        {k: (v["safety"], v["mmlu"]) for k, v in final.items()},
        Path(cfg["io"]["figs"]) / "tradeoff.pdf",
    )
    result["final"] = {
        "layer": shared_layer,
        "alpha": shared_alpha,
        "moment_proj_frac": cfg2["methods"]["moment_k_frac"],
        "lowvar_frac": cfg2["methods"]["lowvar_frac"],
    }
    save_json(sweep_log, result)
    save_sweep(
        result["screen"], result["verify"], Path(cfg["io"]["figs"]) / "sweep.pdf"
    )
    save_variance(
        result["variance"], Path(cfg["io"]["figs"]) / "variance.pdf", shared_alpha
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tee = Tee(Path(cfg["io"]["logs"]) / "sweep.log")
    sw = _sweep_cfg(cfg)
    if args.dry_run:
        tee(
            json.dumps(
                {
                    "experiment": "sweep",
                    **{
                        k: sw[k]
                        for k in ("layers", "alphas", "screen_methods", "fractions")
                    },
                    "stage_counts": stage_counts(sw),
                },
                indent=2,
            )
        )
        return
    run(cfg, tee)


if __name__ == "__main__":
    main()
