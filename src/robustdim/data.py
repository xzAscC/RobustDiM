import random
from typing import Any


def sample_indices(n_total: int, n: int, seed: int) -> list[int]:
    if n > n_total:
        raise ValueError(f"need {n} rows but only {n_total} available")
    return random.Random(seed).sample(range(n_total), n)


def _prompts(rows: Any, field: str = "prompt") -> list[str]:
    texts: list[str] = []
    for row in rows:
        prompt = row.get(field)
        if not prompt or not str(prompt).strip():
            continue
        refusal = row.get("refusal")
        response = row.get("response")
        if refusal and response and str(response).strip() == str(refusal).strip():
            continue
        texts.append(str(prompt).strip())
    return texts


def load_class_prompts(name: str) -> list[str]:
    from datasets import load_dataset

    return _prompts(load_dataset(name, split="train"))


def load_harmbench_contextual(n: int = 100) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("walledai/HarmBench", "contextual", split="train")
    rows: list[dict[str, str]] = []
    for i, row in enumerate(ds):
        item = dict(row)
        behavior = str(item.get("prompt") or item.get("Behavior") or "")
        context = str(item.get("context") or item.get("ContextString") or "")
        rows.append(
            {
                "behavior": behavior,
                "context": context,
                "id": str(item.get("BehaviorID") or item.get("id") or i),
            }
        )
        if len(rows) == n:
            break
    return rows


def load_mmlu_pro(
    split: str = "test", n: int | None = None, seed: int = 0
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    total = len(ds)
    # rows are grouped by category, so subsample instead of slicing
    idx = sample_indices(total, n, seed) if n is not None else range(total)
    return [dict(ds[i]) for i in idx]
