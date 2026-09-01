from pathlib import Path

import matplotlib.pyplot as plt


def save_stability_bar(scores: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(scores)
    vals = [scores[k] for k in names]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(names, vals)
    ax.set_ylim(0, 1)
    ax.set_ylabel("pairwise |cosine|")
    ax.set_title("Steering-direction stability")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_tradeoff(points: dict[str, tuple[float, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (safety, mmlu) in points.items():
        ax.scatter(mmlu, safety, label=name)
        ax.annotate(name, (mmlu, safety), fontsize=8)
    ax.set_xlabel("MMLU-Pro accuracy")
    ax.set_ylabel("HarmBench safety")
    ax.set_title("Safety vs. general ability")
    ax.legend(frameon=False)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
