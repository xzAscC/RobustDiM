from pathlib import Path
from typing import Any

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


def save_subsim(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ks = [int(k) for k in data["ks"]]
    fig = plt.figure(figsize=(11, 6))
    grid = fig.add_gridspec(2, 3)
    curve_ax = fig.add_subplot(grid[0, :])
    for name in ("delta", "cov_bottom", "moment_top"):
        family = data[name]
        curve = family["curve"]
        points = [(k, curve[str(k)]) for k in ks if curve[str(k)] is not None]
        if points:
            curve_ax.plot(*zip(*points), marker="o", label=name)
    curve_ax.set_xscale("log", base=2)
    curve_ax.set_ylim(0, 1)
    curve_ax.set_xlabel("eigenspace width k")
    curve_ax.set_ylabel("SubSim")
    curve_ax.set_title("Rank-resolved covariance SubSim")
    curve_ax.legend(frameon=False)
    image = None
    for name in ("delta", "cov_bottom", "moment_top"):
        matrix = data[name]["pairwise"]
        ax = fig.add_subplot(grid[1, ("delta", "cov_bottom", "moment_top").index(name)])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(name)
        ax.set_xticks(range(len(matrix)))
        ax.set_yticks(range(len(matrix)))
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")
    if image is not None:
        fig.colorbar(image, ax=fig.axes[1:], shrink=0.8)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
