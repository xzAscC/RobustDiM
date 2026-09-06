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
    families = ("delta", "cov_bottom", "pooled_bottom", "moment_top")
    ks = [int(k) for k in data["ks"]]
    fig = plt.figure(figsize=(14, 6))
    grid = fig.add_gridspec(2, len(families))
    curve_ax = fig.add_subplot(grid[0, :])
    for name in families:
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
    for i, name in enumerate(families):
        matrix = data[name]["pairwise"]
        ax = fig.add_subplot(grid[1, i])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(name)
        ax.set_xticks(range(len(matrix)))
        ax.set_yticks(range(len(matrix)))
        for row_i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(
                    j, row_i, f"{value:.2f}", ha="center", va="center", color="white"
                )
    if image is not None:
        fig.colorbar(image, ax=fig.axes[1:], shrink=0.8)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_sweep(
    screen_records: list[dict[str, Any]],
    verify_records: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (safety_ax, deg_ax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for row in screen_records:
        groups.setdefault((row["method"], row["layer"]), []).append(row)
    for (method, layer), rows in groups.items():
        rows.sort(key=lambda r: r["alpha"])
        label = f"{method}, L{layer}"
        safety_ax.plot(
            [r["alpha"] for r in rows],
            [r["safety"] for r in rows],
            marker="o",
            label=label,
        )
        deg_ax.plot(
            [r["alpha"] for r in rows],
            [r["degenerate"] for r in rows],
            marker="o",
            linestyle="--",
            label=label,
        )
    for row in verify_records:
        safety_ax.scatter(
            row["alpha"], row["safety"], facecolors="none", edgecolors="black", zorder=5
        )
        deg_ax.scatter(
            row["alpha"],
            row["degenerate"],
            facecolors="none",
            edgecolors="black",
            zorder=5,
        )
    safety_ax.set_ylabel("safety")
    safety_ax.set_ylim(0, 1)
    safety_ax.set_title("Staged steering sweep (circles: verified)")
    deg_ax.set_xlabel("alpha")
    deg_ax.set_ylabel("degenerate")
    deg_ax.set_ylim(0, 1)
    deg_ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_variance(
    variance_records: list[dict[str, Any]],
    path: Path,
    shared_alpha: Any | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (tradeoff_ax, stability_ax) = plt.subplots(1, 2, figsize=(10, 4))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in variance_records:
        groups.setdefault(row["method"], []).append(row)
    for method, rows in groups.items():
        rows.sort(key=lambda r: r["frac"])
        x = [r["frac"] for r in rows]
        first = rows[0]["local_alphas"]
        alpha = shared_alpha if shared_alpha in first else next(iter(first))
        tradeoff_ax.plot(
            x,
            [r["local_alphas"][alpha]["safety"] for r in rows],
            marker="o",
            label=method,
        )
        stability_ax.plot(x, [r["stability"] for r in rows], marker="o", label=method)
    tradeoff_ax.set_xscale("log")
    tradeoff_ax.set_xlabel("variance fraction")
    tradeoff_ax.set_ylabel("safety at shared alpha")
    tradeoff_ax.set_ylim(0, 1)
    stability_ax.set_xscale("log")
    stability_ax.set_xlabel("variance fraction")
    stability_ax.set_ylabel("pairwise stability")
    stability_ax.set_ylim(0, 1)
    stability_ax.legend(frameon=False)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
