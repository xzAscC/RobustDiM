from collections.abc import Sequence

import torch
from torch import Tensor

from robustdim.directions import unit


def pairwise_stability(vectors: Sequence[Tensor]) -> float:
    units = [unit(v.double()) for v in vectors]
    total = 0.0
    n = 0
    for i, a in enumerate(units):
        for b in units[i + 1 :]:
            total += abs(float(a @ b))
            n += 1
    return total / n


def subspace_similarity(u_a: Tensor, u_b: Tensor) -> float:
    if u_a.ndim != 2 or u_b.ndim != 2:
        raise ValueError("Bases must be rank-2 tensors")
    if u_a.shape != u_b.shape:
        raise ValueError("Bases must have matching shape (d, k)")
    k = int(u_a.shape[1])
    if k == 0:
        return 0.0

    overlap = u_a.to(dtype=torch.float64).T @ u_b.to(dtype=torch.float64)
    return float((overlap.square().sum() / k).item())


def positive_spectrum(sigma: Tensor) -> tuple[Tensor, Tensor]:
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {tuple(sigma.shape)}")

    values, vectors = torch.linalg.eigh(sigma.to(dtype=torch.float64))
    values = torch.flip(values, dims=(0,))
    vectors = torch.flip(vectors, dims=(1,))
    positive = values > 0
    return values[positive], vectors[:, positive]


def spectral_k(values: Tensor, tau: float = 0.95) -> int:
    positive = values.to(dtype=torch.float64)
    positive = positive[positive > 0]
    if positive.numel() == 0:
        raise ValueError("Expected at least one positive eigenvalue")

    mass = positive.square()
    threshold = mass.sum() * tau
    return int(torch.searchsorted(mass.cumsum(0), threshold).item()) + 1


def truncated_subsim(u_a: Tensor, u_b: Tensor) -> float:
    if u_a.ndim != 2 or u_b.ndim != 2:
        raise ValueError("Bases must be rank-2 tensors")
    k = min(int(u_a.shape[1]), int(u_b.shape[1]))
    if k == 0:
        return 0.0
    return subspace_similarity(u_a[:, :k], u_b[:, :k])
