import torch
from torch import Tensor

COV_METHODS = ("lda", "ridge", "lowvar", "moment_proj", "moment")
METHODS = ("dim", *COV_METHODS)


def spectrum_k(width: int, frac: float) -> int:
    return max(1, round(frac * width))


def unit(v: Tensor) -> Tensor:
    return v / v.norm().clamp_min(1e-12)


def dim(pos: Tensor, neg: Tensor) -> Tensor:
    return pos.mean(0) - neg.mean(0)


def centered_cov(h: Tensor) -> Tensor:
    c = h - h.mean(0)
    return (c.T @ c) / h.shape[0]


def second_moment(h: Tensor) -> Tensor:
    return (h.T @ h) / h.shape[0]


def lda(pos: Tensor, neg: Tensor, gamma: float = 0.0) -> Tensor:
    pos64, neg64 = pos.double(), neg.double()
    w = centered_cov(pos64) + centered_cov(neg64)
    if gamma != 0.0:
        w = w + gamma * torch.eye(w.shape[0], dtype=w.dtype, device=w.device)
    delta = dim(pos64, neg64)
    try:
        v = torch.linalg.solve(w, delta)
    except RuntimeError:
        v = torch.linalg.lstsq(w, delta.unsqueeze(1)).solution.squeeze(1)
    return v.to(dtype=pos.dtype)


def ridge_lda(pos: Tensor, neg: Tensor, gamma: float = 1.0) -> Tensor:
    return lda(pos, neg, gamma=gamma)


def _project(basis: Tensor, delta: Tensor, dtype: torch.dtype) -> Tensor:
    return (basis @ (basis.T @ delta)).to(dtype=dtype)


def lowvar(pos: Tensor, neg: Tensor, frac: float = 0.01) -> Tensor:
    pos64, neg64 = pos.double(), neg.double()
    _, evecs = torch.linalg.eigh(centered_cov(pos64))
    k = spectrum_k(pos.shape[1], frac)
    return _project(evecs[:, :k], dim(pos64, neg64), pos.dtype)


def moment_proj(pos: Tensor, neg: Tensor, frac: float = 0.01) -> Tensor:
    pos64, neg64 = pos.double(), neg.double()
    _, evecs = torch.linalg.eigh(second_moment(pos64))
    k = spectrum_k(pos.shape[1], frac)
    return _project(evecs[:, -k:], dim(pos64, neg64), pos.dtype)


def moment(pos: Tensor, neg: Tensor) -> Tensor:
    pos64, neg64 = pos.double(), neg.double()
    _, evecs = torch.linalg.eigh(second_moment(pos64))
    v = evecs[:, -1]
    if torch.dot(v, dim(pos64, neg64)) < 0:
        v = -v
    return v.to(dtype=pos.dtype)


def construct(
    name: str,
    pos: Tensor,
    neg: Tensor,
    *,
    ridge_gamma: float = 1.0,
    lowvar_frac: float = 0.01,
    moment_k_frac: float = 0.01,
) -> Tensor:
    if name == "dim":
        return dim(pos, neg)
    if name == "lda":
        return lda(pos, neg)
    if name == "ridge":
        return ridge_lda(pos, neg, gamma=ridge_gamma)
    if name == "lowvar":
        return lowvar(pos, neg, frac=lowvar_frac)
    if name == "moment_proj":
        return moment_proj(pos, neg, frac=moment_k_frac)
    if name == "moment":
        return moment(pos, neg)
    raise ValueError(f"unknown method: {name}")
