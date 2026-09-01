from collections.abc import Sequence

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
