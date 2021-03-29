from typing import Optional
import torch

from gs_divergence import gs_div


def symmetrized_gs_div(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = -1,
    lmd: float = 0.5,
    reduction: Optional[str] = 'sum',
) -> torch.Tensor:
    lhs = gs_div(input, target, alpha=alpha, lmd=lmd, reduction=reduction)
    rhs = gs_div(target, input, alpha=alpha, lmd=lmd, reduction=reduction)

    return (lhs + rhs) / 2
