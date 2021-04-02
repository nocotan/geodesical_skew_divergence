import torch


def alpha_geodesic(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    lmd: float
) -> torch.Tensor:
    r"""
    $\alpha$-geodesic between two probability distributions
    """
    if alpha == 1:
        return torch.exp((1 - lmd) * torch.log(a) + lmd * torch.log(b))
    elif alpha >= 1e+9:
        return torch.min(a, b)
    elif alpha <= -1e+9:
        return torch.max(a, b)
    else:
        p = (1 - alpha) / 2
        g = ((1 - lmd) * (a ** p) + lmd * (b ** p)) ** (1/p)
        g[g == float('inf')] = 0

        return g
