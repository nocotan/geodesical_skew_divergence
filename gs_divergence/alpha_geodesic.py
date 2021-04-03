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

    a += 1e-12
    b += 1e-12
    if alpha == 1:
        return torch.exp((1 - lmd) * torch.log(a) + lmd * torch.log(b))
    elif alpha >= 1e+9:
        return torch.min(a, b)
    elif alpha <= -1e+9:
        return torch.max(a, b)
    else:
        p = (1 - alpha) / 2
        lhs = a ** p
        rhs = b ** p
        g = ((1 - lmd) * lhs + lmd * rhs) ** (1/p)

        if alpha > 0 and (g == 0).sum() > 0:
            return torch.min(a, b)

        return g
