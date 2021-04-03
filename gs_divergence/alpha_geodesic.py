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

    a_ = a + 1e-12
    b_ = b + 1e-12
    if alpha == 1:
        return torch.exp((1 - lmd) * torch.log(a_) + lmd * torch.log(b_))
    elif alpha >= 1e+9:
        return torch.min(a_, b_)
    elif alpha <= -1e+9:
        return torch.max(a_, b_)
    else:
        p = (1 - alpha) / 2
        lhs = a_ ** p
        rhs = b_ ** p
        g = ((1 - lmd) * lhs + lmd * rhs) ** (1/p)

        if alpha > 0 and (g == 0).sum() > 0:
            return torch.min(a_, b_)

        return g
