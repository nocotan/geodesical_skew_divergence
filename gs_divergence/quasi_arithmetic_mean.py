import torch


def quasi_arithmetic_mean(a: torch.Tensor, b: torch.Tensor,
                          alpha: float, lmd: float):
    if alpha == 1:
        return torch.exp((1 - lmd) * torch.log(a) + lmd * torch.log(b))
    else:
        p = (1 - alpha) / 2
        return ((1 - lmd) * (a ** p) + lmd * (b ** p)) ** (1/p)
