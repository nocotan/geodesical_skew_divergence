from typing import Optional
import torch
import torch.nn.functional as F

from gs_divergence.quasi_arithmetic_mean import quasi_arithmetic_mean


def gs_div(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = -1,
    lmd: float = 0.5,
    reduction: Optional[str] = 'sum',
) -> torch.Tensor:
    r"""The $\alpha$-geodesical skew divergence.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        alpha: Specifies the coordinate systems which equiped the geodesical skew divergence
        lmd: Specifies the position on the geodesic
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean``': the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'sum'``
    """

    assert lmd >= 0 and lmd <= 1

    skew_target = quasi_arithmetic_mean(input, target, alpha=alpha, lmd=lmd)
    div = input * torch.log(input / skew_target)
    if reduction == 'batchmean':
        div = div.sum() / input.size()[0]
    elif reduction == 'sum':
        div = div.sum()
    elif reduction == 'mean':
        div = div.mean()

    return div
