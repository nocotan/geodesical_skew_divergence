from typing import Optional
import torch

from gs_divergence.alpha_geodesic import alpha_geodesic


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

    skew_target = alpha_geodesic(input, target, alpha=alpha, lmd=lmd)
    div = input * torch.log(input / skew_target + 1e-12)
    if reduction == 'batchmean':
        div = div.sum() / input.size()[0]
    elif reduction == 'sum':
        div = div.sum()
    elif reduction == 'mean':
        div = div.mean()

    return div
