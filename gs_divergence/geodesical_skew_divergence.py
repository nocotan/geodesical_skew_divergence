from typing import Optional
import torch
import torch.nn as nn

from gs_divergence.alpha_geodesic import alpha_geodesic


class GSDivLoss(nn.Module):
    r"""The alpha-geodesical skew divergence loss measure

    `alpha-geodesical skew divergence`_ is a useful distance measure for continuous
    distributions and is approximation of the 'Kullback-Leibler divergence`.

    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.
    """
    def __init__(
        self,
        alpha: float = -1,
        lmd: float = 0.5,
        reduction: Optional[str] = 'sum') -> None:

        self.alpha = alpha
        self.lmd = lmd
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:

        return gs_div(input, target, self.alpha, self.lmd, self.reduction)



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
