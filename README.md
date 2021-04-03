# α-Geodesical Skew Divergence

[![GitHub license](https://img.shields.io/github/license/nocotan/geodesical_skew_divergence)](https://github.com/nocotan/geodesical_skew_divergence/blob/main/LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/nocotan/geodesical_skew_divergence)
[![PyPI](https://img.shields.io/pypi/v/gs-divergence)](https://pypi.org/project/gs-divergence/)
[![arXiv](http://img.shields.io/badge/math.IT-arXiv%3A2103.17060-B31B1B.svg)](https://arxiv.org/abs/2103.17060)
![GitHub Repo stars](https://img.shields.io/github/stars/nocotan/geodesical_skew_divergence?style=social)

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/nocotan/geodesical_skew_divergence/Run%20Python%20Tests)
![GitHub issues](https://img.shields.io/github/issues/nocotan/geodesical_skew_divergence)

Official PyTorch Implementation of "[α-Geodesical Skew Divergence](https://arxiv.org/abs/2103.17060)".

[[arXiv](https://arxiv.org/abs/2103.17060)]

> The asymmetric skew divergence smooths one of the distributions by mixing it, to a degree determined by the parameter λ, with the other distribution. Such divergence is an approximation of the KL divergence that does not require the target distribution to be absolutely continuous with respect to the source distribution. In this paper, an information geometric generalization of  the skew divergence called the  α-geodesical skew divergence is proposed, and its properties are studied.

## Installation

### From PyPi

```bash
$ pip install gs_divergence
```

### From GitHub

```bash
$ git clone https://github.com/nocotan/geodesical_skew_divergence
$ python setup.py install
```

## Usage

### Compute divergence from two Tensors

```python
import torch
from gs_divergence import gs_div

a = torch.Tensor([0.1, 0.2, 0.3, 0.4])
b = torch.Tensor([0.2, 0.2, 0.4, 0.2])

div = gs_div(a, b, alpha=-1, lmd=0.5)
```

### Compute gradients

```python
import torch
from gs_divergence import gs_div

a = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
b = torch.tensor([0.2, 0.2, 0.4, 0.2])

div = gs_div(a, b, alpha=-1, lmd=0.5)
dif.backward()
```

| parameter | description                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input     | Tensor of arbitrary shape                                                                                                                                                                                                                                                                                                                                                                                                                |
| target    | Tensor of the same shape as input                                                                                                                                                                                                                                                                                                                                                                                                        |
| alpha     | Specifies the coordinate systems which equiped the geodesical skew divergence (default=``-1``)                                                                                                                                                                                                                                                                                                                                               |
| lmd       | Specifies the position on the geodesic (default=``0.5``)                                                                                                                                                                                                                                                                                                                                                                                     |
| reduction | Specifies the reduction to apply to the output:             ``'none'`` \| ``'batchmean'`` \| ``'sum'`` \| ``'mean'``.             ``'none'``: no reduction will be applied             ``'batchmean``': the sum of the output will be divided by the batchsize             ``'sum'``: the output will be summed             ``'mean'``: the output will be divided by the number of elements in the output             default=``'sum'`` |


## Definition of α-Geodesical Skew Divergence

![](./assets/def_interpolation.png)

![](./assets/def_gs_divergence.png)



## Visualizations of the α-Geodesical Skew Divergence

### Monotonicity of the α-geodesical skew divergence with respect to α

![](./assets/gs_divergence.png)

### Continuity of the α-geodesical skew divergence with respect to α and λ.

![](./assets/gs_divergence_surface.png)

## Citation

```bibtex
@misc{kimura2021geodesical,
    title={$α$-Geodesical Skew Divergence},
    author={Masanari Kimura and Hideitsu Hino},
    year={2021},
    eprint={2103.17060},
    archivePrefix={arXiv},
    primaryClass={cs.IT}
}
```
