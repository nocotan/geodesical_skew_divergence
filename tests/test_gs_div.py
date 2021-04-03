import unittest
import torch

from gs_divergence import gs_div


class TestGSDiv(unittest.TestCase):
    def test_alpha_minus_1(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = gs_div(a, b, alpha=-1, lmd=0.5)

        self.assertIsNotNone(g)

    def test_value_0_2d(self):
        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])

        g = gs_div(a, b, alpha=1, lmd=0.5)

        self.assertTrue(torch.isinf(g).sum() == 0)
