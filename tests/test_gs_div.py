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

    def test_continuity(self):
        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])

        g_0 = gs_div(a, b, alpha=0, lmd=0.5)
        g_1 = gs_div(a, b, alpha=1, lmd=0.5)

        self.assertTrue(g_1 > g_0)

    def test_asymmetry(self):
        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])
        g_0 = gs_div(a, b, alpha=0, lmd=0.5)
        g_1 = gs_div(b, a, alpha=0, lmd=0.5)

        self.assertTrue(~torch.equal(g_0, g_1))

    def test_non_centrosymmetricicy(self):
        lmd = 0.2
        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])
        g_0 = gs_div(a, b, alpha=0, lmd=lmd)
        g_1 = gs_div(a, b, alpha=0, lmd=1-lmd)

        self.assertTrue(~torch.equal(g_0, g_1))
