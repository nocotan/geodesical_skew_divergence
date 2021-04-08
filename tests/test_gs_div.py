import unittest
import torch

from gs_divergence import gs_div


class TestGSDiv(unittest.TestCase):
    def test_alpha_minus_1(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = gs_div(a, b, alpha=-1, lmd=0.5)

        res = (a * torch.log(a / (0.5*a + 0.5*b))).sum()

        self.assertIsNotNone(g)
        self.assertEqual(g, res)

    def test_alpha_0(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = gs_div(a, b, alpha=0, lmd=0.5)

        res = (a * torch.log(a / (0.5*torch.sqrt(a) + 0.5*torch.sqrt(b))**2)).sum()

        self.assertIsNotNone(g)
        self.assertEqual(g, res)

    def test_alpha_1(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = gs_div(a, b, alpha=1, lmd=0.5)

        res = 0.5 * (a * torch.log(a / b)).sum()

        self.assertIsNotNone(g)
        self.assertAlmostEqual(g.item(), res.item(), 3)

    def test_value_0_2d(self):
        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])

        g = gs_div(a, b, alpha=1, lmd=0.5)

        self.assertTrue(torch.isinf(g).sum() == 0)

    def test_monotonicity(self):
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

    def test_subadditivity(self):
        alpha = 0
        beta = 1
        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])

        g_0 = gs_div(a, b, alpha=alpha, lmd=0.5)
        g_1 = gs_div(a, b, alpha=beta, lmd=0.5)
        g_2 = gs_div(a, b, alpha=alpha+beta, lmd=0.5)

        self.assertTrue(g_2 <= g_0 + g_1)

    def test_upper_bound(self):
        alpha_upper = float('inf')
        alpha_list = [-1, 0, 1, 2, 3, 4]

        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])

        g_upper = gs_div(a, b, alpha=alpha_upper, lmd=0.5)
        for alpha in alpha_list:
            g = gs_div(a, b, alpha=alpha, lmd=0.5)
            self.assertTrue(g <= g_upper)

    def test_lower_bound(self):
        alpha_lower = -float('inf')
        alpha_list = [-1, 0, 1, 2, 3, 4]

        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])

        g_lower = gs_div(a, b, alpha=alpha_lower, lmd=0.5)
        for alpha in alpha_list:
            g = gs_div(a, b, alpha=alpha_lower, lmd=0.5)
            self.assertTrue(g >= g_lower)
