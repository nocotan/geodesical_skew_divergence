import unittest
import torch

from gs_divergence.alpha_geodesic import alpha_geodesic


class TestAlphaGeodesic(unittest.TestCase):
    def test_alpha_minus_1(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = alpha_geodesic(a, b, alpha=-1, lmd=0.5)

        self.assertTrue(torch.equal(g, ((a+b) / 2)))

    def test_alpha_1(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = alpha_geodesic(a, b, alpha=1, lmd=0.5)
        res = torch.exp(0.5 * torch.log(a) + 0.5 * torch.log(b))

        self.assertTrue(torch.equal(g, res))

    def test_alpha_0(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = alpha_geodesic(a, b, alpha=0, lmd=0.5)
        res = (0.5 * torch.sqrt(a) + 0.5 * torch.sqrt(b))**2

        self.assertTrue(torch.equal(g, res))

    def test_alpha_3(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = alpha_geodesic(a, b, alpha=3, lmd=0.5)
        res = 1 / (0.5 * 1/a + 0.5 * 1/b)

        self.assertTrue(torch.equal(g, res))

    def test_alpha_inf(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = alpha_geodesic(a, b, alpha=float('inf'), lmd=0.5)
        res = torch.min(a, b)

        self.assertTrue(torch.equal(g, res))

    def test_alpha_minus_inf(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        g = alpha_geodesic(a, b, alpha=-float('inf'), lmd=0.5)
        res = torch.max(a, b)

        self.assertTrue(torch.equal(g, res))

    def test_value_0(self):
        a = torch.Tensor([0, 1, 2])
        b = torch.Tensor([1, 2, 3])
        g = alpha_geodesic(a, b, alpha=-1, lmd=0.5)

        self.assertTrue(torch.isinf(g).sum() == 0)

    def test_value_0_2d(self):
        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])

        g = alpha_geodesic(a, b, alpha=1, lmd=0.5)

        self.assertTrue(torch.isinf(g).sum() == 0)

    def test_value_inf(self):
        a = torch.Tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
        b = torch.Tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])

        g = alpha_geodesic(a, b, alpha=100, lmd=0.5)
        print(g)
        res = torch.min(a, b)

        self.assertTrue(torch.all(torch.isclose(g, res)))

    def test_grad(self):
        a = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]], requires_grad=True)
        b = torch.tensor([[0.4, 0.4, 0.2], [0.2, 0.1, 0.7]])

        g = alpha_geodesic(a, b, alpha=1, lmd=0.5)

        self.assertIsNotNone(g.grad_fn)
