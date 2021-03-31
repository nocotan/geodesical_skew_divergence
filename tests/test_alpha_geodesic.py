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