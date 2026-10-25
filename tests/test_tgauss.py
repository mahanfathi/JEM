import math

import pytest
import torch

from jem.envs import tgauss

LO, HI = torch.tensor(-1.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
CASES = [(0.0, 0.35), (0.8, 0.35), (-0.95, 0.2), (1.6, 0.3), (-3.0, 0.4), (0.3, 2.0)]


def _grid(n: int = 200_001):
    x = torch.linspace(-1, 1, n, dtype=torch.float64)
    return x, x[1] - x[0]


@pytest.mark.parametrize("mu,std", CASES)
def test_density_is_normalised_and_mean_matches(mu, std):
    x, dx = _grid()
    m, s = torch.tensor(mu, dtype=torch.float64), torch.tensor(std, dtype=torch.float64)
    p = tgauss.log_prob(x[:, None], m, s, LO, HI).exp()
    assert torch.trapezoid(p, dx=dx).item() == pytest.approx(1.0, abs=1e-6)
    mean = torch.trapezoid(p * x, dx=dx).item()
    assert tgauss.mean(m, s, LO, HI).item() == pytest.approx(mean, abs=1e-6)


@pytest.mark.parametrize("mu,std", CASES)
def test_samples_follow_the_density(mu, std):
    gen = torch.Generator().manual_seed(0)
    m = torch.full((200_000,), mu, dtype=torch.float64)
    s = torch.full_like(m, std)
    a = tgauss.sample(m, s, LO, HI, gen)
    assert (a >= -1).all() and (a <= 1).all()
    expected = tgauss.mean(m[:1], s[:1], LO, HI).item()
    assert a.mean().item() == pytest.approx(expected, abs=5e-3)
    # Probability of the lower half of the box, by sampling and by the density.
    x, dx = _grid()
    p = tgauss.log_prob(x[:, None], m[0], s[0], LO, HI).exp()
    lower = torch.trapezoid(p * (x < 0), dx=dx).item()
    assert (a < 0).double().mean().item() == pytest.approx(lower, abs=5e-3)


def test_outside_the_box_has_zero_density():
    m, s = torch.zeros(1), torch.full((1,), 0.35)
    lo, hi = torch.full((1,), -1.0), torch.ones(1)
    out = tgauss.log_prob(torch.tensor([[1.2]]), m, s, lo, hi)
    assert out.item() == -math.inf
    assert tgauss.log_prob(torch.tensor([[1.0]]), m, s, lo, hi).isfinite().all()


def test_weighted_products_are_truncated_gaussians():
    """log p_i + w log p_j - log TN(product) is constant over the box (same normaliser)."""
    gen = torch.Generator().manual_seed(1)
    mus = torch.rand(2, 5, generator=gen, dtype=torch.float64) * 2 - 1
    stds = torch.full_like(mus, 0.35)
    lo, hi = torch.full((5,), -1.0, dtype=torch.float64), torch.ones(5, dtype=torch.float64)
    for w in (0.25, 1.0, 4.0):
        mu, std = tgauss.product(mus, stds, torch.tensor([1.0, w], dtype=torch.float64))
        assert torch.allclose(std, 0.35 / torch.sqrt(torch.tensor(1 + w, dtype=torch.float64)))
        a = torch.rand(64, 5, generator=gen, dtype=torch.float64) * 2 - 1
        lp = tgauss.log_prob(a, mus[0], stds[0], lo, hi) + w * tgauss.log_prob(
            a, mus[1], stds[1], lo, hi
        )
        gap = lp - tgauss.log_prob(a, mu, std, lo, hi)
        assert gap.std().item() < 1e-9


def test_float32_is_stable_in_the_tails():
    m = torch.tensor([4.0, -4.0, 0.0])
    s = torch.tensor([0.3, 0.3, 0.05])
    lo, hi = torch.full((3,), -1.0), torch.ones(3)
    assert tgauss.log_mass(m, s, lo, hi).isfinite().all()
    a = tgauss.sample(m.expand(1000, 3), s.expand(1000, 3), lo, hi)
    assert a.isfinite().all() and (a.abs() <= 1).all()
    # Mass piles up at the face nearest the mean (an exponential tail of rate ~33 here).
    assert torch.allclose(a.mean(0), tgauss.mean(m, s, lo, hi), atol=5e-3)
