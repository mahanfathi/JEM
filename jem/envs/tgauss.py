"""Diagonal Gaussians truncated to an action box: exact sampling, log-density, products.

An expert that adds Gaussian noise to its mean action and then *clips* puts point masses
on the faces of the box, which no density model can match. Truncating keeps a proper
density on the box. Per dimension, with alpha = (lo - mu) / s and beta = (hi - mu) / s,

    log TN(a; mu, s) = log N(a; mu, s^2) - log(Phi(beta) - Phi(alpha)),   lo <= a <= hi.

Truncated Gaussians on a common box are closed under weighted products, because
truncation only restricts the support: prod_k TN(mu_k, s_k)^{w_k} is proportional to
TN(mu, s) with precision 1/s^2 = sum_k w_k / s_k^2 and mean mu = s^2 sum_k w_k mu_k / s_k^2.
So any weighted product of experts has an exact, normalised reference density.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.special import log_ndtr, ndtri

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def _log1mexp(x: Tensor) -> Tensor:
    """log(1 - exp(x)) for x <= 0, accurate at both ends."""
    return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def _standardized(mu: Tensor, std: Tensor, lo: Tensor, hi: Tensor) -> tuple[Tensor, Tensor]:
    return (lo - mu) / std, (hi - mu) / std


def log_mass(mu: Tensor, std: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """log(Phi(beta) - Phi(alpha)) per dimension, computed in whichever tail is accurate."""
    alpha, beta = _standardized(mu, std, lo, hi)
    # When most of the window lies above the mean, mirror it: Phi(b) - Phi(a) = Phi(-a) - Phi(-b).
    flip = alpha + beta > 0
    a, b = torch.where(flip, -beta, alpha), torch.where(flip, -alpha, beta)
    lb = log_ndtr(b)
    return lb + _log1mexp(log_ndtr(a) - lb)


def log_prob(act: Tensor, mu: Tensor, std: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """Exact log-density of `act` under the truncated diagonal Gaussian; sums the last axis."""
    z = (act - mu) / std
    per_dim = -0.5 * z.square() - std.log() - _LOG_SQRT_2PI - log_mass(mu, std, lo, hi)
    # Relative slack: a sample on the boundary can round just outside it in fp32.
    tol = 1e-6 * (hi - lo)
    inside = ((act >= lo - tol) & (act <= hi + tol)).all(-1)
    return torch.where(inside, per_dim.sum(-1), torch.full_like(inside, -math.inf, dtype=z.dtype))


def sample(
    mu: Tensor, std: Tensor, lo: Tensor, hi: Tensor, gen: torch.Generator | None = None
) -> Tensor:
    """Exact draws by the inverse CDF, in the tail where the CDF is far from 1.

    The CDF is inverted in float64, which resolves windows down to ~37 standard deviations
    from the mean (float32 underflows near 13).
    """
    alpha, beta = _standardized(mu.double(), std.double(), lo.double(), hi.double())
    flip = alpha + beta > 0
    a, b = torch.where(flip, -beta, alpha), torch.where(flip, -alpha, beta)
    u = torch.rand(mu.shape, generator=gen, device=mu.device, dtype=mu.dtype).double()
    pa, pb = log_ndtr(a).exp(), log_ndtr(b).exp()  # ndtr itself underflows below -8
    x = ndtri((pa + u * (pb - pa)).clamp(torch.finfo(torch.float64).tiny, 1.0)).clamp(a, b)
    return (mu.double() + std.double() * torch.where(flip, -x, x)).to(mu.dtype)


def mean(mu: Tensor, std: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """E[a] = mu + s (pdf(alpha) - pdf(beta)) / (Phi(beta) - Phi(alpha)), per dimension."""
    alpha, beta = _standardized(mu, std, lo, hi)
    log_z = log_mass(mu, std, lo, hi)
    pdf_a = torch.exp(-0.5 * alpha.square() - _LOG_SQRT_2PI - log_z)
    pdf_b = torch.exp(-0.5 * beta.square() - _LOG_SQRT_2PI - log_z)
    return mu + std * (pdf_a - pdf_b)


def product(mus: Tensor, stds: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """Untruncated parameters of prod_k N(mu_k, s_k)^{w_k}: (K, ..., da) -> (mu, s) (..., da).

    Truncating the result to the box gives the product of the truncated experts exactly.
    """
    w = weights.reshape(-1, *([1] * (mus.dim() - 1)))
    prec = w / stds.square()
    total = prec.sum(0)
    return (prec * mus).sum(0) / total, total.rsqrt()
