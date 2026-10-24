"""Option inference and action proposals.

Newton (option-linear energy only). Given scaled candidate features U (B, N, K, d) for
N context steps -- index 0 the demonstrated action, 1..M uniform proposals -- the MAP
option minimises the strongly convex

    F(z) = sum_t [ logsumexp_k <z, U_tk> - <z, U_t0> ] + |z|^2 / (2 sigma^2),

    grad F = sum_t (E_pi_t[U_t] - U_t0) + z / sigma^2          (moment matching)
    hess F = sum_t Cov_pi_t[U_t] + I / sigma^2                 (>= I / sigma^2)

with pi_t = softmax_k <z, U_tk>. Damped Newton with a parallel line search converges
in a handful of iterations, from z = 0 (the "no preference" option), deterministically.
Only dot products with cached features are needed: the network runs once per batch.

Gradients through z* use the implicit function theorem. Because grad F(z*) = 0, one
extra Newton step taken *with* autograd, z* - H^{-1} grad F(z*, theta) (H detached),
leaves the value unchanged and has Jacobian exactly -H^{-1} d(grad F)/d theta.

Langevin / GD (the 2022 algorithm): K steps of z <- z - step/2 dE/dz (+ sqrt(step) noise)
on sum_t E(s_t, z, a_t), optionally backpropagated through (create_graph=True).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from jem.config import Config, InferConfig

LINE_SEARCH = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0)


class Proposals:
    """Uniform action proposals on a box via randomly shifted Sobol points (randomised QMC)."""

    def __init__(self, act_dim: int, low: float, high: float, num: int, device, seed: int = 0):
        self.base = SobolEngine(act_dim, scramble=True, seed=seed).draw(num).to(device)
        self.low, self.high = low, high
        self.log_volume = act_dim * math.log(high - low)

    def sample(self, gen: torch.Generator | None = None) -> Tensor:
        shift = torch.rand(self.base.shape[-1], generator=gen, device=self.base.device)
        return self.low + (self.high - self.low) * torch.remainder(self.base + shift, 1.0)


# ----------------------------------------------------------------------------- Newton


def option_objective(z: Tensor, U: Tensor, prec: float) -> Tensor:
    """F at C candidate points per problem: z (B, C, d), U (B, N, K, d) -> (B, C)."""
    logits = torch.einsum("bcd,bnkd->bcnk", z, U)
    nll = (logits.logsumexp(-1) - logits[..., 0]).sum(-1)
    return nll + 0.5 * prec * z.pow(2).sum(-1)


def _grad(z: Tensor, U: Tensor, prec: float) -> tuple[Tensor, Tensor, Tensor]:
    pi = torch.einsum("bd,bnkd->bnk", z, U).softmax(-1)
    mean = torch.einsum("bnk,bnkd->bnd", pi, U)
    return (mean - U[:, :, 0]).sum(1) + prec * z, pi, mean


def _hessian(pi: Tensor, mean: Tensor, U: Tensor, prec: float) -> Tensor:
    b, n, k, d = U.shape
    x = (pi.sqrt().unsqueeze(-1) * U).reshape(b, n * k, d)
    h = x.transpose(1, 2) @ x - mean.transpose(1, 2) @ mean
    return h + prec * torch.eye(d, device=U.device, dtype=U.dtype)


@torch.no_grad()
def newton_solve(U: Tensor, prec: float, iters: int) -> tuple[Tensor, Tensor]:
    """Minimise F; returns z* (B, d) and the final gradient norm (B,)."""
    b, _, _, d = U.shape
    z = U.new_zeros(b, d)
    steps = torch.tensor(LINE_SEARCH, device=U.device, dtype=U.dtype)
    rows = torch.arange(b, device=U.device)
    for _ in range(iters):
        g, pi, mean = _grad(z, U, prec)
        delta = torch.linalg.solve(_hessian(pi, mean, U, prec), g)
        cand = z[:, None] - steps[None, :, None] * delta[:, None]
        z = cand[rows, option_objective(cand, U, prec).argmin(-1)]
    return z, _grad(z, U, prec)[0].norm(dim=-1)


def infer_newton(
    U: Tensor, prec: float, iters: int, implicit_grad: bool = True
) -> tuple[Tensor, Tensor]:
    """MAP option with (optionally) the exact implicit gradient. Returns (z, residual)."""
    z_star, residual = newton_solve(U.detach(), prec, iters)
    if not (implicit_grad and U.requires_grad):
        return z_star, residual
    g, pi, mean = _grad(z_star, U, prec)
    with torch.no_grad():
        h = _hessian(pi, mean, U, prec)
    return z_star - torch.linalg.solve(h, g), residual


# ----------------------------------------------------------------------------- Langevin


def langevin(
    energy_fn,
    x: Tensor,
    steps: int,
    step_size: float,
    grad_clip: float,
    noise: bool,
    create_graph: bool,
    gen: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """x <- x - step/2 * clip(dE/dx) + sqrt(step) * eps. Returns (x_K, clipped fraction)."""
    clipped = x.new_zeros(())
    with torch.enable_grad():
        for _ in range(steps):
            if not x.requires_grad:
                x = x.detach().requires_grad_()
            (g,) = torch.autograd.grad(energy_fn(x).sum(), x, create_graph=create_graph)
            if grad_clip > 0:
                clipped = clipped + (g.abs() > grad_clip).float().mean().detach() / steps
                g = g.clamp(-grad_clip, grad_clip)
            x = x - 0.5 * step_size * g
            if noise:
                eps = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
                x = x + math.sqrt(step_size) * eps
            if not create_graph:
                x = x.detach()
    return x, clipped


def infer_langevin(
    model,
    obs: Tensor,
    act: Tensor,
    cfg: InferConfig,
    noise: bool,
    create_graph: bool = False,
    gen: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """2022 option inference on sum_t E(s_t, z, a_t). Returns (z_K, z_0, clip fraction)."""
    b, n, _ = obs.shape
    z0 = cfg.init_std * torch.randn(b, model.option_dim, generator=gen, device=obs.device)

    def energy(z: Tensor) -> Tensor:
        return model.energy(obs, z[:, None, :].expand(b, n, z.shape[-1]), act)

    z, clipped = langevin(
        energy, z0, cfg.iters, cfg.step_size, cfg.grad_clip, noise, create_graph, gen
    )
    return z, z0, clipped


# ----------------------------------------------------------------------------- dispatch


def infer_option(
    model,
    obs: Tensor,
    act: Tensor,
    cfg: Config,
    proposals: Tensor | None = None,
    *,
    create_graph: bool = False,
    noise: bool | None = None,
    gen: torch.Generator | None = None,
) -> Tensor:
    """Infer one option per context window: obs (B, N, ds), act (B, N, da) -> z (B, d)."""
    ic = cfg.infer
    if ic.method == "newton":
        if model.kind != "linear" or proposals is None:
            raise ValueError("newton inference needs the option-linear energy and proposals")
        U = model.scale * model.candidate_features(obs, act, proposals).float()
        z, _ = infer_newton(U, ic.prior_std**-2, ic.iters, ic.implicit_grad and create_graph)
        return z
    if ic.method not in ("langevin", "gd"):
        raise KeyError(f"unknown inference method {ic.method!r}")
    if noise is None:
        noise = ic.method == "langevin"
    z, _, _ = infer_langevin(model, obs, act, ic, noise, create_graph, gen)
    return z
