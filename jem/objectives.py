"""Training objectives. Each returns (loss, stats) where stats are detached scalars.

jem     Infer the option from the context steps of a window (exact Newton MAP with an
        implicit gradient), then score the held-out target steps with InfoNCE against
        fixed uniform proposals. Predicting *held-out* steps is what makes z an option:
        only what persists across time helps. With `symmetric`, the roles of the two
        halves are swapped as well (cross-fitting; features are shared, so it is cheap).
        Negatives never come from the model, so there is no sampler/energy feedback loop.
        With `objective.hazard` h > 0 (options may switch), context steps are weighted by
        (1-h)^distance-to-boundary and each target is scored by a two-component mixture:
        same option with probability (1-h)^lag, else an outlier scored by the z = 0 model.
        With prototype options (`model.prototypes` P > 0, active from
        `objective.prototype_start`), each window also scores its targets by the mixture
        sum_c pi(c | context) p(targets | z_c) over the learned prototypes z_c, pi the
        recognition posterior from the context: the exact held-out likelihood of a
        mixture of P options. A prototype must explain every demonstration it is
        assigned, so it cannot absorb per-demonstration nuisance the way a per-window z can.
        With `objective.proposal = "prior"`, most candidates are drawn from a learned
        behaviour prior q (jem/prior.py) and every logit carries the importance offset
        -log r(a), r the proposal mixture: the ranking objective still learns
        <z, phi> = log p(a|s,z) + const relative to the uniform base measure.

legacy  Faithful port of the 2022 `loss_ML_KL_KLz`: Langevin inference of z from the
        context (backpropagated through), Langevin action negatives, softplus ML loss
        with a hinge mask, KL terms through the unrolled chains with frozen parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call

from jem.config import Config
from jem.data import Window
from jem.inference import infer_langevin, infer_newton, langevin


def _spread(z: Tensor) -> Tensor:
    """Dispersion of options across a batch relative to their size; ~0 means option collapse."""
    return (z - z.mean(0)).norm(dim=-1).mean() / (z.norm(dim=-1).mean() + 1e-8)


def candidates(
    model, win: Window, proposals: Tensor, cfg: Config, prior=None, gen=None
) -> tuple[Tensor, Tensor | None]:
    """Candidate features (B, L, K, d), index 0 observed, and logit offsets (B, L, K) or None.

    Uniform proposals need no offsets (they are constant). With a behaviour prior, each
    half of the window (context, targets) gets one candidate set shared by its steps:
    Q draws from q(.|s) at the half's first state, which none of the half's actions has
    influenced, plus the first M_u shared uniform points, so the factored action tower
    runs once per half. Every logit carries -log r(a), r = (1 - rho) q(.|s_first) + rho U
    the density the candidates were drawn from (rho = M_u / (Q + M_u)).
    """
    if prior is None:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
            return model.candidate_features(win.obs, win.act, proposals), None
    oc = cfg.objective
    m_u = max(1, round(oc.uniform_frac * oc.num_proposals))
    shared, rho = proposals[:m_u], m_u / oc.num_proposals
    phis, offs = [], []
    nc = cfg.data.context_len
    for part in (slice(0, nc), slice(nc, None)):
        obs, act = win.obs[:, part], win.act[:, part]
        first = obs[:, 0]
        with torch.no_grad():
            own = prior.sample(first, oc.num_proposals - m_u, gen)  # (B, Q, da)
            cand = torch.cat([own, shared.expand(len(own), -1, -1)], 1)
            log_r = prior.log_mixture(first, torch.cat([act, cand], 1), rho)  # (B, n + Q + M_u)
            n = act.shape[1]
            offs.append(-torch.cat([log_r[:, :n, None], log_r[:, None, n:].expand(-1, n, -1)], -1))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
            phis.append(model.candidate_features(obs, act, shared, own))
    return torch.cat(phis, 1), torch.cat(offs, 1)


def prototype_loglik(proto: Tensor, U: Tensor, offs: Tensor) -> Tensor:
    """Per-step ranking log-likelihood of every prototype: (P, d), (B, N, K, d) -> (B, N, P)."""
    logits = torch.einsum("pd,bnkd->bnpk", proto, U) + offs[:, :, None]
    return logits.log_softmax(-1)[..., 0]


def jem_loss(
    model, win: Window, proposals: Tensor, cfg: Config, prior=None, gen=None, prototypes=False
) -> tuple[Tensor, dict]:
    nc = cfg.data.context_len
    phi, offs = candidates(model, win, proposals, cfg, prior, gen)
    U = model.scale * phi.float()  # (B, L, K, d)
    prec = cfg.infer.prior_std**-2

    b, nt = U.shape[0], U.shape[1] - nc
    # Steps from the context/target boundary, for the switching-aware variant (hazard > 0);
    # targets that start `gap` steps after the context lie that much further away.
    before = torch.arange(nc - 1, -1, -1, device=U.device)  # context steps, forward pair
    after = torch.arange(1, nt + 1, device=U.device)  # target lags, forward pair
    if win.gap is not None:
        before, after = before.expand(b, -1), after + win.gap[:, None]
    if offs is None:  # uniform proposals: constant offsets
        offs = torch.zeros(U.shape[:3], device=U.device)
    pairs = [(U[:, :nc], U[:, nc:], offs[:, :nc], offs[:, nc:], before, after)]
    if cfg.objective.symmetric:
        # Reverse pair: infer from the second half; first-half step t lies nc - t steps back.
        back = torch.arange(nt, device=U.device)
        lag = before + 1 if win.gap is None else before + 1 + win.gap[:, None]
        pairs.append((U[:, nc:], U[:, :nc], offs[:, nc:], offs[:, :nc], back, lag))
    if len(pairs) == 2 and pairs[0][0].shape == pairs[1][0].shape:
        pairs = [
            tuple(
                torch.cat([p[i] if i < 4 else p[i].expand(b, -1) for p in pairs]) for i in range(6)
            )
        ]

    nlls, losses, zs, residuals, ess, gains, mixes, ents = [], [], [], [], [], [], [], []
    hazard = cfg.objective.hazard
    proto = model.prototypes if prototypes else None
    for ctx, tgt, ctx_offs, tgt_offs, dist, lag in pairs:
        w = (1 - hazard) ** dist.float().expand(ctx.shape[0], -1) if hazard > 0 else None
        z, residual = infer_newton(ctx, prec, cfg.infer.iters, cfg.infer.implicit_grad, w, ctx_offs)
        logp = (torch.einsum("bd,bnkd->bnk", z, tgt) + tgt_offs).log_softmax(-1)
        nll = -logp[..., 0].mean()
        nlls.append(nll)
        # Nats above the z = 0 model, which is log K - nll under uniform proposals.
        gains.append(tgt_offs.log_softmax(-1)[..., 0].mean().detach().neg() - nll.detach())
        if hazard > 0:
            # A target shares the context's option with probability (1-h)^lag; otherwise it is
            # an outlier scored by the z = 0 model (1/K in InfoNCE units under uniform props).
            log_stay = lag.float().expand(tgt.shape[0], -1) * math.log1p(-hazard)
            base = tgt_offs.log_softmax(-1)[..., 0]
            log_switch = torch.log(-torch.expm1(log_stay)) + base
            losses.append(-torch.logaddexp(log_stay + logp[..., 0], log_switch).mean())
        else:
            losses.append(nll)
        zs.append(z.detach())
        residuals.append(residual)
        ess.append((1.0 / logp.detach().exp().pow(2).sum(-1)).mean())
        if proto is not None:  # held-out likelihood of the mixture over prototypes
            post = prototype_loglik(proto, ctx, ctx_offs).sum(1).log_softmax(-1)  # (B, P)
            mix = (post + prototype_loglik(proto, tgt, tgt_offs).sum(1)).logsumexp(-1)
            mixes.append(-mix.mean() / tgt.shape[1])
            ents.append(-(post.exp() * post).sum(-1).mean().detach())
    loss = torch.stack(losses).mean()
    if mixes:
        oc = cfg.objective
        loss = oc.continuous_weight * loss + oc.prototype_weight * torch.stack(mixes).mean()

    with torch.no_grad():
        z = torch.cat(zs)
        k = U.shape[2]
        prop = phi[:, :, 1:].float()
        stats = {
            "train/nll": torch.stack(nlls).mean().detach(),
            "train/gain": torch.stack(gains).mean(),  # nats above the option-free model
            "diag/z_norm": z.norm(dim=-1).mean(),
            "diag/z_spread": _spread(z),
            "diag/scale": model.scale.detach(),
            "diag/ess_frac": torch.stack(ess).mean() / k,
            "diag/newton_residual": torch.cat(residuals).mean(),
            "diag/feat_collapse": prop.mean(-2).norm(dim=-1).mean(),
        }
        if mixes:
            stats["train/proto_nll"] = torch.stack(mixes).mean().detach()
            stats["diag/proto_entropy"] = torch.stack(ents).mean()  # recognition certainty
    return loss, stats


def legacy_loss(
    model, win: Window, cfg: Config, gen: torch.Generator | None = None
) -> tuple[Tensor, dict]:
    oc, ic = cfg.objective, cfg.infer
    nc = cfg.data.context_len
    obs_c, act_c = win.obs[:, :nc], win.act[:, :nc]
    obs_t, act_t = win.obs[:, nc:], win.act[:, nc:]
    b, nt, da = act_t.shape
    noise = ic.method == "langevin"

    # Option from the context, differentiated through the chain (as in 2022).
    z, z0, clip_z = infer_langevin(model, obs_c, act_c, ic, noise, create_graph=True, gen=gen)
    z_t = z[:, None, :].expand(b, nt, z.shape[-1])

    # Action negatives: Langevin chains from U[-1, 1], also kept differentiable for L_KL.
    a0 = torch.rand(b, nt, oc.action_chains, da, generator=gen, device=act_t.device) * 2 - 1
    a_neg, clip_a = langevin(
        lambda a: model.energy(obs_t, z_t, a),
        a0,
        oc.action_steps,
        oc.action_step_size,
        oc.action_grad_clip,
        noise,
        create_graph=True,
        gen=gen,
    )

    e_pos = model.energy(obs_t, z_t, act_t)  # (B, Nt)
    e_neg = model.energy(obs_t, z_t, a_neg.detach())  # (B, Nt, C)
    mask = ((a_neg.detach() - act_t[:, :, None]).norm(dim=-1) > oc.hinge_eps).float()
    loss_ml = F.softplus(e_pos - (e_neg * mask).mean(-1)).mean()  # plain mean, as in 2022

    frozen = {name: p.detach() for name, p in model.named_parameters()}
    loss_kl = functional_call(model, frozen, (obs_t, z_t.detach(), a_neg)).mean()

    reps = oc.klz_chains
    zb, _, _ = infer_langevin(
        model,
        obs_c.repeat(reps, 1, 1),
        act_c.repeat(reps, 1, 1),
        ic,
        noise,
        create_graph=True,
        gen=gen,
    )
    zb_c = zb[:, None, :].expand(-1, nc, zb.shape[-1])
    loss_klz = functional_call(
        model, frozen, (obs_c.repeat(reps, 1, 1), zb_c, act_c.repeat(reps, 1, 1))
    ).mean()

    loss = loss_ml + oc.kl_coef * loss_kl + oc.klz_coef * loss_klz
    with torch.no_grad():
        stats = {
            "train/loss_ml": loss_ml.detach(),
            "train/loss_kl": loss_kl.detach(),
            "train/loss_klz": loss_klz.detach(),
            "diag/e_pos": e_pos.detach().mean(),
            "diag/e_neg": e_neg.detach().mean(),
            "diag/z_norm": z.detach().norm(dim=-1).mean(),
            "diag/z_spread": _spread(z.detach()),
            "diag/z_moved": (z.detach() - z0).norm(dim=-1).mean(),
            "diag/z_init_norm": z0.norm(dim=-1).mean(),
            "diag/a_moved": (a_neg.detach() - a0).norm(dim=-1).mean(),
            "diag/clip_frac_z": clip_z,
            "diag/clip_frac_a": clip_a,
            "diag/hinge_masked": 1.0 - mask.mean(),
        }
    return loss, stats
