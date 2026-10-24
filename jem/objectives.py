"""Training objectives. Each returns (loss, stats) where stats are detached scalars.

jem     Infer the option from the context steps of a window (exact Newton MAP with an
        implicit gradient), then score the held-out target steps with InfoNCE against
        fixed uniform proposals. Predicting *held-out* steps is what makes z an option:
        only what persists across time helps. With `symmetric`, the roles of the two
        halves are swapped as well (cross-fitting; features are shared, so it is cheap).
        Negatives never come from the model, so there is no sampler/energy feedback loop.

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


def jem_loss(model, win: Window, proposals: Tensor, cfg: Config) -> tuple[Tensor, dict]:
    nc = cfg.data.context_len
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
        phi = model.candidate_features(win.obs, win.act, proposals)
    U = model.scale * phi.float()  # (B, L, 1+M, d)
    prec = cfg.infer.prior_std**-2

    pairs = [(U[:, :nc], U[:, nc:])]
    if cfg.objective.symmetric:
        pairs.append((U[:, nc:], U[:, :nc]))
    if len(pairs) == 2 and pairs[0][0].shape == pairs[1][0].shape:
        pairs = [(torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs]))]

    nlls, zs, residuals, ess = [], [], [], []
    for ctx, tgt in pairs:
        z, residual = infer_newton(ctx, prec, cfg.infer.iters, cfg.infer.implicit_grad)
        logp = torch.einsum("bd,bnkd->bnk", z, tgt).log_softmax(-1)
        nlls.append(-logp[..., 0].mean())
        zs.append(z.detach())
        residuals.append(residual)
        ess.append((1.0 / logp.detach().exp().pow(2).sum(-1)).mean())
    loss = torch.stack(nlls).mean()

    with torch.no_grad():
        z = torch.cat(zs)
        k = U.shape[2]
        prop = phi[:, :, 1:].float()
        stats = {
            "train/nll": loss.detach(),
            "train/gain": math.log(k) - loss.detach(),  # nats above uniform
            "diag/z_norm": z.norm(dim=-1).mean(),
            "diag/z_spread": _spread(z),
            "diag/scale": model.scale.detach(),
            "diag/ess_frac": torch.stack(ess).mean() / k,
            "diag/newton_residual": torch.cat(residuals).mean(),
            "diag/feat_collapse": prop.mean(-2).norm(dim=-1).mean(),
        }
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
