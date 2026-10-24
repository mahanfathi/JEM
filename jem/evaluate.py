"""Evaluation: is z an option, does it generate the right behaviour, does it compose?

library      Infer z from 16-step held-out windows, k-means into K options, match clusters
             to experts (expert labels are used only here) -> NMI, cluster accuracy.
recognition  Classify fresh N-step windows with the energy itself,
             argmax_k sum_t log p(a_t | s_t, z_k)  -> accuracy as evidence accumulates.
fidelity     KL(expert || model) at held-out states from exact expert densities and an
             importance estimate of each option's partition function; heading error of
             the model's mean action.
rollouts     Closed-loop sampling from each option; agreement with the matched expert.
composition  Sum of energies of two options. Experts 90 degrees apart should compose to
             their bisector (product of experts); antipodal experts should cancel.
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor

from jem.config import Config
from jem.data import Demos, collect, sample_windows
from jem.envs import ParticleWorld
from jem.envs.particle import wrap
from jem.inference import Proposals, infer_option

CHUNK = 2**21  # candidate (state, action) pairs per forward pass


# ----------------------------------------------------------------------------- primitives


def option_logits(model, obs: Tensor, zs: Tensor, acts: Tensor) -> Tensor:
    """-E for P options at S states over K actions.

    obs (S, ds), zs (P, d), acts (S|1, K, da) -> (S, P, K).
    """
    if model.kind == "linear":
        phi = (
            model.shared_features(obs, acts[0]) if acts.shape[0] == 1 else model.features(obs, acts)
        )
        return model.scale * torch.einsum("pd,skd->spk", zs, phi)
    acts = acts.expand(obs.shape[0], *acts.shape[-2:])
    out = [model.logits(obs, z.expand(obs.shape[0], -1), acts) for z in zs]
    return torch.stack(out, dim=1)


def option_logits_chunked(model, obs: Tensor, zs: Tensor, acts: Tensor) -> Tensor:
    """`option_logits` over state chunks so a forward pass sees at most ~CHUNK pairs."""
    per = max(1, CHUNK // (acts.shape[-2] * (1 if model.kind == "linear" else zs.shape[0])))
    shared = acts.shape[0] == 1
    return torch.cat(
        [
            option_logits(model, obs[i : i + per], zs, acts if shared else acts[i : i + per])
            for i in range(0, obs.shape[0], per)
        ]
    )


def log_prob(logit_at: Tensor, logit_props: Tensor, log_volume: float) -> Tensor:
    """Importance estimate log p(a|s,z) = u(a) - log(V/M sum_j exp u(a_j)), a_j ~ U(box)."""
    m = logit_props.shape[-1]
    return logit_at - (logit_props.logsumexp(-1) + log_volume - math.log(m))


def mean_action(logits: Tensor, acts: Tensor) -> Tensor:
    """Self-normalised mean action: logits (..., K), acts (K, da) -> (..., da)."""
    return logits.softmax(-1) @ acts


def angle_deg(u: Tensor, v: Tensor) -> Tensor:
    cos = torch.nn.functional.cosine_similarity(u, v, dim=-1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def infer_chunked(model, obs: Tensor, act: Tensor, cfg: Config, proposals: Tensor, gen) -> Tensor:
    per = max(1, CHUNK // (obs.shape[1] * (proposals.shape[0] + 1)))
    parts = [
        infer_option(
            model, obs[i : i + per], act[i : i + per], cfg, proposals, noise=False, gen=gen
        )
        for i in range(0, obs.shape[0], per)
    ]
    return torch.cat(parts)


def step_loglik(model, obs, act, zs, proposals: Proposals, gen) -> Tensor:
    """log p(a_t | s_t, z_k) for every step and option: (W, N, ·), (P, d) -> (W, N, P)."""
    w, n, _ = obs.shape
    props = proposals.sample(gen)
    cand = torch.cat([act.unsqueeze(-2), props.expand(w, n, *props.shape)], dim=-2)
    flat_obs, flat_cand = obs.reshape(w * n, -1), cand.reshape(w * n, *cand.shape[-2:])
    per = max(1, CHUNK // (cand.shape[-2] * max(1, zs.shape[0] if model.kind != "linear" else 1)))
    out = []
    for i in range(0, w * n, per):
        u = option_logits(model, flat_obs[i : i + per], zs, flat_cand[i : i + per])
        out.append(log_prob(u[..., 0], u[..., 1:], proposals.log_volume))
    return torch.cat(out).reshape(w, n, -1)


def hmm_filter(loglik: Tensor, hazard: float) -> Tensor:
    """Online option posterior under a switching prior; loglik (B, T, K) -> beliefs (B, T, K).

    The active option persists with probability 1 - hazard per step and otherwise
    jumps to one of the other K - 1 options uniformly.
    """
    k = loglik.shape[-1]
    stay = torch.full((k, k), hazard / (k - 1), device=loglik.device).fill_diagonal_(1 - hazard)
    log_trans = stay.log()
    log_b = loglik[:, 0] - loglik[:, 0].logsumexp(-1, keepdim=True)
    beliefs = [log_b]
    for t in range(1, loglik.shape[1]):
        log_b = (log_b[:, :, None] + log_trans).logsumexp(1) + loglik[:, t]
        log_b = log_b - log_b.logsumexp(-1, keepdim=True)
        beliefs.append(log_b)
    return torch.stack(beliefs, 1).exp()


# ----------------------------------------------------------------------------- clustering


def kmeans(x: Tensor, k: int, gen, iters: int = 100, restarts: int = 8) -> tuple[Tensor, Tensor]:
    best, best_inertia = None, math.inf
    for _ in range(restarts):
        centers = x[torch.randint(len(x), (1,), generator=gen, device=x.device)]
        for _ in range(1, k):  # k-means++ seeding
            d2 = torch.cdist(x, centers).pow(2).min(1).values
            probs = d2 / d2.sum() if d2.sum() > 0 else torch.full_like(d2, 1 / len(d2))
            centers = torch.cat([centers, x[torch.multinomial(probs, 1, generator=gen)]])
        for _ in range(iters):
            assign = torch.cdist(x, centers).argmin(1)
            onehot = torch.nn.functional.one_hot(assign, k).to(x.dtype)
            counts = onehot.sum(0)
            new = torch.where(
                counts[:, None] > 0, onehot.T @ x / counts.clamp_min(1)[:, None], centers
            )
            if torch.allclose(new, centers):
                break
            centers = new
        inertia = torch.cdist(x, centers).pow(2).min(1).values.sum().item()
        if inertia < best_inertia:
            best, best_inertia = centers, inertia
    return best, torch.cdist(x, best).argmin(1)


def contingency(a: Tensor, b: Tensor, ka: int, kb: int) -> Tensor:
    c = torch.zeros(ka, kb, device=a.device)
    c.index_put_((a, b), torch.ones_like(a, dtype=c.dtype), accumulate=True)
    return c


def nmi(a: Tensor, b: Tensor, ka: int, kb: int) -> float:
    p = contingency(a, b, ka, kb)
    p = p / p.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * (p[nz].log() - (pa[:, None] * pb[None, :])[nz].log())).sum()
    ha = -(pa[pa > 0] * pa[pa > 0].log()).sum()
    hb = -(pb[pb > 0] * pb[pb > 0].log()).sum()
    return (2 * mi / (ha + hb)).item() if ha + hb > 0 else 0.0


def match(assign: Tensor, labels: Tensor, k: int) -> list[int]:
    """Cluster -> expert bijection maximising agreement (exhaustive; k <= 8)."""
    c = contingency(assign, labels, k, k).cpu()
    perms = itertools.permutations(range(k))
    return list(max(perms, key=lambda p: sum(c[i, p[i]].item() for i in range(k))))


# ----------------------------------------------------------------------------- rollouts


@torch.no_grad()
def rollout(
    model, world: ParticleWorld, zs: Tensor, start: Tensor, steps: int, proposals, gen, weights=None
):
    """Sample a ~ p(a|s) ∝ exp(sum_p w_p logits(s, z_p, a)) by importance resampling.

    zs (P, d) are composed by a weighted sum of energies (weights default to 1).
    Returns obs (R, T+1, ds), act (R, T, da).
    """
    w = torch.ones(len(zs), device=zs.device) if weights is None else torch.as_tensor(weights)
    w = w.to(zs.device, zs.dtype)[None, :, None]
    obs, traj, acts = start, [start], []
    for _ in range(steps):
        props = proposals.sample(gen)
        u = (w * option_logits(model, obs, zs, props[None])).sum(1)  # (R, M)
        a = props[torch.multinomial(u.softmax(-1), 1, generator=gen).squeeze(-1)]
        obs = world.step(obs, a)
        traj.append(obs)
        acts.append(a)
    return torch.stack(traj, 1), torch.stack(acts, 1)


# ----------------------------------------------------------------------------- main entry


def expert_pairs(world: ParticleWorld) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(adjacent pairs 90 degrees apart, antipodal pairs)."""
    adjacent, antipodal = [], []
    for i, j in itertools.combinations(range(world.num_experts), 2):
        gap = abs(wrap(world.expert_angles[i] - world.expert_angles[j]).item())
        if abs(gap - math.pi / 2) < 1e-3:
            adjacent.append((i, j))
        elif abs(gap - math.pi) < 1e-3:
            antipodal.append((i, j))
    return adjacent, antipodal


@torch.no_grad()
def evaluate(model, world: ParticleWorld, demos: Demos, cfg: Config, seed: int = 0) -> dict:
    """Run every evaluation. Returns {"metrics": {...}, "artifacts": {...}} for the figures."""
    ec = cfg.eval
    device = world.device
    gen = torch.Generator(device=device).manual_seed(seed)
    k = world.num_experts
    was_training = model.training
    model.eval()
    ctx_props = Proposals(world.act_dim, world.act_low, world.act_high, 1024, device, seed=11)
    big_props = Proposals(
        world.act_dim, world.act_low, world.act_high, ec.num_proposals, device, 13
    )
    metrics: dict[str, float] = {}

    # Library -------------------------------------------------------------------------
    n_lib = max(ec.context_lens)
    win = sample_windows(demos, ec.windows, n_lib, gen)
    labels = win.expert.mode(dim=1).values
    z = infer_chunked(model, win.obs, win.act, cfg, ctx_props.sample(gen), gen)
    centroids, assign = kmeans(z, k, gen)
    perm = match(assign, labels, k)  # cluster c <-> expert perm[c]
    inv = [perm.index(e) for e in range(k)]
    library = centroids[inv]  # library[e] = option matched to expert e
    metrics["eval/nmi"] = nmi(assign, labels, k, k)
    metrics["eval/cluster_acc"] = (
        (torch.tensor(perm, device=device)[assign] == labels).float().mean().item()
    )
    metrics["eval/z_norm"] = z.norm(dim=-1).mean().item()

    # Recognition by energy -------------------------------------------------------------
    recog, bayes = {}, {}
    for n in ec.context_lens:
        w = sample_windows(demos, ec.windows, n, gen)
        truth = w.expert.mode(dim=1).values
        ll = step_loglik(model, w.obs, w.act, library, ctx_props, gen).sum(1)
        recog[n] = (ll.argmax(-1) == truth).float().mean().item()
        # Ceiling: the same rule with the true expert densities.
        experts = torch.arange(k, device=device).expand(*w.expert.shape, k)
        true_ll = world.expert_log_prob(w.obs[..., None, :], experts, w.act[..., None, :]).sum(1)
        bayes[n] = (true_ll.argmax(-1) == truth).float().mean().item()
        metrics[f"eval/recog@{n}"] = recog[n]
        metrics[f"eval/recog_bayes@{n}"] = bayes[n]

    # Fidelity --------------------------------------------------------------------------
    flat = demos.obs.reshape(-1, demos.obs.shape[-1])
    states = flat[torch.randint(len(flat), (1024,), generator=gen, device=device)]
    props = big_props.sample(gen)
    kls, heading = [], []
    for e in range(k):
        ex = torch.full((len(states),), e, device=device)
        a = world.expert_act(states, ex, gen)
        cand = torch.cat([a[:, None], props.expand(len(a), *props.shape)], 1)
        u = option_logits_chunked(model, states, library[e : e + 1], cand)[:, 0]
        kls.append(
            (
                world.expert_log_prob(states, ex, a)
                - log_prob(u[:, 0], u[:, 1:], big_props.log_volume)
            ).mean()
        )
        heading.append(
            angle_deg(mean_action(u[:, 1:], props), world.expert_mean_dir(states, ex)).mean()
        )
    metrics["eval/policy_kl"] = torch.stack(kls).mean().item()
    metrics["eval/heading_err_deg"] = torch.stack(heading).mean().item()

    # Composition -----------------------------------------------------------------------
    adjacent, antipodal = expert_pairs(world)
    u_all = option_logits_chunked(model, states, library, props[None])  # (S, K, M)
    comp_err, cancel, antipodal_cos = [], [], []
    for i, j in adjacent:
        m = mean_action(u_all[:, i] + u_all[:, j], props)
        comp_err.append(angle_deg(m, world.ideal_composition_dir(states, i, j)).mean())
    unit = props / props.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    for i, j in antipodal:
        cancel.append(mean_action(u_all[:, i] + u_all[:, j], unit).norm(dim=-1).mean())
        if library.shape[1] > 1:  # cosines of 1-D options are trivially +-1
            antipodal_cos.append(
                torch.nn.functional.cosine_similarity(library[i], library[j], dim=0)
            )
    single = torch.stack([mean_action(u_all[:, e], unit).norm(dim=-1).mean() for e in range(k)])
    if comp_err:
        metrics["eval/compose_err_deg"] = torch.stack(comp_err).mean().item()
    if cancel:
        metrics["eval/cancel_resultant"] = torch.stack(cancel).mean().item()
        metrics["eval/single_resultant"] = single.mean().item()
    if antipodal_cos:
        metrics["eval/antipodal_cos"] = torch.stack(antipodal_cos).mean().item()

    # Behaviour algebra: z_i + w z_j should steer atan(w) off expert i's heading ------------
    if adjacent:
        i, j = adjacent[0]
        dir_i = world.expert_mean_dir(states, torch.full((len(states),), i, device=device))
        algebra = {}
        for wgt in ec.algebra_weights:
            m = mean_action(u_all[:, i] + wgt * u_all[:, j], props)
            algebra[wgt] = angle_deg(m, dir_i).mean().item()
        target = {wgt: math.degrees(math.atan(wgt)) for wgt in ec.algebra_weights}
        metrics["eval/algebra_err_deg"] = sum(
            abs(algebra[wgt] - target[wgt]) for wgt in algebra
        ) / len(algebra)

    # Tracking: options that switch mid-trajectory, filtered online over the library -----
    track = {}
    if ec.track_traj > 0:
        switching = collect(world, ec.track_traj, demos.obs.shape[1], gen, ec.track_hazard)
        ll = step_loglik(model, switching.obs, switching.act, library, ctx_props, gen)
        experts = torch.arange(k, device=device).expand(*switching.expert.shape, k)
        true_ll = world.expert_log_prob(
            switching.obs[..., None, :], experts, switching.act[..., None, :]
        )
        belief = hmm_filter(ll, ec.track_hazard)
        bayes_belief = hmm_filter(true_ll, ec.track_hazard)
        metrics["eval/track_acc"] = (belief.argmax(-1) == switching.expert).float().mean().item()
        metrics["eval/track_acc_bayes"] = (
            (bayes_belief.argmax(-1) == switching.expert).float().mean().item()
        )
        track = {"expert": switching.expert[0], "belief": belief[0], "bayes": bayes_belief[0]}

    # Rollouts --------------------------------------------------------------------------
    sir = Proposals(world.act_dim, world.act_low, world.act_high, 1024, device, seed=17)
    start = world.reset(ec.rollouts, gen)
    agree = []
    rollouts = {}
    for e in range(k):
        traj, acts = rollout(model, world, library[e : e + 1], start, ec.rollout_len, sir, gen)
        ex = torch.full(acts.shape[:2], e, device=device)
        cos = torch.nn.functional.cosine_similarity(
            acts, world.expert_mean_dir(traj[:, :-1], ex), dim=-1
        )
        agree.append(cos.mean())
        rollouts[world.expert_names[e]] = traj
    metrics["eval/rollout_agree"] = torch.stack(agree).mean().item()
    metrics["eval/rollout_agree_expert"] = math.exp(-0.5 * world.cfg.angle_std**2)
    for i, j in adjacent + antipodal:
        traj, _ = rollout(model, world, library[[i, j]], start, ec.rollout_len, sir, gen)
        rollouts[f"{world.expert_names[i]}+{world.expert_names[j]}"] = traj

    model.train(was_training)
    return {
        "metrics": metrics,
        "artifacts": {
            "z": z,
            "labels": labels,
            "library": library,
            "recog": recog,
            "recog_bayes": bayes,
            "track": track,
            "rollouts": rollouts,
            "start": start,
            "adjacent": adjacent,
            "antipodal": antipodal,
        },
    }
