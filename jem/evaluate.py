"""Evaluation: is z an option, does it generate the right behaviour, does it compose?

library      Infer z from 16-step held-out windows, k-means into K options, match clusters
             to experts (expert labels are used only here) -> NMI, cluster accuracy.
recognition  Classify fresh N-step windows with the energy itself,
             argmax_k sum_t log p(a_t | s_t, z_k)  -> accuracy as evidence accumulates.
fidelity     KL(expert || model) at held-out states from exact expert densities and an
             importance estimate of each option's partition function; the angle between
             the task-space effects of the model's and the expert's mean actions.
rollouts     Closed-loop sampling from each option; agreement with the matched expert.
composition  Sum of energies of two options against the product of the two true experts:
             orthogonal experts should compose to their product, antipodal ones cancel.
             Worlds with closed-form products (truncated-Gaussian experts) are also scored
             by the exact KL(product || composed option) in nats.

A world provides `effect(obs, act)` (what an action does in task space) and
`expert_effect(obs, k)` (the expert's mean effect), so the same metrics serve the
particle (effect = velocity) and the arm (effect = end-effector velocity).
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor

from jem.config import Config
from jem.data import Demos, Window, collect, sample_windows
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
    """Self-normalised mean: logits (..., K), acts (K, da) or (..., K, da) -> (..., da)."""
    if acts.dim() == 2:
        return logits.softmax(-1) @ acts
    return torch.einsum("...k,...kd->...d", logits.softmax(-1), acts)


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


def fair_snippets(world, demos: Demos, n: int, length: int, gen) -> Window:
    """Recognition test data: start states pooled over all demos, labels drawn uniformly,
    then that expert rolled out. The start state carries no information about the label,
    so the rule argmax_k sum_t log p_k(a_t | s_t) with true densities is Bayes-optimal.
    (Windows cut from demos would leak the label through where each expert tends to be.)
    """
    flat = demos.state.reshape(-1, demos.state.shape[-1])
    state = flat[torch.randint(len(flat), (n,), generator=gen, device=flat.device)]
    expert = torch.randint(world.num_experts, (n,), generator=gen, device=flat.device)
    obs_t, act_t = [], []
    for _ in range(length):
        obs = world.obs(state)
        act = world.expert_act(obs, expert, gen)
        obs_t.append(obs)
        act_t.append(act)
        state = world.step(state, act)
    return Window(torch.stack(obs_t, 1), torch.stack(act_t, 1), expert[:, None].expand(n, length))


def true_logits(world, states: Tensor, props: Tensor) -> Tensor:
    """log p_k(a | s) of every true expert over shared actions: (S, ds), (M, da) -> (S, K, M)."""
    k = world.num_experts
    experts = torch.arange(k, device=states.device).expand(len(states), k)[..., None]
    return world.expert_log_prob(states[:, None, None, :], experts, props[None, None])


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
def rollout(model, world, zs: Tensor, start: Tensor, steps: int, proposals, gen, weights=None):
    """Sample a ~ p(a|s) ∝ exp(sum_p w_p logits(s, z_p, a)) by importance resampling.

    zs (P, d) are composed by a weighted sum of energies (weights default to 1). `start`
    holds simulator states. Returns obs (R, T+1, ds), act (R, T, da).
    """
    w = torch.ones(len(zs), device=zs.device) if weights is None else torch.as_tensor(weights)
    w = w.to(zs.device, zs.dtype)[None, :, None]
    state = start
    obs = world.obs(state)
    traj, acts = [obs], []
    for _ in range(steps):
        props = proposals.sample(gen)
        u = (w * option_logits(model, obs, zs, props[None])).sum(1)  # (R, M)
        a = props[torch.multinomial(u.softmax(-1), 1, generator=gen).squeeze(-1)]
        state = world.step(state, a)
        obs = world.obs(state)
        traj.append(obs)
        acts.append(a)
    return torch.stack(traj, 1), torch.stack(acts, 1)


@torch.no_grad()
def expert_rollout(world, expert: int, start: Tensor, steps: int, gen) -> tuple[Tensor, Tensor]:
    """Closed-loop trajectories of a true expert: obs (R, T+1, ds), act (R, T, da)."""
    state = start
    obs = world.obs(state)
    ex = torch.full(state.shape[:1], expert, device=state.device)
    traj, acts = [obs], []
    for _ in range(steps):
        a = world.expert_act(obs, ex, gen)
        state = world.step(state, a)
        obs = world.obs(state)
        traj.append(obs)
        acts.append(a)
    return torch.stack(traj, 1), torch.stack(acts, 1)


def agreement(world, obs: Tensor, act: Tensor, expert: int) -> Tensor:
    """Mean cosine between the effect of each executed action and the expert's mean effect."""
    ex = torch.full(act.shape[:-1], expert, device=act.device)
    return torch.nn.functional.cosine_similarity(
        world.effect(obs, act), world.expert_effect(obs, ex), dim=-1
    ).mean()


# ----------------------------------------------------------------------------- main entry


@torch.no_grad()
def evaluate(model, world, demos: Demos, cfg: Config, seed: int = 0) -> dict:
    """Run every evaluation. Returns {"metrics": {...}, "artifacts": {...}} for the figures."""
    ec = cfg.eval
    device = world.device
    gen = torch.Generator(device=device).manual_seed(seed)
    k = world.num_experts
    was_training = model.training
    model.eval()
    ctx_props = Proposals(
        world.act_dim, world.act_low, world.act_high, ec.context_proposals, device, seed=11
    )
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
    proto = getattr(model, "prototypes", None)
    if proto is not None and len(proto) == k and proto.abs().sum() > 0:
        # Learned prototype options: windows are assigned by recognition with the energy.
        metrics["eval/nmi_kmeans"] = nmi(assign, labels, k, k)
        centroids = proto.detach()
        assign = step_loglik(model, win.obs, win.act, centroids, ctx_props, gen).sum(1).argmax(-1)
    perm = match(assign, labels, k)  # cluster c <-> expert perm[c]
    inv = [perm.index(e) for e in range(k)]
    library = centroids[inv]  # library[e] = option matched to expert e
    metrics["eval/nmi"] = nmi(assign, labels, k, k)
    metrics["eval/cluster_acc"] = (
        (torch.tensor(perm, device=device)[assign] == labels).float().mean().item()
    )
    metrics["eval/z_norm"] = z[:, : model.option_dim].norm(dim=-1).mean().item()

    # Recognition by energy -------------------------------------------------------------
    recog, bayes = {}, {}
    for n in ec.context_lens:
        w = fair_snippets(world, demos, ec.windows, n, gen)
        truth = w.expert[:, 0]
        ll = step_loglik(model, w.obs, w.act, library, ctx_props, gen).sum(1)
        recog[n] = (ll.argmax(-1) == truth).float().mean().item()
        # The same rule with the true densities. On fair snippets this is Bayes-optimal.
        experts = torch.arange(k, device=device).expand(*w.expert.shape, k)
        true_ll = world.expert_log_prob(w.obs[..., None, :], experts, w.act[..., None, :]).sum(1)
        bayes[n] = (true_ll.argmax(-1) == truth).float().mean().item()
        metrics[f"eval/recog@{n}"] = recog[n]
        metrics[f"eval/recog_bayes@{n}"] = bayes[n]

    # Fidelity --------------------------------------------------------------------------
    flat = demos.obs.reshape(-1, demos.obs.shape[-1])
    states = flat[torch.randint(len(flat), (1024,), generator=gen, device=device)]
    props = big_props.sample(gen)
    u_all = option_logits_chunked(model, states, library, props[None])  # (S, K, M)
    kls, heading = [], []
    for e in range(k):
        ex = torch.full((len(states),), e, device=device)
        a = world.expert_act(states, ex, gen)
        u_at = option_logits_chunked(model, states, library[e : e + 1], a[:, None])[:, 0, 0]
        log_model = log_prob(u_at, u_all[:, e], big_props.log_volume)
        kls.append((world.expert_log_prob(states, ex, a) - log_model).mean())
        mean = world.effect(states, mean_action(u_all[:, e], props))
        heading.append(angle_deg(mean, world.expert_effect(states, ex)).mean())
    metrics["eval/policy_kl"] = torch.stack(kls).mean().item()
    metrics["eval/heading_err_deg"] = torch.stack(heading).mean().item()

    # Composition -----------------------------------------------------------------------
    # References are the true products of experts: in closed form where the world has it,
    # otherwise pushed through the same estimator as the model.
    adjacent, antipodal = world.expert_pairs()
    exact = hasattr(world, "product_log_prob")
    t_all = None if exact else true_logits(world, states, props)  # (S, K, M)

    def composed(i: int, j: int, wgt: float) -> Tensor:  # mean action of z_i + wgt z_j
        return world.effect(states, mean_action(u_all[:, i] + wgt * u_all[:, j], props))

    def reference(i: int, j: int, wgt: float) -> Tensor:  # mean action of p_i p_j^wgt
        if exact:
            return world.effect(states, world.product_mean(states, (i, j), (1.0, wgt)))
        return world.effect(states, mean_action(t_all[:, i] + wgt * t_all[:, j], props))

    def product_kl(i: int, j: int, wgt: float) -> Tensor:  # KL(p_i p_j^wgt || model)
        a = world.product_act(states, (i, j), (1.0, wgt), gen)
        u_at = option_logits_chunked(model, states, library[[i, j]], a[:, None])[..., 0]
        u_props = u_all[:, i] + wgt * u_all[:, j]
        log_model = log_prob(u_at[:, 0] + wgt * u_at[:, 1], u_props, big_props.log_volume)
        return (world.product_log_prob(states, (i, j), (1.0, wgt), a) - log_model).mean()

    comp_err = [angle_deg(composed(i, j, 1.0), reference(i, j, 1.0)).mean() for i, j in adjacent]
    unit = world.effect(states[:, None], props)  # task-space effect of every proposal
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cancel, antipodal_cos = [], []
    for i, j in antipodal:
        cancel.append(mean_action(u_all[:, i] + u_all[:, j], unit).norm(dim=-1).mean())
        d = model.option_dim  # the inferred part, without a base exponent
        if d > 1:  # cosines of 1-D options are trivially +-1
            antipodal_cos.append(
                torch.nn.functional.cosine_similarity(library[i, :d], library[j, :d], dim=0)
            )
    single = torch.stack([mean_action(u_all[:, e], unit).norm(dim=-1).mean() for e in range(k)])
    if comp_err:
        metrics["eval/compose_err_deg"] = torch.stack(comp_err).mean().item()
    if cancel:
        metrics["eval/cancel_resultant"] = torch.stack(cancel).mean().item()
        metrics["eval/single_resultant"] = single.mean().item()
    if antipodal_cos:
        metrics["eval/antipodal_cos"] = torch.stack(antipodal_cos).mean().item()
    if exact:
        if adjacent:
            kl = [product_kl(i, j, 1.0) for i, j in adjacent]
            metrics["eval/compose_kl"] = torch.stack(kl).mean().item()
        if antipodal:
            kl = [product_kl(i, j, 1.0) for i, j in antipodal]
            metrics["eval/cancel_kl"] = torch.stack(kl).mean().item()

    # Behaviour algebra: z_i + w z_j against the weighted product of the true experts ----
    if adjacent:
        i, j = adjacent[0]
        errs = [angle_deg(composed(i, j, w), reference(i, j, w)).mean() for w in ec.algebra_weights]
        metrics["eval/algebra_err_deg"] = torch.stack(errs).mean().item()
        if exact:
            kl = [product_kl(i, j, w) for w in ec.algebra_weights]
            metrics["eval/algebra_kl"] = torch.stack(kl).mean().item()

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
        agree.append(agreement(world, traj[:, :-1], acts, e))
        rollouts[world.expert_names[e]] = traj
    metrics["eval/rollout_agree"] = torch.stack(agree).mean().item()
    for i, j in adjacent + antipodal:
        traj, _ = rollout(model, world, library[[i, j]], start, ec.rollout_len, sir, gen)
        rollouts[f"{world.expert_names[i]}+{world.expert_names[j]}"] = traj
    ceiling = getattr(world, "agreement_ceiling", None)
    if ceiling is None:  # the experts' own closed-loop agreement, by Monte Carlo
        own = []
        for e in range(k):
            traj, acts = expert_rollout(world, e, start, ec.rollout_len, gen)
            own.append(agreement(world, traj[:, :-1], acts, e))
        ceiling = torch.stack(own).mean().item()
    metrics["eval/rollout_agree_expert"] = ceiling

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
