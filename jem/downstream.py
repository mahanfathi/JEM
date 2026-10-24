"""Composing options over time: reach the target around an obstacle never seen in the demos.

Concurrent composition (summing options) is evaluated in `evaluate.py`. Here options
are also sequenced. A planner picks which option, or which sum of options, to run at
each decision, and the energy model executes it. Each episode puts a circular obstacle
between the agent and the target, so following "toward" alone gets stuck and the
planner has to find a detour.

Every planner is model-predictive control in the true simulator with the same budget of
simulated rollouts per decision: plan over `horizon` steps, execute `replan` steps, repeat.

    greedy      always the "toward" option (no search)
    library     the K discovered options and their pairwise sums (discrete set)
    option-cem  cross-entropy method over z in R^d (the continuous option space)
    action-cem  cross-entropy method over raw action sequences (no options at all)
"""

from __future__ import annotations

import dataclasses as dc
import itertools
import math

import torch
from torch import Tensor

from jem.inference import Proposals


@dc.dataclass
class DownstreamConfig:
    episodes: int = 64
    steps: int = 150  # episode length
    horizon: int = 20  # planning horizon (steps)
    replan: int = 5  # steps executed per decision
    candidates: int = 64  # rollouts per CEM iteration
    iters: int = 3  # CEM iterations (budget = candidates * iters rollouts per decision)
    elite_frac: float = 0.125
    tolerance: float = 0.08  # success radius around the target
    proposals: int = 256  # SIR proposals for the low-level policy
    seed: int = 0
    task: str = "circle"  # circle | cup


def make_tasks(
    n: int, gen: torch.Generator, device, kind: str = "circle"
) -> tuple[Tensor, Tensor, Tensor]:
    """Start and target on opposite sides of an obstacle -> obs, centers (n, O, 2), radii (n, O).

    circle  one disc in the way: going straight slides around it on contact
    cup     a concave wall of overlapping discs whose mouth faces the agent: a trap for
            any controller that only ever heads toward the target
    """
    theta = torch.rand(n, generator=gen, device=device) * 2 * math.pi
    unit = torch.stack([theta.cos(), theta.sin()], -1)
    jitter = 0.1 * (torch.rand(n, 2, generator=gen, device=device) - 0.5)
    # Targets stay inside the training range (|x|, |y| <= 0.5).
    if kind == "circle":
        target, start = 0.5 * unit, -0.75 * unit + jitter
        centers = 0.5 * (start + target) + 0.5 * jitter.flip(-1)
        radii = 0.25 + 0.1 * torch.rand(n, 1, generator=gen, device=device)
        return torch.cat([start, target], -1), centers[:, None], radii
    if kind == "cup":
        # Nine discs (r = 0.1) on an arc; the wall is >= 0.083 thick at its thinnest, more
        # than the largest step dt * sqrt(2) = 0.071, so nothing tunnels through.
        target, start = 0.5 * unit, -0.8 * unit + jitter
        phi = theta[:, None] + torch.deg2rad(torch.linspace(-80, 80, 9, device=device))
        centers = -0.1 * unit[:, None] + 0.32 * torch.stack([phi.cos(), phi.sin()], -1)
        return torch.cat([start, target], -1), centers, torch.full((n, 9), 0.1, device=device)
    raise KeyError(f"unknown task kind {kind!r}")


def step_with_obstacles(world, obs: Tensor, act: Tensor, centers: Tensor, radii: Tensor) -> Tensor:
    """World step, then project any agent that ended inside an obstacle onto its boundary.

    Discs may overlap (the cup), and leaving one disc can enter a neighbour, so the
    projection is repeated until no disc contains the agent.
    """
    nxt = world.step(obs, act)
    p = nxt[..., :2]
    for _ in range(8):
        moved = False
        for o in range(centers.shape[-2]):
            c, r = centers[..., o, :], radii[..., o : o + 1]
            d = p - c
            dist = d.norm(dim=-1, keepdim=True)
            inside = dist < r
            if inside.any():
                p = torch.where(inside, c + d / dist.clamp_min(1e-9) * r, p)
                moved = True
        if not moved:
            break
    return torch.cat([p, nxt[..., 2:]], -1)


def per_state_logits(model, obs: Tensor, z: Tensor, props: Tensor) -> Tensor:
    """-E with one option per state: obs (S, ds), z (S, d), shared props (K, da) -> (S, K)."""
    if model.kind == "linear":
        return model.scale * torch.einsum("sd,skd->sk", z, model.shared_features(obs, props))
    return model.logits(obs, z, props.expand(obs.shape[0], *props.shape))


def _distance(obs: Tensor) -> Tensor:
    return (obs[..., :2] - obs[..., 2:]).norm(dim=-1)


class Planner:
    """Shared machinery: simulate candidate plans in parallel and score them."""

    def __init__(self, lib, cfg: DownstreamConfig, gen: torch.Generator):
        self.lib, self.cfg, self.gen = lib, cfg, gen
        w = lib.world
        self.sir = Proposals(
            w.act_dim, w.act_low, w.act_high, cfg.proposals, lib.library.device, 29
        )

    def option_action(self, obs: Tensor, z: Tensor) -> Tensor:
        props = self.sir.sample(self.gen)
        u = per_state_logits(self.lib.model, obs, z, props)
        return props[torch.multinomial(u.softmax(-1), 1, generator=self.gen)[:, 0]]

    def cost(self, obs, plan, centers, radii, kind: str) -> Tensor:
        """Roll out N candidate plans per episode: obs (B, 4), plan (B, N, ...) -> cost (B, N)."""
        b, n = plan.shape[:2]
        flat = obs.repeat_interleave(n, 0)
        c, r = centers.repeat_interleave(n, 0), radii.repeat_interleave(n, 0)
        best = _distance(flat)
        for h in range(self.cfg.horizon):
            if kind == "option":
                a = self.option_action(flat, plan.reshape(b * n, -1))
            else:
                a = plan.reshape(b * n, self.cfg.horizon, -1)[:, h]
            flat = step_with_obstacles(self.lib.world, flat, a, c, r)
            best = torch.minimum(best, _distance(flat))
        return (best + 0.1 * _distance(flat)).reshape(b, n)

    def cem(self, obs, centers, radii, dim: int, kind: str, std0: float, lo=None, hi=None):
        b, cfg = obs.shape[0], self.cfg
        mu = torch.zeros(b, dim, device=obs.device)
        std = torch.full((b, dim), std0, device=obs.device)
        n_elite = max(2, int(cfg.elite_frac * cfg.candidates))
        for _ in range(cfg.iters):
            eps = torch.randn(b, cfg.candidates, dim, generator=self.gen, device=obs.device)
            cand = mu[:, None] + std[:, None] * eps
            if lo is not None:
                cand = cand.clamp(lo, hi)
            cost = self.cost(obs, cand, centers, radii, kind)
            idx = cost.topk(n_elite, dim=1, largest=False).indices
            elite = cand.gather(1, idx[..., None].expand(-1, -1, dim))
            mu, std = elite.mean(1), elite.std(1).clamp_min(0.05 * std0)
            best = cand.gather(1, idx[:, :1, None].expand(-1, -1, dim))[:, 0]
        return best


def run(lib, planner: str, cfg: DownstreamConfig, tasks) -> dict:
    """Run one planner on a batch of tasks; returns success rate, time to target, trajectories."""
    gen = torch.Generator(device=lib.library.device).manual_seed(cfg.seed)
    plan = Planner(lib, cfg, gen)
    obs, centers, radii = (t.clone() for t in tasks)
    b, d = obs.shape[0], lib.library.shape[1]
    toward = lib["toward"].expand(b, d)
    k = len(lib.names)
    pairs = [lib.library[[i]] for i in range(k)]
    pairs += [lib.library[[i]] + lib.library[[j]] for i, j in itertools.combinations(range(k), 2)]
    library_set = torch.cat(pairs)  # (C, d)
    reps = max(1, cfg.candidates * cfg.iters // len(library_set))
    done = torch.zeros(b, dtype=torch.bool, device=obs.device)
    t_hit = torch.full((b,), float(cfg.steps), device=obs.device)
    traj = [obs]
    with torch.no_grad():
        for t0 in range(0, cfg.steps, cfg.replan):
            if planner == "greedy":
                z_plan, a_plan = toward, None
            elif planner == "library":
                cand = library_set[None, :, None].expand(b, -1, reps, -1).reshape(b, -1, d)
                cost = plan.cost(obs, cand, centers, radii, "option").reshape(b, -1, reps).mean(-1)
                z_plan, a_plan = library_set[cost.argmin(1)], None
            elif planner == "option-cem":
                z_plan, a_plan = (
                    plan.cem(obs, centers, radii, d, "option", lib.cfg.infer.prior_std),
                    None,
                )
            elif planner == "action-cem":
                seq = plan.cem(obs, centers, radii, cfg.horizon * 2, "action", 0.6, -1.0, 1.0)
                z_plan, a_plan = None, seq.reshape(b, cfg.horizon, 2)
            else:
                raise KeyError(planner)
            for h in range(min(cfg.replan, cfg.steps - t0)):
                a = plan.option_action(obs, z_plan) if a_plan is None else a_plan[:, h]
                nxt = step_with_obstacles(lib.world, obs, a, centers, radii)
                obs = torch.where(done[:, None], obs, nxt)
                hit = (~done) & (_distance(obs) < cfg.tolerance)
                t_hit = torch.where(hit, torch.tensor(float(t0 + h + 1), device=obs.device), t_hit)
                done |= hit
                traj.append(obs)
    success = done.float().mean().item()
    return {
        "success": success,
        "steps_to_target": t_hit[done].mean().item() if done.any() else math.nan,
        "traj": torch.stack(traj, 1),
        "centers": centers,
        "radii": radii,
    }


PLANNERS = ("greedy", "library", "option-cem", "action-cem")


def evaluate_downstream(lib, cfg: DownstreamConfig | None = None) -> dict:
    cfg = cfg or DownstreamConfig()
    gen = torch.Generator(device=lib.library.device).manual_seed(cfg.seed + 1)
    tasks = make_tasks(cfg.episodes, gen, lib.library.device, cfg.task)
    return {name: run(lib, name, cfg, tasks) for name in PLANNERS}
