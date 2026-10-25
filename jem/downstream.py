"""Composing options over time: reach the target around an obstacle never seen in the demos.

Concurrent composition (summing options) is evaluated in `evaluate.py`. Here options
are also sequenced. A planner picks which option, or which sum of options, to run at
each decision, and the energy model executes it. Each episode puts an obstacle between
the agent and the target, so following "toward" alone gets stuck and the planner has to
find a detour.

Every planner is model-predictive control in the true simulator with the same budget of
simulated rollouts per decision: plan over `horizon` steps, execute `replan` steps, repeat.

    greedy      always the "toward" option (no search)
    library     the K discovered options and their pairwise sums (discrete set)
    option-cem  cross-entropy method over z in R^d (the continuous option space)
    action-cem  cross-entropy method over raw action sequences (no options at all)
    oracle      the library planner run with the true experts and their exact products
                (worlds with closed-form experts only): a ceiling for planning over options

With `segments` S > 1 a plan is a sequence of S options, each held for horizon / S steps:
the library and oracle planners then score S-tuples drawn at random from their set (options
and pairwise sums), as many as the rollout budget allows, and option-CEM searches S option
vectors. Sequences let a myopic cost see a detour that pays off only later (up, then over a
wall, then down).

A task holds the per-episode obstacles. It steps simulator states, measures the distance
to the target, and repeats itself for many parallel rollouts per episode.
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
    task: str = "circle"  # particle: circle | cup; panda: bin | pillar
    segments: int = 1  # options per plan, each held for horizon / segments steps


# The arm: 5 s episodes (the bin: 8 s), a 4 cm success radius, more proposals for 7-D
# actions; the bin (up, over the rim, down) is planned as sequences of two options.
ARM_DEFAULTS = {"steps": 100, "tolerance": 0.04, "proposals": 1024}
ARM_TASKS = {"bin": {"steps": 160, "segments": 2}, "pillar": {}}


def disc_tasks(
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


@dc.dataclass
class DiscTask:
    """Particle episodes with disc obstacles (the circle and cup tasks)."""

    world: object
    start: Tensor  # (B, 4) initial states
    centers: Tensor  # (B, O, 2)
    radii: Tensor  # (B, O)

    def repeat(self, n: int) -> DiscTask:
        rep = (t.repeat_interleave(n, 0) for t in (self.start, self.centers, self.radii))
        return DiscTask(self.world, *rep)

    def step(self, state: Tensor, act: Tensor) -> Tensor:
        return step_with_obstacles(self.world, state, act, self.centers, self.radii)

    def distance(self, state: Tensor) -> Tensor:
        return (state[..., :2] - state[..., 2:]).norm(dim=-1)


TASKS = {"particle": ("circle", "cup"), "panda": ("bin", "pillar")}


def make_task(world, kind: str, n: int, gen: torch.Generator):
    if kind not in TASKS.get(world.name, ()):
        raise ValueError(f"task {kind!r} is not defined for {world.name}: {TASKS.get(world.name)}")
    if world.name == "particle":
        return DiscTask(world, *disc_tasks(n, gen, world.device, kind))
    return world.make_task(kind, n, gen)


def per_state_logits(model, obs: Tensor, z: Tensor, props: Tensor) -> Tensor:
    """-E with one option per state: obs (S, ds), z (S, d), shared props (K, da) -> (S, K)."""
    if model.kind == "linear":
        return model.scale * torch.einsum("sd,skd->sk", z, model.shared_features(obs, props))
    return model.logits(obs, z, props.expand(obs.shape[0], *props.shape))


class Planner:
    """Shared machinery: simulate candidate plans in parallel and score them."""

    def __init__(self, lib, cfg: DownstreamConfig, gen: torch.Generator):
        self.lib, self.cfg, self.gen = lib, cfg, gen
        w = self.world = lib.world
        self.sir = Proposals(
            w.act_dim, w.act_low, w.act_high, cfg.proposals, lib.library.device, 29
        )
        self.pairs = [(i, i) for i in range(len(lib.names))]
        self.pairs += list(itertools.combinations(range(len(lib.names)), 2))
        self.pair_index = torch.tensor(self.pairs, device=lib.library.device)

    def option_action(self, state: Tensor, z: Tensor) -> Tensor:
        props = self.sir.sample(self.gen)
        u = per_state_logits(self.lib.model, self.world.obs(state), z, props)
        return props[torch.multinomial(u.softmax(-1), 1, generator=self.gen)[:, 0]]

    def oracle_action(self, state: Tensor, pair: Tensor) -> Tensor:
        """True expert i (pair i == j) or the exact product of experts i and j."""
        ij = self.pair_index[pair]  # (S, 2)
        weights = torch.stack([torch.ones_like(pair), (ij[:, 0] != ij[:, 1]).long()], -1)
        return self.world.product_act(self.world.obs(state), ij, weights.float(), self.gen)

    def act(self, kind: str, state: Tensor, plan: Tensor, h: int) -> Tensor:
        segs = self.cfg.segments
        if segs > 1 and kind != "action":  # the option of the segment that step h is in
            plan = plan[:, min(segs - 1, h * segs // self.cfg.horizon)]
        if kind == "option":
            return self.option_action(state, plan)
        if kind == "oracle":
            return self.oracle_action(state, plan)
        return plan.reshape(plan.shape[0], self.cfg.horizon, -1)[:, h]

    def cost(self, task, state: Tensor, plan: Tensor, kind: str) -> Tensor:
        """Roll out N candidate plans per episode: state (B, ·), plan (B, N, ...) -> cost (B, N)."""
        b, n = plan.shape[:2]
        rep = task.repeat(n)
        flat = state.repeat_interleave(n, 0)
        plan = plan.reshape(b * n, *plan.shape[2:])
        best = rep.distance(flat)
        for h in range(self.cfg.horizon):
            flat = rep.step(flat, self.act(kind, flat, plan, h))
            best = torch.minimum(best, rep.distance(flat))
        return (best + 0.1 * rep.distance(flat)).reshape(b, n)

    def cem(self, task, state, dim: int, kind: str, std0: float, lo=None, hi=None, decode=None):
        """CEM over flat vectors (B, dim); `decode` maps candidates to the plans simulated."""
        b, cfg = state.shape[0], self.cfg
        mu = torch.zeros(b, dim, device=state.device)
        std = torch.full((b, dim), std0, device=state.device)
        n_elite = max(2, int(cfg.elite_frac * cfg.candidates))
        for _ in range(cfg.iters):
            eps = torch.randn(b, cfg.candidates, dim, generator=self.gen, device=state.device)
            cand = mu[:, None] + std[:, None] * eps
            if lo is not None:
                cand = cand.clamp(lo, hi)
            cost = self.cost(task, state, cand if decode is None else decode(cand), kind)
            idx = cost.topk(n_elite, dim=1, largest=False).indices
            elite = cand.gather(1, idx[..., None].expand(-1, -1, dim))
            mu, std = elite.mean(1), elite.std(1).clamp_min(0.05 * std0)
            best = cand.gather(1, idx[:, :1, None].expand(-1, -1, dim))[:, 0]
        return best


def run(lib, planner: str, cfg: DownstreamConfig, task) -> dict:
    """Run one planner on a batch of tasks; returns success rate, time to target, trajectories."""
    gen = torch.Generator(device=lib.library.device).manual_seed(cfg.seed)
    plan = Planner(lib, cfg, gen)
    world = lib.world
    state = task.start.clone()
    b, d = state.shape[0], lib.library.shape[1]
    segs, budget = cfg.segments, cfg.candidates * cfg.iters
    toward = lib["toward"].expand(b, d) if segs == 1 else lib["toward"].expand(b, segs, d)
    library_set = torch.stack([lib.library[i] + (j != i) * lib.library[j] for i, j in plan.pairs])
    reps = max(1, budget // len(library_set)) if segs == 1 else 1
    done = torch.zeros(b, dtype=torch.bool, device=state.device)
    t_hit = torch.full((b,), float(cfg.steps), device=state.device)
    traj = [world.obs(state)]
    with torch.no_grad():
        for t0 in range(0, cfg.steps, cfg.replan):
            kind = "option"
            if planner == "greedy":
                chosen = toward
            elif planner in ("library", "oracle"):
                kind = "option" if planner == "library" else "oracle"
                if segs == 1:  # every option and pairwise sum (oracle: indices into pairs)
                    ids = torch.arange(len(library_set), device=state.device)
                    plans = library_set if planner == "library" else ids
                    cand = plans[None, :, None].expand(b, -1, reps, *plans.shape[1:])
                    cand = cand.reshape(b, -1, *plans.shape[1:])
                    cost = plan.cost(task, state, cand, kind).reshape(b, -1, reps).mean(-1)
                    chosen = plans[cost.argmin(1)]
                else:  # random sequences of options and sums, one rollout each
                    n = len(library_set)
                    ids = torch.randint(n, (b, budget, segs), generator=gen, device=state.device)
                    cand = library_set[ids] if planner == "library" else ids
                    cost = plan.cost(task, state, cand, kind)
                    chosen = cand[torch.arange(b, device=state.device), cost.argmin(1)]
            elif planner == "option-cem":
                decode = None if segs == 1 else (lambda z: z.unflatten(-1, (segs, d)))
                std0 = lib.cfg.infer.prior_std
                z = plan.cem(task, state, segs * d, "option", std0, decode=decode)
                chosen = z if decode is None else decode(z)
            elif planner == "action-cem":
                lo, hi = world.act_low, world.act_high
                if isinstance(lo, Tensor):
                    lo, hi = lo.repeat(cfg.horizon), hi.repeat(cfg.horizon)
                dim = cfg.horizon * world.act_dim
                chosen = plan.cem(task, state, dim, "action", 0.6, lo, hi)
                kind = "action"
            else:
                raise KeyError(planner)
            for h in range(min(cfg.replan, cfg.steps - t0)):
                nxt = task.step(state, plan.act(kind, state, chosen, h))
                state = torch.where(done[:, None], state, nxt)
                hit = (~done) & (task.distance(state) < cfg.tolerance)
                now = torch.tensor(float(t0 + h + 1), device=state.device)
                t_hit = torch.where(hit, now, t_hit)
                done |= hit
                traj.append(world.obs(state))
    success = done.float().mean().item()
    return {
        "success": success,
        "steps_to_target": t_hit[done].mean().item() if done.any() else math.nan,
        "traj": torch.stack(traj, 1),
        "task": task,
    }


PLANNERS = ("greedy", "library", "option-cem", "action-cem")


def evaluate_downstream(lib, cfg: DownstreamConfig | None = None) -> dict:
    cfg = cfg or DownstreamConfig()
    gen = torch.Generator(device=lib.library.device).manual_seed(cfg.seed + 1)
    task = make_task(lib.world, cfg.task, cfg.episodes, gen)
    planners = PLANNERS + (("oracle",) if hasattr(lib.world, "product_act") else ())
    return {name: run(lib, name, cfg, task) for name in planners}
