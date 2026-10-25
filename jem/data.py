"""Unlabelled demonstrations: collection and window sampling.

A world separates its simulator *state* (everything needed to continue the simulation,
e.g. joint positions and velocities) from the *observation* the policy sees (the expert's
inputs). For the particle the two coincide. Demonstrations keep both, so evaluation can
restart the simulator from any visited state.

Expert ids are stored only for evaluation; training never reads them.
"""

from __future__ import annotations

import dataclasses as dc

import torch
from torch import Tensor


@dc.dataclass
class Demos:
    obs: Tensor  # (N, T, obs_dim)
    act: Tensor  # (N, T, act_dim)
    expert: Tensor  # (N, T) int64, evaluation only
    state: Tensor  # (N, T, state_dim) simulator state before each action

    def __len__(self) -> int:
        return self.obs.shape[0]

    def to(self, device) -> Demos:
        return Demos(*(t.to(device) for t in (self.obs, self.act, self.expert, self.state)))


@dc.dataclass
class Window:
    obs: Tensor  # (B, L, obs_dim)
    act: Tensor  # (B, L, act_dim)
    expert: Tensor  # (B, L)
    gap: Tensor | None = None  # (B,) steps skipped between the context and the targets


@torch.no_grad()
def collect(world, n: int, horizon: int, gen: torch.Generator, switch_prob: float = 0.0) -> Demos:
    state = world.reset(n, gen)
    k = world.num_experts
    expert = torch.randint(k, (n,), generator=gen, device=world.device)
    obs_t, act_t, exp_t, state_t = [], [], [], []
    for _ in range(horizon):
        if switch_prob > 0:
            switch = torch.rand(n, generator=gen, device=world.device) < switch_prob
            other = (expert + torch.randint(1, k, (n,), generator=gen, device=world.device)) % k
            expert = torch.where(switch, other, expert)
        obs = world.obs(state)
        act = world.expert_act(obs, expert, gen)
        obs_t.append(obs)
        act_t.append(act)
        exp_t.append(expert)
        state_t.append(state)
        state = world.step(state, act)
    stack = [torch.stack(t, 1) for t in (obs_t, act_t, exp_t, state_t)]
    return Demos(*stack)


def sample_windows(
    demos: Demos,
    batch: int,
    length: int,
    gen: torch.Generator,
    context: int = 0,
    max_gap: int = 0,
) -> Window:
    """Random windows of `length` steps, each from one demonstration.

    With max_gap > 0 the window splits after `context` steps, and the rest (the targets)
    starts a uniform 0..max_gap steps later in the same demonstration: far enough that the
    state has moved on, so only what persists (the option) helps predict the targets.
    """
    n, horizon = demos.expert.shape
    device = demos.obs.device
    traj = torch.randint(n, (batch,), generator=gen, device=device)
    max_gap = min(max_gap, horizon - length) if context else 0
    steps = torch.arange(length, device=device)
    if max_gap:
        gap = torch.randint(max_gap + 1, (batch,), generator=gen, device=device)
        room = (horizon - length - gap + 1).float()
        start = (torch.rand(batch, generator=gen, device=device) * room).long()
        t = start[:, None] + steps + (steps >= context) * gap[:, None]
    else:
        gap = None
        start = torch.randint(horizon - length + 1, (batch,), generator=gen, device=device)
        t = start[:, None] + steps
    obs, act = demos.obs[traj[:, None], t], demos.act[traj[:, None], t]
    return Window(obs, act, demos.expert[traj[:, None], t], gap)
