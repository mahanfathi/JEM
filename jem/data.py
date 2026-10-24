"""Unlabelled demonstrations: collection and window sampling.

Expert ids are stored only for evaluation; training never reads them.
"""

from __future__ import annotations

import dataclasses as dc

import torch
from torch import Tensor

from jem.envs import ParticleWorld


@dc.dataclass
class Demos:
    obs: Tensor  # (N, T, obs_dim)
    act: Tensor  # (N, T, act_dim)
    expert: Tensor  # (N, T) int64, evaluation only

    def __len__(self) -> int:
        return self.obs.shape[0]

    def to(self, device) -> Demos:
        return Demos(self.obs.to(device), self.act.to(device), self.expert.to(device))


@dc.dataclass
class Window:
    obs: Tensor  # (B, L, obs_dim)
    act: Tensor  # (B, L, act_dim)
    expert: Tensor  # (B, L)


@torch.no_grad()
def collect(
    world: ParticleWorld, n: int, horizon: int, gen: torch.Generator, switch_prob: float = 0.0
) -> Demos:
    obs = world.reset(n, gen)
    k = world.num_experts
    expert = torch.randint(k, (n,), generator=gen, device=world.device)
    obs_t, act_t, exp_t = [], [], []
    for _ in range(horizon):
        if switch_prob > 0:
            switch = torch.rand(n, generator=gen, device=world.device) < switch_prob
            other = (expert + torch.randint(1, k, (n,), generator=gen, device=world.device)) % k
            expert = torch.where(switch, other, expert)
        act = world.expert_act(obs, expert, gen)
        obs_t.append(obs)
        act_t.append(act)
        exp_t.append(expert)
        obs = world.step(obs, act)
    return Demos(torch.stack(obs_t, 1), torch.stack(act_t, 1), torch.stack(exp_t, 1))


def sample_windows(demos: Demos, batch: int, length: int, gen: torch.Generator) -> Window:
    n, horizon = demos.expert.shape
    device = demos.obs.device
    traj = torch.randint(n, (batch,), generator=gen, device=device)
    start = torch.randint(horizon - length + 1, (batch,), generator=gen, device=device)
    t = start[:, None] + torch.arange(length, device=device)
    return Window(
        demos.obs[traj[:, None], t], demos.act[traj[:, None], t], demos.expert[traj[:, None], t]
    )
