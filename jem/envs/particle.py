"""2-D particle-and-target world with stochastic scripted experts (batched, torch).

State  s = (agent_xy, target_xy) in R^4.  Action a in [-1, 1]^2 is a velocity:
agent <- clip(agent + dt * a, -arena, arena); the target never moves.

Each expert k steers at a fixed angle theta_k relative to the bearing to the target
(toward: 0, cw orbit: +pi/2, away: pi, ccw orbit: -pi/2), with Gaussian heading noise
of std `angle_std` and speed ~ U[speed_min, speed_max]. The noise makes a single
transition ambiguous, so recognising an expert needs evidence from several steps.
Because the expert densities are known in closed form, models can be scored by the
true KL(expert || model) rather than by action L2.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from jem.config import EnvConfig

# Steering +90 degrees off the bearing to the target circles it clockwise.
EXPERT_ANGLES = {"toward": 0.0, "cw": math.pi / 2, "away": math.pi, "ccw": -math.pi / 2}


def wrap(angle: Tensor) -> Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


class ParticleWorld:
    name = "particle"
    obs_dim = 4
    state_dim = 4  # the observation is the whole state
    act_dim = 2
    act_low = -1.0
    act_high = 1.0

    def __init__(self, cfg: EnvConfig, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.expert_names = tuple(cfg.experts)
        self.expert_angles = torch.tensor(
            [EXPERT_ANGLES[name] for name in cfg.experts], device=self.device
        )
        # E[cos(heading noise)]: how well the experts' own actions agree with their mean.
        self.agreement_ceiling = math.exp(-0.5 * cfg.angle_std**2)

    @property
    def num_experts(self) -> int:
        return len(self.expert_names)

    def obs(self, state: Tensor) -> Tensor:
        return state

    def effect(self, obs: Tensor, act: Tensor) -> Tensor:
        """What an action does in task space: here the velocity itself."""
        return act

    def expert_effect(self, obs: Tensor, expert: Tensor) -> Tensor:
        """Direction of the expert's mean effect (a unit vector)."""
        return self.expert_mean_dir(obs, expert)

    def expert_pairs(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """(pairs 90 degrees apart, antipodal pairs)."""
        adjacent, antipodal = [], []
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                gap = abs(wrap(self.expert_angles[i] - self.expert_angles[j]).item())
                if abs(gap - math.pi / 2) < 1e-3:
                    adjacent.append((i, j))
                elif abs(gap - math.pi) < 1e-3:
                    antipodal.append((i, j))
        return adjacent, antipodal

    def reset(self, n: int, gen: torch.Generator | None = None) -> Tensor:
        cfg = self.cfg
        target = (torch.rand(n, 2, generator=gen, device=self.device) * 2 - 1) * cfg.target_range
        agent = (torch.rand(n, 2, generator=gen, device=self.device) * 2 - 1) * cfg.arena
        for _ in range(100):  # resample agents that start on top of their target
            close = (agent - target).norm(dim=-1) < cfg.min_dist
            if not close.any():
                break
            fresh = (torch.rand(n, 2, generator=gen, device=self.device) * 2 - 1) * cfg.arena
            agent = torch.where(close[:, None], fresh, agent)
        return torch.cat([agent, target], dim=-1)

    def step(self, state: Tensor, act: Tensor) -> Tensor:
        agent, target = state[..., :2], state[..., 2:]
        agent = agent + self.cfg.dt * act.clamp(self.act_low, self.act_high)
        return torch.cat([agent.clamp(-self.cfg.arena, self.cfg.arena), target], dim=-1)

    def bearing(self, obs: Tensor) -> Tensor:
        delta = obs[..., 2:] - obs[..., :2]
        return torch.atan2(delta[..., 1], delta[..., 0])

    def expert_heading(self, obs: Tensor, expert: Tensor) -> Tensor:
        return wrap(self.bearing(obs) + self.expert_angles[expert])

    def expert_mean_dir(self, obs: Tensor, expert: Tensor) -> Tensor:
        """Unit vector of the expert's mean heading at each state."""
        heading = self.expert_heading(obs, expert)
        return torch.stack([heading.cos(), heading.sin()], dim=-1)

    def expert_act(self, obs: Tensor, expert: Tensor, gen: torch.Generator | None = None) -> Tensor:
        cfg = self.cfg
        shape = obs.shape[:-1]
        noise = torch.randn(shape, generator=gen, device=self.device) * cfg.angle_std
        heading = self.expert_heading(obs, expert) + noise
        u = torch.rand(shape, generator=gen, device=self.device)
        speed = cfg.speed_min + (cfg.speed_max - cfg.speed_min) * u
        return speed[..., None] * torch.stack([heading.cos(), heading.sin()], dim=-1)

    def expert_log_prob(self, obs: Tensor, expert: Tensor, act: Tensor) -> Tensor:
        """Exact log-density of `act` under the expert, w.r.t. Lebesgue measure on R^2."""
        cfg = self.cfg
        speed = act.norm(dim=-1)
        delta = wrap(torch.atan2(act[..., 1], act[..., 0]) - self.expert_heading(obs, expert))
        var = cfg.angle_std**2
        wraps = torch.arange(-3, 4, device=act.device, dtype=act.dtype) * (2 * math.pi)
        log_heading = torch.logsumexp(
            -((delta[..., None] + wraps) ** 2) / (2 * var), dim=-1
        ) - 0.5 * math.log(2 * math.pi * var)
        # Relative slack: a sampled speed of exactly speed_min can round below it in fp32.
        inside = (speed >= cfg.speed_min * (1 - 1e-6)) & (speed <= cfg.speed_max * (1 + 1e-6))
        log_speed = torch.where(
            inside,
            torch.full_like(speed, -math.log(cfg.speed_max - cfg.speed_min)),
            torch.full_like(speed, -math.inf),
        )
        return log_heading + log_speed - speed.clamp_min(1e-12).log()

    def ideal_composition_dir(self, obs: Tensor, i: int, j: int) -> Tensor:
        """Mean heading of the product of experts i and j (exact for equal-noise von Mises).

        For experts 90 degrees apart this is the bisector. For antipodal experts the
        product has no preferred heading, so the returned vector is ~0.
        """
        idx = torch.full(obs.shape[:-1], i, device=obs.device, dtype=torch.long)
        jdx = torch.full(obs.shape[:-1], j, device=obs.device, dtype=torch.long)
        return 0.5 * (self.expert_mean_dir(obs, idx) + self.expert_mean_dir(obs, jdx))
