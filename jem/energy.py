"""Energy functions E(s, z, a) over state s, option z and action a.

OptionLinearEnergy   E(s, z, a) = -tau <z, phi(s, a)>,   ||phi(s, a)|| = 1

    p(a | s, z) ∝ exp(-E) is an exponential family over actions whose natural
    parameter is the option z (sufficient statistics phi are learned). Hence:
      * composition is vector addition:   E(s, z1, a) + E(s, z2, a) = E(s, z1 + z2, a),
        i.e. the product of experts p(a|s,z1) p(a|s,z2) is again an option;
      * -log p(a_1:N | s_1:N, z) is convex in z (a log-partition function), so option
        inference has a unique optimum that Newton's method finds exactly;
      * |E| <= tau ||z||, so the network cannot inflate the energy scale; sharpness is
        carried by ||z|| (regularised by the prior) and one learned temperature tau.

    phi comes in two architectures (`ModelConfig.arch`):
      joint     phi = MLP([s, a]): fully general, but the network runs on every
                (state, candidate action) pair, which dominates training cost;
      factored  phi = normalize(W(s) psi(a)), W(s) in R^{d x p}: two towers that run once
                per state and once per candidate, joined by a batched matmul. A rank-p
                separable approximation of the joint form; still linear in z.

ConcatEnergy         E(s, z, a) = f([s, z, a])  (optionally f^2): the 2022 form, kept as
                     the baseline. Nothing ties its z-dependence to composition or makes
                     inference over z well posed.

Both expose `logits(obs, z, act) = -E` over a trailing axis of K candidate actions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from jem.config import ModelConfig


def mlp(in_dim: int, hidden: int, depth: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


def with_candidates(act: Tensor, proposals: Tensor) -> Tensor:
    """Observed action at index 0, then shared proposals: (..., da), (M, da) -> (..., 1+M, da)."""
    props = proposals.expand(*act.shape[:-1], *proposals.shape)
    return torch.cat([act.unsqueeze(-2), props], dim=-2)


class OptionLinearEnergy(nn.Module):
    kind = "linear"

    def __init__(self, obs_dim: int, act_dim: int, cfg: ModelConfig):
        super().__init__()
        if cfg.depth < 1:
            raise ValueError("depth must be >= 1")
        self.arch = cfg.arch
        self.option_dim = d = cfg.option_dim
        self.feature_norm = cfg.feature_norm
        if cfg.arch == "joint":
            # First layer split so the state half is computed once per state.
            self.obs_in = nn.Linear(obs_dim, cfg.hidden)
            self.act_in = nn.Linear(act_dim, cfg.hidden, bias=False)
            self.trunk = nn.Sequential(nn.SiLU(), mlp(cfg.hidden, cfg.hidden, cfg.depth - 1, d))
        elif cfg.arch == "factored":
            self.rank = p = cfg.action_features
            self.state_net = mlp(obs_dim, cfg.hidden, cfg.depth, d * p)
            self.action_net = mlp(act_dim, cfg.hidden, cfg.depth, p)
        else:
            raise KeyError(f"unknown arch {cfg.arch!r}")
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(cfg.init_scale)), requires_grad=cfg.learn_scale
        )

    @property
    def scale(self) -> Tensor:
        return self.log_scale.exp()

    def _normalize(self, f: Tensor) -> Tensor:
        if self.feature_norm == "sphere":
            return f * torch.rsqrt(f.pow(2).sum(-1, keepdim=True) + 1e-6)
        if self.feature_norm == "tanh":
            return torch.tanh(f)
        if self.feature_norm == "none":
            return f
        raise ValueError(f"unknown feature_norm {self.feature_norm!r}")

    def _state_matrix(self, obs: Tensor) -> Tensor:
        return self.state_net(obs).unflatten(-1, (self.option_dim, self.rank))

    def features(self, obs: Tensor, act: Tensor) -> Tensor:
        """phi(s, a): obs (..., ds) with act (..., da) or candidates (..., K, da)."""
        candidates = act.dim() == obs.dim() + 1
        if self.arch == "joint":
            hs = self.obs_in(obs)
            f = self.trunk((hs.unsqueeze(-2) if candidates else hs) + self.act_in(act))
        elif candidates:
            f = torch.einsum("...dp,...kp->...kd", self._state_matrix(obs), self.action_net(act))
        else:
            f = torch.einsum("...dp,...p->...d", self._state_matrix(obs), self.action_net(act))
        return self._normalize(f)

    def shared_features(self, obs: Tensor, act: Tensor) -> Tensor:
        """phi for one candidate set shared by all states: (S, ds), (K, da) -> (S, K, d)."""
        if self.arch == "joint":
            return self.features(obs, act.expand(obs.shape[0], *act.shape))
        f = torch.einsum("sdp,kp->skd", self._state_matrix(obs), self.action_net(act))
        return self._normalize(f)

    def candidate_features(self, obs: Tensor, act: Tensor, proposals: Tensor) -> Tensor:
        """(B, N, ds), (B, N, da), (M, da) -> (B, N, 1+M, d); index 0 is the observed action."""
        if self.arch == "joint":
            return self.features(obs, with_candidates(act, proposals))
        w = self._state_matrix(obs)
        pos = torch.einsum("...dp,...p->...d", w, self.action_net(act))
        props = torch.einsum("...dp,kp->...kd", w, self.action_net(proposals))
        return self._normalize(torch.cat([pos.unsqueeze(-2), props], dim=-2))

    def logits(self, obs: Tensor, z: Tensor, act: Tensor) -> Tensor:
        """-E for K candidates: obs (..., ds), z (..., d), act (..., K, da) -> (..., K)."""
        return self.scale * torch.einsum("...d,...kd->...k", z, self.features(obs, act))


class ConcatEnergy(nn.Module):
    kind = "mlp"

    def __init__(self, obs_dim: int, act_dim: int, cfg: ModelConfig):
        super().__init__()
        self.option_dim = cfg.option_dim
        self.squared = cfg.squared
        self.net = mlp(obs_dim + cfg.option_dim + act_dim, cfg.hidden, cfg.depth, 1)

    def forward(self, obs: Tensor, z: Tensor, act: Tensor) -> Tensor:
        """E(s, z, a); if act carries an extra candidate axis, obs and z are broadcast over it."""
        if act.dim() == obs.dim() + 1:
            obs = obs.unsqueeze(-2).expand(*act.shape[:-1], obs.shape[-1])
            z = z.unsqueeze(-2).expand(*act.shape[:-1], z.shape[-1])
        f = self.net(torch.cat([obs, z, act], dim=-1)).squeeze(-1)
        return f.square() if self.squared else f

    energy = forward

    def logits(self, obs: Tensor, z: Tensor, act: Tensor) -> Tensor:
        return -self.forward(obs, z, act)


def build_model(cfg: ModelConfig, obs_dim: int, act_dim: int) -> nn.Module:
    if cfg.kind == "linear":
        return OptionLinearEnergy(obs_dim, act_dim, cfg)
    if cfg.kind == "mlp":
        return ConcatEnergy(obs_dim, act_dim, cfg)
    raise KeyError(f"unknown model kind {cfg.kind!r}")
