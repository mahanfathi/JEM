"""A behaviour prior q(a|s): where any demonstrator might act, used to propose actions.

Uniform proposals over the action box stop working as actions get higher-dimensional:
with per-dimension expert noise sigma on a box of width w, only a fraction ~ (2.5 sigma / w)^d_a
of them lands where an expert acts (about 4% for the arm's 7-D experts), so the ranking
objective learns little about the shape of the density where it matters.

q is a mixture of C truncated diagonal Gaussians (a mixture density network) fitted by
maximum likelihood to the unlabelled demonstrations. It sees the state only, never the
context, so it models the marginal over experts and cannot leak the option. JEM uses it
purely as an importance sampler: candidates are drawn from the defensive mixture

    r(a|s) = (1 - rho) q(a|s) + rho U(a),

and every candidate's logit carries the z-independent offset -log r(a) (see objectives.py).
The model's base measure stays uniform, so options and their compositions keep their meaning.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from jem.energy import mlp
from jem.envs import tgauss


class BehaviourPrior(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, low, high, components: int, hidden: int):
        super().__init__()
        self.act_dim, self.components = act_dim, components
        low = torch.as_tensor(low, dtype=torch.float32).expand(act_dim).clone()
        high = torch.as_tensor(high, dtype=torch.float32).expand(act_dim).clone()
        self.register_buffer("low", low)
        self.register_buffer("high", high)
        self.log_volume = float((high - low).log().sum())  # a Python float: no device sync
        self.net = mlp(obs_dim, hidden, 2, components * (1 + 2 * act_dim))

    def params(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Mixture log-weights (..., C), means and stds (..., C, da); means inside the box."""
        c, da = self.components, self.act_dim
        out = self.net(obs)
        logits = out[..., :c].log_softmax(-1)
        raw_mu, raw_std = out[..., c:].unflatten(-1, (c, 2 * da)).split(da, dim=-1)
        width = self.high - self.low
        mu = self.low + width * torch.sigmoid(raw_mu)
        std = width * (0.01 + 0.5 * torch.sigmoid(raw_std))
        return logits, mu, std

    def log_prob(self, obs: Tensor, act: Tensor) -> Tensor:
        """log q(a|s): obs (..., ds) with act (..., da), or candidates (..., K, da) -> (..., K)."""
        logits, mu, std = self.params(obs)
        if act.dim() == obs.dim() + 1:  # a candidate axis: broadcast the mixture over it
            logits, mu, std = logits[..., None, :], mu[..., None, :, :], std[..., None, :, :]
        lp = tgauss.log_prob(act.unsqueeze(-2), mu, std, self.low, self.high)
        return (logits + lp).logsumexp(-1)

    @torch.no_grad()
    def sample(self, obs: Tensor, num: int, gen: torch.Generator | None = None) -> Tensor:
        """num draws per state: obs (..., ds) -> (..., num, da)."""
        logits, mu, std = self.params(obs)
        flat = logits.reshape(-1, self.components).exp()
        comp = torch.multinomial(flat, num, replacement=True, generator=gen)
        comp = comp.reshape(*logits.shape[:-1], num)
        idx = comp[..., None].expand(*comp.shape, self.act_dim)
        mu_k, std_k = mu.gather(-2, idx), std.gather(-2, idx)
        return tgauss.sample(mu_k, std_k, self.low, self.high, gen)

    def log_mixture(self, obs: Tensor, act: Tensor, uniform_frac: float) -> Tensor:
        """log r(a|s) = log((1 - rho) q(a|s) + rho / volume) for candidates (..., K, da)."""
        lq = self.log_prob(obs, act) + math.log1p(-uniform_frac)
        lu = math.log(uniform_frac) - self.log_volume
        return torch.logaddexp(lq, torch.full_like(lq, lu))
