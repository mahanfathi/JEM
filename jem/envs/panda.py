"""Franka Panda arm with scripted task-space experts (MuJoCo Warp physics, torch experts).

State  s = (q, qdot, g) in R^17: joint positions and velocities, and a target point g.
Obs    o = (q, x, g, g - x) in R^16 with x the end-effector position (torch kinematics):
       exactly the experts' inputs. Velocities are left out because they would reveal the
       active option without it ever having to be inferred.
Action a in [-1, 1]^7: a joint velocity command qd = a * qd_max. The position servos
       (kp / kv = 1 / tau) track it when their targets lead the measured joints by tau qd:
       each control step sets ctrl = q + tau qd for one period dt (frame_skip physics
       steps). Gravity is compensated, as on the real arm.

The experts move the end effector in a cylindrical frame around the target: radially
(toward, away; horizontal), tangentially (cw, ccw about the vertical axis through g) or
vertically (up, down), at speed v. A task-space velocity v_k becomes a joint velocity by
damped least squares, plus a posture term in the Jacobian null space that all experts
share and that therefore carries no option information:

    mu_k(q, g) = sat( [J^T (J J^T + lam^2 I)^-1 v_k + (I - J^+ J) c (q_home - q)] / qd_max )
    a ~ TN(mu_k, sigma^2 I) on [-1, 1]^7      (truncated, never clipped: exact densities)

With one shared sigma, every weighted product of experts is again a truncated Gaussian
(jem/envs/tgauss.py), so compositions have exact, closed-form references.
"""

from __future__ import annotations

import dataclasses as dc
import math
from pathlib import Path

import mujoco
import torch
from torch import Tensor

from jem.config import EnvConfig
from jem.envs import tgauss

ASSET = Path(__file__).parent / "assets" / "panda.xml"
EXPERTS = ("toward", "away", "cw", "ccw", "up", "down")
ANTIPODES = (("toward", "away"), ("cw", "ccw"), ("up", "down"))
VELOCITY_LIMITS = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)  # rad/s, Franka datasheet
SATURATION = 0.95  # expert means stay inside the action box


def soft_clip(x: Tensor, limit: float, sharpness: float = 4.0) -> Tensor:
    """Close to the identity well inside [-limit, limit], and never beyond it."""
    return x / (1 + (x.abs() / limit) ** sharpness) ** (1 / sharpness)


class PandaWorld:
    name = "panda"
    obs_dim = 16
    state_dim = 17
    act_dim = 7

    def __init__(self, cfg: EnvConfig, device: torch.device | str = "cuda"):
        from jem.envs.kinematics import SerialChain
        from jem.envs.mjw import MjwSim

        self.cfg, self.pc = cfg, cfg.panda
        self.device = torch.device(device)
        unknown = set(cfg.experts) - set(EXPERTS)
        if unknown:
            raise KeyError(f"unknown panda experts {sorted(unknown)}; choose from {EXPERTS}")
        self.expert_names = tuple(cfg.experts)
        self.mjm = mujoco.MjModel.from_xml_path(str(ASSET))
        self.frame_skip = round(cfg.dt / self.mjm.opt.timestep)
        if abs(self.frame_skip * self.mjm.opt.timestep - cfg.dt) > 1e-9:
            raise ValueError(f"dt={cfg.dt} is not a multiple of {self.mjm.opt.timestep}")
        self.sim = MjwSim(self.mjm, self.device, self.frame_skip)
        self.chain = SerialChain(self.mjm, "ee", self.device)

        dev = self.device
        self.act_low = torch.full((7,), -1.0, device=dev)
        self.act_high = torch.ones(7, device=dev)
        self.qd_max = self.pc.qd_scale * torch.tensor(VELOCITY_LIMITS, device=dev)
        # Servo lead time tau = kv / kp, from the actuators' affine bias (0, -kp, -kv).
        bias = torch.tensor(self.mjm.actuator_biasprm[:7, 1:3], device=dev, dtype=torch.float32)
        self.lead = bias[:, 1] / bias[:, 0]
        self.q_home = torch.tensor(self.mjm.key("home").qpos[:7], device=dev, dtype=torch.float32)
        ranges = torch.tensor(self.mjm.jnt_range[:7], device=dev, dtype=torch.float32)
        self.q_lo, self.q_hi = ranges[:, 0], ranges[:, 1]
        self.std = torch.full((7,), self.pc.act_std, device=dev)
        self.target_lo = torch.tensor(self.pc.target_lo, device=dev)
        self.target_hi = torch.tensor(self.pc.target_hi, device=dev)

    @property
    def num_experts(self) -> int:
        return len(self.expert_names)

    # --------------------------------------------------------------------- state and dynamics

    def reset(self, n: int, gen: torch.Generator | None = None) -> Tensor:
        dev, pc = self.device, self.pc
        noise = (torch.rand(n, 7, generator=gen, device=dev) * 2 - 1) * pc.init_noise
        margin = 0.05
        q = (self.q_home + noise).clamp(self.q_lo + margin, self.q_hi - margin)
        x = self.chain.site(q)
        span = self.target_hi - self.target_lo
        g = self.target_lo + torch.rand(n, 3, generator=gen, device=dev) * span
        for _ in range(100):  # resample targets that start on top of the end effector
            close = (x - g).norm(dim=-1) < self.cfg.min_dist
            if not close.any():
                break
            fresh = self.target_lo + torch.rand(n, 3, generator=gen, device=dev) * span
            g = torch.where(close[:, None], fresh, g)
        return torch.cat([q, torch.zeros_like(q), g], dim=-1)

    def obs(self, state: Tensor) -> Tensor:
        q, g = state[..., :7], state[..., 14:17]
        x = self.chain.site(q)
        return torch.cat([q, x, g, g - x], dim=-1)

    def step(self, state: Tensor, act: Tensor) -> Tensor:
        return self.step_in(self.sim, state, act)

    def step_in(self, sim, state: Tensor, act: Tensor, mocap_pos: Tensor | None = None) -> Tensor:
        """One control period in `sim` (free space, or a scene with obstacles)."""
        q, qd, g = state[..., :7], state[..., 7:14], state[..., 14:17]
        lead = self.lead * act.clamp(-1.0, 1.0) * self.qd_max
        ctrl = q + lead  # MuJoCo clamps the targets to the joint range
        q, qd = sim.step(q.contiguous(), qd.contiguous(), ctrl.contiguous(), mocap_pos)
        return torch.cat([q, qd, g], dim=-1)

    # --------------------------------------------------------------------------- the experts

    def expert_velocities(self, x: Tensor, g: Tensor) -> Tensor:
        """Task-space velocity of every expert: (..., 3), (..., 3) -> (..., K, 3)."""
        r = x - g
        rh = torch.stack([r[..., 0], r[..., 1], torch.zeros_like(r[..., 0])], -1)
        rho = (rh.square().sum(-1, keepdim=True) + self.pc.axis_radius**2).sqrt()
        radial = rh / rho  # fades to zero on the vertical axis through the target
        ccw = torch.stack([-rh[..., 1], rh[..., 0], torch.zeros_like(rh[..., 0])], -1) / rho
        up = torch.zeros_like(r)
        up[..., 2] = 1.0
        dirs = {"toward": -radial, "away": radial, "ccw": ccw, "cw": -ccw, "up": up, "down": -up}
        return self.pc.ee_speed * torch.stack([dirs[n] for n in self.expert_names], -2)

    def expert_means(self, obs: Tensor) -> Tensor:
        """Mean action of every expert: (..., 16) -> (..., K, 7)."""
        q, x, g = obs[..., :7], obs[..., 7:10], obs[..., 10:13]
        jac = self.chain.jacobian(q)  # (..., 3, 7)
        eye = torch.eye(3, device=obs.device, dtype=obs.dtype)
        gram = jac @ jac.mT + self.pc.damping**2 * eye
        v = self.expert_velocities(x, g)
        task = (jac.mT @ torch.linalg.solve(gram, v.mT)).mT  # (..., K, 7) joint velocities
        pull = self.pc.posture_gain * (self.q_home - q)
        null = pull - (jac.mT @ torch.linalg.solve(gram, (jac @ pull[..., None])))[..., 0]
        return soft_clip((task + null[..., None, :]) / self.qd_max, SATURATION)

    def _means_of(self, obs: Tensor, expert: Tensor) -> Tensor:
        """Mean action of the given experts; expert (...) broadcasts against obs (..., 16)."""
        mu = self.expert_means(obs)
        shape = torch.broadcast_shapes(mu.shape[:-2], expert.shape)
        mu = mu.expand(*shape, *mu.shape[-2:])
        idx = expert.expand(shape)[..., None, None].expand(*shape, 1, mu.shape[-1])
        return mu.gather(-2, idx).squeeze(-2)

    def expert_act(self, obs: Tensor, expert: Tensor, gen: torch.Generator | None = None) -> Tensor:
        mu = self._means_of(obs, expert)
        return tgauss.sample(mu, self.std.expand_as(mu), self.act_low, self.act_high, gen)

    def expert_log_prob(self, obs: Tensor, expert: Tensor, act: Tensor) -> Tensor:
        """Exact log-density of `act` under the expert, w.r.t. Lebesgue measure on the box."""
        mu = self._means_of(obs, expert)
        return tgauss.log_prob(act, mu, self.std, self.act_low, self.act_high)

    def effect(self, obs: Tensor, act: Tensor) -> Tensor:
        """Commanded end-effector velocity of an action, J(q) qd in m/s: (..., 3)."""
        jac = self.chain.jacobian(obs[..., :7])
        return torch.einsum("...ij,...j->...i", jac, act * self.qd_max)

    def expert_effect(self, obs: Tensor, expert: Tensor) -> Tensor:
        mu = self._means_of(obs, expert)
        return self.effect(obs, tgauss.mean(mu, self.std, self.act_low, self.act_high))

    def expert_pairs(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """(orthogonal pairs, antipodal pairs) of experts."""
        names = self.expert_names
        antipodal = [
            tuple(sorted((names.index(a), names.index(b))))
            for a, b in ANTIPODES
            if a in names and b in names
        ]
        pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
        return [p for p in pairs if p not in antipodal], antipodal

    # ------------------------------------------------------------- exact products of experts

    def _product(self, obs: Tensor, experts, weights) -> tuple[Tensor, Tensor]:
        """Untruncated (mean, std) of prod_p TN(mu_{e_p}, sigma)^{w_p}, over (..., P)."""
        experts = torch.as_tensor(experts, device=obs.device)
        weights = torch.as_tensor(weights, device=obs.device, dtype=obs.dtype)
        mu = self._means_of(obs[..., None, :], experts)  # (..., P, 7)
        w = weights[..., None]
        total = w.sum(-2)
        return (w * mu).sum(-2) / total, self.std / total.sqrt()

    def product_mean(self, obs: Tensor, experts, weights) -> Tensor:
        mu, std = self._product(obs, experts, weights)
        return tgauss.mean(mu, std, self.act_low, self.act_high)

    def product_act(self, obs: Tensor, experts, weights, gen=None) -> Tensor:
        mu, std = self._product(obs, experts, weights)
        return tgauss.sample(mu, std.expand_as(mu), self.act_low, self.act_high, gen)

    def product_log_prob(self, obs: Tensor, experts, weights, act: Tensor) -> Tensor:
        mu, std = self._product(obs, experts, weights)
        return tgauss.log_prob(act, mu, std, self.act_low, self.act_high)

    # -------------------------------------------------------------------- downstream tasks

    def ik(self, target: Tensor, q: Tensor | None = None, iters: int = 200) -> Tensor:
        """Joint positions placing the end effector at `target` (damped least squares)."""
        q = self.q_home.expand(len(target), 7).clone() if q is None else q.clone()
        eye = torch.eye(3, device=target.device)
        for _ in range(iters):
            jac = self.chain.jacobian(q)
            err = target - self.chain.site(q)
            gram = jac @ jac.mT + 0.01**2 * eye
            dq = (jac.mT @ torch.linalg.solve(gram, err[..., None]))[..., 0]
            pull = 0.1 * (self.q_home - q)
            pull = pull - (jac.mT @ torch.linalg.solve(gram, jac @ pull[..., None]))[..., 0]
            q = (q + (dq + pull).clamp(-0.2, 0.2)).clamp(self.q_lo + 0.02, self.q_hi - 0.02)
        return q

    def make_task(self, kind: str, n: int, gen: torch.Generator) -> PandaTask:
        """Episodes with an obstacle between the end effector and the target.

        bin     the target sits in an open box on a pedestal, below the rim, and the end
                effector starts outside it, also below the rim: go up, over, and down
        pillar  a pillar taller than the arm's reach stands halfway to the target
        """
        from jem.envs.mjw import MjwSim

        dev = self.device
        g = torch.empty(n, 3, device=dev)
        start = torch.empty(n, 3, device=dev)
        q = torch.empty(n, 7, device=dev)
        todo = torch.ones(n, dtype=torch.bool, device=dev)
        for _ in range(20):  # resample episodes whose start the arm cannot reach
            m = int(todo.sum())
            gi, si = self._episode(kind, m, gen)
            qi = self.ik(si)
            g[todo], start[todo], q[todo] = gi, si, qi
            reached = (self.chain.site(q) - start).norm(dim=-1) < 0.01
            todo = ~reached
            if not todo.any():
                break
        if todo.any():
            raise RuntimeError(f"{int(todo.sum())} {kind} starts are out of reach")
        mocap = torch.cat([g[:, :2], torch.zeros(n, 1, device=dev)], -1)
        if kind == "pillar":  # halfway between start and target
            mocap[:, :2] = 0.5 * (g[:, :2] + start[:, :2])
        state = torch.cat([q, torch.zeros_like(q), g], -1)
        sim = MjwSim(obstacle_scene(kind), dev, self.frame_skip)
        return PandaTask(self, sim, state, mocap[:, None, :])

    def _episode(self, kind: str, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """Target and end-effector start positions; the start is offset sideways (along the
        tangent of the target's horizontal position) so that both stay in reach."""
        dev = self.device
        lo, hi = self.target_lo, self.target_hi
        g = lo + torch.rand(n, 3, generator=gen, device=dev) * (hi - lo)
        tangent = torch.stack([-g[:, 1], g[:, 0]], -1) / g[:, :2].norm(dim=-1, keepdim=True)
        sign = torch.randint(2, (n, 1), generator=gen, device=dev) * 2 - 1
        angle = (torch.rand(n, generator=gen, device=dev) * 2 - 1) * math.radians(30)
        c, s = angle.cos()[:, None], angle.sin()[:, None]
        t = sign * tangent
        offset = torch.cat([c * t[:, :1] - s * t[:, 1:], s * t[:, :1] + c * t[:, 1:]], -1)
        if kind == "bin":
            g[:, 2] = BIN_FLOOR + 0.06
            height = torch.full((n, 1), BIN_FLOOR + 0.1, device=dev)  # below the rim
            return g, torch.cat([g[:, :2] + 0.26 * offset, height], -1)
        if kind == "pillar":
            return g, torch.cat([g[:, :2] + 0.28 * offset, g[:, 2:]], -1)
        raise KeyError(f"unknown panda task {kind!r}")


BIN_FLOOR = 0.3  # the bin stands on a pedestal, so its target lies in the training range


def obstacle_scene(kind: str) -> mujoco.MjModel:
    """The arm plus one mocap obstacle body, placed per episode through mocap_pos."""
    return obstacle_spec(kind).compile()


def obstacle_spec(kind: str, spec: mujoco.MjSpec | None = None) -> mujoco.MjSpec:
    """Add the task's obstacle, a mocap body, to `spec` (default: the arm alone)."""
    spec = mujoco.MjSpec.from_file(str(ASSET)) if spec is None else spec
    body = spec.worldbody.add_body(name="obstacle", mocap=True)
    style = dict(contype=0, conaffinity=1, rgba=[0.55, 0.55, 0.52, 1])
    box = mujoco.mjtGeom.mjGEOM_BOX
    if kind == "bin":  # inner half-width w, wall thickness t, wall height h; open at the top
        w, t, h = 0.09, 0.015, 0.2
        outer = w + 2 * t
        pedestal = [outer, outer, BIN_FLOOR / 2]
        body.add_geom(type=box, size=pedestal, pos=[0, 0, BIN_FLOOR / 2], **style)
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            size = [t, outer, h / 2] if sx else [outer, t, h / 2]
            pos = [sx * (w + t), sy * (w + t), BIN_FLOOR + h / 2]
            body.add_geom(type=box, size=size, pos=pos, **style)
    elif kind == "pillar":  # taller than the arm can reach over
        cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
        body.add_geom(type=cyl, size=[0.05, 0.6, 0], pos=[0, 0, 0.6], **style)
    else:
        raise KeyError(f"unknown panda task {kind!r}")
    return spec


@dc.dataclass
class PandaTask:
    world: PandaWorld
    sim: object  # MjwSim of the obstacle scene
    start: Tensor  # (B, 17) initial states
    mocap: Tensor  # (B, 1, 3) obstacle positions

    def repeat(self, n: int) -> PandaTask:
        rep = (t.repeat_interleave(n, 0) for t in (self.start, self.mocap))
        return PandaTask(self.world, self.sim, *rep)

    def step(self, state: Tensor, act: Tensor) -> Tensor:
        return self.world.step_in(self.sim, state, act, self.mocap)

    def distance(self, state: Tensor) -> Tensor:
        return (self.world.chain.site(state[..., :7]) - state[..., 14:17]).norm(dim=-1)
