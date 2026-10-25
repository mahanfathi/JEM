"""Typed experiment configuration: nested dataclasses, named presets, dotted overrides.

cfg = make_config("jem", ["model.option_dim=2", "train.steps=5000"])
"""

from __future__ import annotations

import dataclasses as dc
import json
import typing
import warnings
from typing import Any


@dc.dataclass
class PandaConfig:
    """The arm's experts (jem/envs/panda.py). Speeds in m/s, angles in rad."""

    ee_speed: float = 0.25  # experts' end-effector speed
    act_std: float = 0.35  # expert noise per joint, in action units (the box is [-1, 1]^7)
    qd_scale: float = 0.2  # |a_j| = 1 commands qd_scale * joint j's velocity limit
    damping: float = 0.05  # damped least squares lambda (m)
    posture_gain: float = 1.0  # null-space pull toward the home posture (1/s)
    axis_radius: float = 0.05  # radial and orbit speeds fade within this distance of the axis
    init_noise: float = 0.3  # uniform joint noise around the home posture at reset
    # Targets ~ U[target_lo, target_hi]: reachable within home +- 0.3 rad on every joint.
    target_lo: tuple[float, ...] = (0.45, -0.2, 0.35)
    target_hi: tuple[float, ...] = (0.6, 0.2, 0.7)


@dc.dataclass
class EnvConfig:
    name: str = "particle"
    experts: tuple[str, ...] = ("toward", "away", "cw", "ccw")
    dt: float = 0.05
    angle_std: float = 0.5  # expert heading noise (rad); makes one transition ambiguous
    speed_min: float = 0.5
    speed_max: float = 1.0
    arena: float = 1.0  # agent confined to [-arena, arena]^2
    target_range: float = 0.5  # target ~ U[-target_range, target_range]^2
    min_dist: float = 0.15  # initial agent-target separation
    switch_prob: float = 0.0  # per-step chance that the demonstrator switches expert
    panda: PandaConfig = dc.field(default_factory=PandaConfig)


@dc.dataclass
class DataConfig:
    num_traj: int = 4096
    val_traj: int = 512
    traj_len: int = 64
    context_len: int = 8  # steps the option is inferred from
    target_len: int = 8  # steps the option must then explain
    target_gap: int = 0  # targets start a uniform 0..target_gap steps after the context
    seed: int = 1234  # dataset seed, fixed across model seeds


@dc.dataclass
class ModelConfig:
    kind: str = "linear"  # "linear": E = -tau <z, phi(s,a)>   "mlp": E = f(s, z, a) (2022 form)
    arch: str = "factored"  # linear only: factored phi = W(s) psi(a) | joint phi = MLP(s, a)
    action_features: int = 64  # factored only: rank p of the state/action factorisation
    option_dim: int = 8
    hidden: int = 256
    depth: int = 3
    feature_norm: str = "sphere"  # linear only: sphere | ball | tanh | none
    prototypes: int = 0  # linear only: learned prototype options trained as a mixture (0: off)
    init_scale: float = 5.0  # linear only: initial temperature tau
    learn_scale: bool = True
    squared: bool = False  # mlp only: E = f^2, as in 2022


@dc.dataclass
class InferConfig:
    method: str = "newton"  # newton (linear only) | langevin | gd
    prior_std: float = 1.0  # z ~ N(0, prior_std^2 I)
    iters: int = 8  # newton iterations (fp32 floor by ~6) or langevin/gd steps
    implicit_grad: bool = True  # newton: exact IFT gradient through z*; False = EM-style
    step_size: float = 5e-3  # langevin/gd: z <- z - step/2 dE/dz (+ sqrt(step) noise)
    grad_clip: float = 10.0  # langevin/gd: per-coordinate clip on dE/dz, 0 disables
    init_std: float = 1.0  # langevin/gd: z_0 ~ N(0, init_std^2 I)


@dc.dataclass
class ObjectiveConfig:
    name: str = "jem"  # jem | legacy
    num_proposals: int = 256  # action proposals per step
    proposal: str = "uniform"  # uniform (randomised Sobol) | prior (learned behaviour prior)
    uniform_frac: float = 0.25  # prior: share of the proposals that stay uniform (defensive)
    prior_components: int = 16  # prior: truncated-Gaussian mixture components
    prior_hidden: int = 256
    prior_steps: int = 2000  # prior: maximum-likelihood steps before JEM; then it is frozen
    prototype_start: int = 2000  # prototypes: k-means of inferred options initialises them here
    prototype_weight: float = 1.0  # prototypes: weight of the mixture held-out loss
    continuous_weight: float = 1.0  # prototypes: weight of the per-window loss once they start
    symmetric: bool = True  # jem: also infer from targets and predict the context
    hazard: float = 0.0  # jem: assumed per-step option-switch probability (0 = options persist)
    # legacy (2022, Mordatch-style) terms
    action_chains: int = 64
    action_steps: int = 10
    action_step_size: float = 5e-3
    action_grad_clip: float = 10.0
    kl_coef: float = 1.0
    klz_coef: float = 1.0
    klz_chains: int = 64
    hinge_eps: float = 0.1


@dc.dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 256
    lr: float = 1e-3
    warmup: int = 500
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    amp: bool = False  # bf16 autocast for the feature network
    log_every: int = 50
    eval_every: int = 2_000
    seed: int = 0
    device: str = "auto"


@dc.dataclass
class EvalConfig:
    num_proposals: int = 4096  # partition functions in policy KL and composition metrics
    context_proposals: int = 1024  # option inference, recognition and tracking
    windows: int = 2048
    context_lens: tuple[int, ...] = (1, 2, 4, 8, 16)
    rollouts: int = 16
    rollout_len: int = 80
    track_traj: int = 256  # switching trajectories for online option tracking (0 disables)
    track_hazard: float = 1 / 16  # per-step switch probability in the tracking test
    algebra_weights: tuple[float, ...] = (0.25, 0.5, 2.0, 4.0)
    figures: bool = True


@dc.dataclass
class Config:
    name: str = "jem"
    env: EnvConfig = dc.field(default_factory=EnvConfig)
    data: DataConfig = dc.field(default_factory=DataConfig)
    model: ModelConfig = dc.field(default_factory=ModelConfig)
    infer: InferConfig = dc.field(default_factory=InferConfig)
    objective: ObjectiveConfig = dc.field(default_factory=ObjectiveConfig)
    train: TrainConfig = dc.field(default_factory=TrainConfig)
    eval: EvalConfig = dc.field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dc.asdict(self)


# The 2022 implementation, ported faithfully (hyperparameters from config/defaults.py there).
_LEGACY = {
    "model.kind": "mlp",
    "model.option_dim": 1,
    "model.hidden": 64,
    "model.depth": 3,
    "model.squared": True,
    "infer.method": "langevin",
    "infer.iters": 10,
    "infer.step_size": 5e-3,
    "infer.grad_clip": 10.0,
    "objective.name": "legacy",
    "data.context_len": 1,
    "data.target_len": 1,
    "train.lr": 1e-5,
    "train.warmup": 0,
}

PRESETS: dict[str, dict[str, Any]] = {
    "jem": {},
    "jem-em": {"infer.implicit_grad": False},
    "jem-d2": {"model.option_dim": 2},
    "jem-unnormed": {"model.feature_norm": "none"},
    "legacy": _LEGACY,
    # Same algorithm, given every chance: larger chains/steps, wider z, sane lr.
    "legacy-tuned": {
        **_LEGACY,
        "model.option_dim": 8,
        "model.hidden": 256,
        "infer.iters": 30,
        "infer.step_size": 0.1,
        "objective.action_steps": 30,
        "objective.action_step_size": 0.1,
        "data.context_len": 8,
        "data.target_len": 8,
        "objective.action_chains": 16,
        "objective.klz_chains": 8,
        "train.lr": 1e-4,
    },
    # Franka Panda in MuJoCo Warp: six task-space experts, 7-D joint velocity commands.
    # Candidates come from a learned behaviour prior (uniform ones are too sparse in 7-D),
    # targets lie up to 2 s past the context, and after 2k steps the options consolidate
    # into six prototypes trained as a mixture (per-window options absorb nuisance here).
    "panda": {
        "env.name": "panda",
        "env.experts": ("toward", "away", "cw", "ccw", "up", "down"),
        "env.min_dist": 0.1,
        "objective.proposal": "prior",
        "data.target_gap": 40,
        "model.prototypes": 6,
        "objective.continuous_weight": 0.0,
        "train.steps": 10_000,
        "eval.num_proposals": 16384,
        "eval.context_proposals": 4096,
    },
    "panda-uniform": {  # the arm with the particle's recipe (E6a): uniform proposals only
        "env.name": "panda",
        "env.experts": ("toward", "away", "cw", "ccw", "up", "down"),
        "env.min_dist": 0.1,
        "train.steps": 10_000,
        "eval.num_proposals": 16384,
        "eval.context_proposals": 4096,
    },
    "panda-smoke": {},  # filled in below: the smoke settings on the arm
    "smoke": {
        "data.num_traj": 64,
        "data.val_traj": 32,
        "data.traj_len": 24,
        "model.hidden": 32,
        "model.depth": 2,
        "objective.num_proposals": 32,
        "train.steps": 6,
        "train.batch_size": 16,
        "train.warmup": 0,
        "train.log_every": 2,
        "train.eval_every": 5,
        "eval.num_proposals": 256,
        "eval.windows": 64,
        "eval.context_lens": (1, 4),
        "eval.rollouts": 2,
        "eval.rollout_len": 10,
        "eval.track_traj": 4,
    },
}


PRESETS["panda-smoke"] = {
    **PRESETS["panda"],
    **PRESETS["smoke"],
    "objective.prototype_start": 2,
    "objective.prior_steps": 20,
    "eval.num_proposals": 1024,
    "eval.context_proposals": 256,
}


def _coerce(value: str, typ: Any) -> Any:
    """Parse a CLI string into the annotated field type."""
    origin = typing.get_origin(typ)
    if typ is bool:
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a bool: {value!r}")
    if origin is tuple:
        (item_t, *_) = typing.get_args(typ)
        items = json.loads(value) if value.startswith("[") else value.split(",")
        return tuple(_coerce(str(v), item_t) for v in items if str(v) != "")
    if typ in (int, float):
        return typ(float(value)) if typ is int else float(value)
    return value


def _set(cfg: Config, key: str, value: Any) -> None:
    *path, leaf = key.split(".")
    node: Any = cfg
    for part in path:
        node = getattr(node, part)
    fields = {f.name: f for f in dc.fields(node)}
    if leaf not in fields:
        raise KeyError(f"unknown config key {key!r}")
    hints = typing.get_type_hints(type(node))
    if isinstance(value, str) and hints[leaf] is not str:
        value = _coerce(value, hints[leaf])
    setattr(node, leaf, value)


def make_config(preset: str = "jem", overrides: list[str] | None = None) -> Config:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    cfg = Config(name=preset)
    for key, value in PRESETS[preset].items():
        _set(cfg, key, value)
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        if key == "name":
            cfg.name = value
        else:
            _set(cfg, key.strip(), value.strip())
    return cfg


def _leaves(prefix: str, values: dict[str, Any]):
    for key, value in values.items():
        if isinstance(value, dict):
            yield from _leaves(f"{prefix}.{key}", value)
        else:
            yield f"{prefix}.{key}", tuple(value) if isinstance(value, list) else value


def from_dict(d: dict[str, Any]) -> Config:
    """Rebuild a Config from `Config.to_dict()` output (e.g. a run's config.json).

    Keys of options that have since been removed are skipped with a warning, so old runs
    stay loadable.
    """
    cfg = Config(name=d.get("name", "jem"))
    for section, values in d.items():
        if section == "name" or not isinstance(values, dict):
            continue
        for key, value in _leaves(section, values):
            try:
                _set(cfg, key, value)
            except (KeyError, AttributeError):
                warnings.warn(f"ignoring unknown config key {key!r}", stacklevel=2)
    return cfg
