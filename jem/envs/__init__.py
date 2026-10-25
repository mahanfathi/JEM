from jem.config import EnvConfig
from jem.envs.particle import ParticleWorld

__all__ = ["ParticleWorld", "make_world"]


def make_world(cfg: EnvConfig, device="cpu"):
    if cfg.name == "particle":
        return ParticleWorld(cfg, device)
    if cfg.name == "panda":  # needs the `sim` extra (MuJoCo Warp)
        from jem.envs.panda import PandaWorld

        return PandaWorld(cfg, device)
    raise KeyError(f"unknown env {cfg.name!r}")
