from jem.config import EnvConfig
from jem.envs.particle import ParticleWorld

__all__ = ["ParticleWorld", "make_world"]


def make_world(cfg: EnvConfig, device="cpu") -> ParticleWorld:
    if cfg.name != "particle":
        raise KeyError(f"unknown env {cfg.name!r}")
    return ParticleWorld(cfg, device)
