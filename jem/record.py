"""Recordings of the arm as animations: the experts, the learned options, sums of
options, and the paths a planner executes around obstacles never seen in the demos.

Joint trajectories are simulated with MuJoCo Warp, as everywhere else, then replayed
through MuJoCo's own renderer (headless, EGL). The target is a red sphere and the hand
leaves a trail in its option's colour. Animations play in real time (20 Hz control):
animated WebP by default (true colour, small), or GIF.

    uv run jem record runs/<arm run> --out media/
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; set before importing mujoco

from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from jem.downstream import ARM_DEFAULTS, ARM_TASKS, DownstreamConfig, make_task, run  # noqa: E402
from jem.envs.panda import ASSET, obstacle_spec  # noqa: E402
from jem.evaluate import expert_rollout, rollout  # noqa: E402
from jem.figures import SERIES  # noqa: E402

TARGET = (0.9, 0.12, 0.12, 1.0)
NEW = 6  # SERIES colour of behaviour nobody demonstrated: sums of options, planned paths
INK, PAPER = (24, 24, 24), (250, 250, 250)
COMPOSITIONS = (("toward", "cw"), ("cw", "up"), ("toward", "up"), ("up", "down"))
# Camera (lookat offset from the target, distance, azimuth, elevation) per recording.
CAMERAS = {
    "free": ((-0.15, 0.0, 0.05), 1.85, 205.0, -40.0),
    "bin": ((0.0, 0.0, 0.0), 1.3, 120.0, -35.0),  # into the bin: up, over, down
    "pillar": ((0.0, 0.0, 0.0), 1.45, 135.0, -65.0),  # from above: around
}


def render_model(kind: str | None = None) -> mujoco.MjModel:
    """The arm (and the task's obstacle, see-through) plus render-only extras: a pale sky,
    a big buffer."""
    spec = mujoco.MjSpec.from_file(str(ASSET))
    if kind is not None:
        spec = obstacle_spec(kind, spec)
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.97, 0.97, 0.98],
        rgb2=[0.76, 0.8, 0.86],
        width=256,
        height=256,
    )
    for texture in spec.textures:  # a pale floor reads better (and compresses better)
        if texture.name == "grid":
            texture.rgb1, texture.rgb2 = [0.86, 0.87, 0.88], [0.8, 0.81, 0.83]
    for material in spec.materials:
        if material.name == "grid":
            material.reflectance = 0.05
    for light in spec.lights:  # a high, wide spot: even light, soft shadows
        light.pos = [0.3, 0.0, 4.0]
        light.cutoff, light.exponent = 60.0, 1.0
        light.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    model = spec.compile()
    if kind is not None:  # the path inside the bin, or behind the pillar, stays visible
        model.geom_rgba[model.geom_bodyid == model.body("obstacle").id, 3] = 0.35
    return model


def _add(scene: mujoco.MjvScene):
    if scene.ngeom >= scene.maxgeom:
        return None
    scene.ngeom += 1
    return scene.geoms[scene.ngeom - 1]


def _glow(g, emission: float) -> None:
    """Overlays are partly lit from within, so they keep their colour in any light."""
    g.emission, g.specular = emission, 0.0


def _sphere(scene, pos, radius: float, rgba) -> None:
    if (g := _add(scene)) is not None:
        size, mat = np.array([radius, 0.0, 0.0]), np.eye(3).ravel()
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size, np.asarray(pos), mat, rgba)
        _glow(g, 0.3)


def _segment(scene, a, b, width: float, rgba) -> None:
    if (g := _add(scene)) is not None:
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a), np.asarray(b))
        _glow(g, 0.25)


class Stage:
    """Renders arm poses with the target and the hand's trail from one camera."""

    def __init__(self, model: mujoco.MjModel, width: int, height: int, camera, target):
        offset, distance, azimuth, elevation = camera
        self.model, self.data = model, mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height, width, max_geom=4000)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = np.asarray(target) + np.asarray(offset)
        self.camera.distance = distance
        self.camera.azimuth, self.camera.elevation = azimuth, elevation

    def frame(self, q, target, trail, rgba, mocap=None) -> np.ndarray:
        self.data.qpos[: len(q)] = q
        if mocap is not None:
            self.data.mocap_pos[:] = mocap
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        _sphere(self.renderer.scene, target, 0.025, TARGET)
        for a, b in zip(trail[:-1], trail[1:], strict=True):
            _segment(self.renderer.scene, a, b, 0.007, rgba)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def _rgba(hex_colour: str) -> tuple[float, ...]:
    return tuple(int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)) + (1.0,)


def panel(model, obs: np.ndarray, rgba, camera, size, label, mocap=None) -> list[np.ndarray]:
    """Frames of one episode: obs (T+1, 16) arm observations (q, x, g, g - x)."""
    stage = Stage(model, *size, camera, obs[0, 10:13])
    frames = []
    for t in range(len(obs)):
        img = stage.frame(obs[t, :7], obs[t, 10:13], obs[: t + 1, 7:10], rgba, mocap)
        frames.append(_caption(img, label(t) if callable(label) else label))
    stage.close()
    return frames


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(13, img.shape[0] // 17))
    left, top, right, bottom = draw.textbbox((10, 7), text, font=font)
    draw.rounded_rectangle((left - 5, top - 4, right + 5, bottom + 4), radius=4, fill=PAPER)
    draw.text((10, 7), text, fill=INK, font=font)
    return np.asarray(im)


def tile(panels: list[list[np.ndarray]], cols: int) -> list[np.ndarray]:
    """Panels of frame lists -> one frame list; shorter panels hold their last frame."""
    length = max(len(p) for p in panels)
    blank = np.full_like(panels[0][0], 255)
    rows = []
    for t in range(length):
        cells = [p[min(t, len(p) - 1)] for p in panels]
        cells += [blank] * (-len(cells) % cols)
        grid = [np.concatenate(cells[i : i + cols], 1) for i in range(0, len(cells), cols)]
        rows.append(np.concatenate(grid, 0))
    return rows


def _swatches(width: int) -> np.ndarray:
    """A strip of every overlay colour, so a GIF palette keeps each of them."""
    base = [TARGET, (0.8, 0.4, 0.1, 1.0)] + [_rgba(c) for c in SERIES]
    colours = [np.array(c[:3]) * k for c in base for k in (0.55, 0.75, 1.0)]  # as lit
    cell = max(1, width // len(colours))
    cells = [np.full((24, cell, 3), c * 255) for c in colours]
    strip = np.concatenate(cells, 1).astype(np.uint8)
    return np.pad(strip, ((0, 0), (0, width - strip.shape[1]), (0, 0)), mode="edge")


def save_animation(frames: list[np.ndarray], path: Path, fps: float = 20.0) -> Path:
    """An animation in the format of `path`'s suffix: .webp (lossy, true colour) or .gif.

    A GIF gets one palette for all frames (no flicker), taken from the last frame, which
    shows every trail, plus swatches of the overlay colours."""
    images = [Image.fromarray(f) for f in frames]
    options = {"quality": 82, "method": 4}
    if path.suffix == ".gif":
        ref = np.concatenate([frames[-1], _swatches(frames[-1].shape[1])], 0)
        palette = Image.fromarray(ref).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        images = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
        options = {}
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=round(1000 / fps),
        **options,
    )
    return path


@torch.no_grad()
def record_options(
    lib, out: Path, steps: int = 80, seed: int = 0, size=(360, 300), fmt: str = "webp"
) -> list[Path]:
    """experts, options (both from one shared start) and compositions, as animations."""
    world = lib.world
    gen = torch.Generator(device=world.device).manual_seed(seed)
    start = world.reset(1, gen)
    model = render_model()
    names = list(world.expert_names)  # expert e's index, its name, its library option
    colour = {n: _rgba(SERIES[i]) for i, n in enumerate(names)}
    experts, options = [], []
    for e, name in enumerate(names):
        obs, _ = expert_rollout(world, e, start, steps, gen)
        experts.append(
            panel(
                model, obs[0].cpu().numpy(), colour[name], CAMERAS["free"], size, f"expert: {name}"
            )
        )
        obs = lib.rollout(lib[name], start, steps)
        options.append(
            panel(
                model,
                obs[0].cpu().numpy(),
                colour[name],
                CAMERAS["free"],
                size,
                f"learned option: {name}",
            )
        )
    comps = []
    for i, j in COMPOSITIONS:
        zs = torch.stack([lib[i], lib[j]])
        obs, _ = rollout(lib.model, world, zs, start, steps, lib.proposals, lib.gen)
        comps.append(
            panel(
                model,
                obs[0].cpu().numpy(),
                _rgba(SERIES[NEW]),
                CAMERAS["free"],
                size,
                f"z_{i} + z_{j}",
            )
        )
    out.mkdir(parents=True, exist_ok=True)
    return [
        save_animation(tile(experts, 3), out / f"experts.{fmt}"),
        save_animation(tile(options, 3), out / f"options.{fmt}"),
        save_animation(tile(comps, 2), out / f"compositions.{fmt}"),
    ]


@torch.no_grad()
def record_planning(
    lib,
    out: Path,
    episodes: int = 3,
    planner: str = "option-cem",
    seed: int = 0,
    size=(360, 300),
    fmt: str = "webp",
) -> Path:
    """plan-<planner>: the paths a planner executes, one row per task (bin, pillar)."""
    panels = []
    dt = lib.world.cfg.dt
    for kind in ("bin", "pillar"):
        cfg = DownstreamConfig(
            episodes=episodes,
            seed=seed,
            task=kind,
            horizon=40,
            **{**ARM_DEFAULTS, **ARM_TASKS[kind]},
        )
        gen = torch.Generator(device=lib.library.device).manual_seed(cfg.seed + 1)
        task = make_task(lib.world, kind, episodes, gen)
        res = run(lib, planner, cfg, task)
        traj = res["traj"].cpu().numpy()
        dist = np.linalg.norm(traj[..., 13:16], axis=-1)  # |g - x| at every step
        model = render_model(kind)
        for b in range(episodes):
            hit = np.flatnonzero(dist[b] < cfg.tolerance)
            reached = hit[0] if len(hit) else None

            def label(t, kind=kind, b=b, reached=reached):
                done = reached is not None and t >= reached
                state = f"reached in {reached * dt:.1f} s" if done else f"{t * dt:.1f} s"
                return f"{kind} {b + 1} ({planner}): {state}"

            mocap = task.mocap[b].cpu().numpy()
            panels.append(
                panel(model, traj[b], _rgba(SERIES[NEW]), CAMERAS[kind], size, label, mocap)
            )
    out.mkdir(parents=True, exist_ok=True)
    return save_animation(tile(panels, episodes), out / f"plan-{planner}.{fmt}")
