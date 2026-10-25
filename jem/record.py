"""Recordings of the arm as animations: the experts, the learned options, online
recognition of a demonstration that switches experts, sums of options, and the paths a
planner executes around obstacles never seen in the demos.

Joint trajectories are simulated with MuJoCo Warp, as everywhere else, then replayed
through MuJoCo's own renderer (headless, EGL; `jem/__init__.py` selects it before mujoco
is imported). The target is a red sphere and the hand leaves a trail in its option's
colour. Animations play in real time, one frame per control step: animated WebP by default
(true colour, small), or GIF.

    uv run jem record runs/<arm run> --out media/
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from jem.downstream import ARM_DEFAULTS, ARM_TASKS, DownstreamConfig, make_task, run
from jem.envs.panda import ASSET, obstacle_spec
from jem.evaluate import expert_rollout, hmm_filter, rollout, step_loglik
from jem.figures import SERIES
from jem.inference import Proposals

TARGET = (0.9, 0.12, 0.12, 1.0)
NEW = 6  # SERIES colour of behaviour nobody demonstrated: sums of options, planned paths
INK, PAPER = (24, 24, 24), (250, 250, 250)
MUTED, TRACK, SHADE = (96, 96, 96), (222, 222, 226), (232, 232, 236)
# The demonstration for online recognition switches expert every 1.5 s in this order: each
# expert's stretch of the path is its own (none retraces another), and the exact Bayes
# filter ranks the acting expert first from the second step after every switch.
TOUR = ("ccw", "down", "toward", "up", "away", "cw")
RECOGNITION_VIEW = (600, 420)  # the 3-D view; the belief panel and timeline fill 1080 wide
# Sums of options and their shapes: spirals in and out, a helix, a diagonal, two that cancel.
COMPOSITIONS = (
    ("toward", "cw"),
    ("away", "cw"),
    ("cw", "up"),
    ("toward", "up"),
    ("toward", "away"),
    ("up", "down"),
)
# Camera (lookat offset from the target, distance, azimuth, elevation) per recording.
CAMERAS = {
    "free": ((-0.15, 0.0, 0.05), 1.85, 205.0, -40.0),
    "bin": ((0.0, 0.0, 0.0), 1.3, 120.0, -35.0),  # into the bin: up, over, down
    "pillar": ((0.0, 0.0, 0.0), 1.45, 135.0, -65.0),  # from above: around
    # The TOUR's loop, which passes through the target, from the side where the hand hides
    # least of each stretch as it is drawn.
    "tour": ((-0.1, 0.0, 0.1), 1.6, 105.0, -25.0),
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
        """Pose q with the target and the trail (n, 3), in one colour rgba (4,) or in one
        colour per segment (n - 1, 4)."""
        self.data.qpos[: len(q)] = q
        if mocap is not None:
            self.data.mocap_pos[:] = mocap
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        _sphere(self.renderer.scene, target, 0.025, TARGET)
        colours = np.broadcast_to(np.asarray(rgba, dtype=float), (max(len(trail) - 1, 0), 4))
        for a, b, c in zip(trail[:-1], trail[1:], colours, strict=True):
            _segment(self.renderer.scene, a, b, 0.007, c)
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


def _caption(img: np.ndarray, text: str, size: int | None = None) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=size or max(13, img.shape[0] // 17))
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


def save_animation(frames: list[np.ndarray], path: Path, fps: float, hold: float = 0.0) -> Path:
    """An animation in the format of `path`'s suffix: .webp (lossy, true colour) or .gif.
    The last frame stays up `hold` seconds longer, so the end can be read before the loop.

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
        duration=[round(1000 / fps)] * (len(images) - 1) + [round(1000 * (1 / fps + hold))],
        **options,
    )
    return path


@torch.no_grad()
def record_options(
    lib, out: Path, steps: int = 80, seed: int = 0, size=(360, 300), fmt: str = "webp"
) -> list[Path]:
    """experts, options (both from one shared start) and compositions, as animations."""
    world = lib.world
    fps = 1 / world.cfg.dt  # one frame per control step: real time
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
        save_animation(tile(experts, 3), out / f"experts.{fmt}", fps),
        save_animation(tile(options, 3), out / f"options.{fmt}", fps),
        save_animation(tile(comps, 3), out / f"compositions.{fmt}", fps),
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
    return save_animation(tile(panels, episodes), out / f"plan-{planner}.{fmt}", 1 / dt)


@torch.no_grad()
def switching_demo(world, start: Tensor, experts: Tensor, gen) -> tuple[Tensor, Tensor]:
    """A demonstration from one start state (1, ds) in which experts[t] acts at step t:
    observations (T+1, do) and actions (T, da)."""
    state, obs_t, act_t = start, [], []
    for e in experts:
        obs = world.obs(state)
        act = world.expert_act(obs, e.reshape(1), gen)
        obs_t.append(obs[0])
        act_t.append(act[0])
        state = world.step(state, act)
    obs_t.append(world.obs(state)[0])
    return torch.stack(obs_t), torch.stack(act_t)


@torch.no_grad()
def online_beliefs(lib, obs: Tensor, act: Tensor, gen) -> tuple[Tensor, Tensor]:
    """Beliefs over the library options after each step, (T, K) in library order: JEM's,
    from the model's step likelihoods, and the Bayes filter's, from the true expert
    densities. Both use the run's tracking evaluation settings (switch hazard, proposals)."""
    world, ec = lib.world, lib.cfg.eval
    props = Proposals(
        world.act_dim, world.act_low, world.act_high, ec.context_proposals, obs.device, seed=11
    )
    ll = step_loglik(lib.model, obs[None, :-1], act[None], lib.library, props, gen)
    names = list(world.expert_names)
    k = len(names)
    experts = torch.arange(k, device=obs.device).expand(len(act), k)
    true_ll = world.expert_log_prob(obs[:-1, None, :], experts, act[:, None, :])
    order = [names.index(n) for n in lib.names]  # true experts in library order
    jem = hmm_filter(ll, ec.track_hazard)[0]
    return jem, hmm_filter(true_ll[None, :, order], ec.track_hazard)[0]


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))


def _span(draw, x0: int, x1: int, y0: int, y1: int, fill) -> None:
    """Fill pixels [x0, x1) x [y0, y1); nothing if empty (PIL's rectangles are inclusive)."""
    if x1 > x0 and y1 > y0:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)


def belief_panel(size, names, colours, jem, bayes, acting, header: str, footer: str):
    """One bar per option for JEM's belief, a tick at the Bayes filter's, and the acting
    expert's row shaded (acting=None: none)."""
    width, height = size
    im = Image.new("RGB", size, PAPER)
    draw = ImageDraw.Draw(im)
    big, small = ImageFont.load_default(size=19), ImageFont.load_default(size=15)
    draw.text((24, 18), header, fill=INK, font=big)
    x0, x1, top, row = 132, width - 84, 64, 44
    for r, name in enumerate(names):
        y = top + r * row
        if r == acting:
            draw.rounded_rectangle((12, y - 8, width - 12, y + 30), radius=6, fill=SHADE)
        _span(draw, 24, 38, y + 4, y + 18, colours[r])
        draw.text((46, y), name, fill=INK, font=big)
        _span(draw, x0, x1, y + 1, y + 21, TRACK)
        _span(draw, x0, x0 + round(float(jem[r]) * (x1 - x0)), y + 1, y + 21, colours[r])
        tick = x0 + round(float(bayes[r]) * (x1 - x0))
        _span(draw, tick - 1, tick + 2, y - 4, y + 26, INK)
        draw.text((x1 + 12, y + 2), f"{float(jem[r]):.2f}", fill=INK, font=small)
    y = top + len(names) * row + 2
    _span(draw, 24, 52, y + 5, y + 17, MUTED)
    draw.text((60, y), "JEM's belief", fill=MUTED, font=small)
    _span(draw, 186, 189, y + 1, y + 21, INK)
    draw.text((198, y), "Bayes filter, true densities", fill=MUTED, font=small)
    draw.rounded_rectangle((24, y + 28, 52, y + 46), radius=4, fill=SHADE)
    draw.text((60, y + 28), "the expert acting", fill=MUTED, font=small)
    draw.text((24, height - 32), footer, fill=INK, font=small)
    return np.asarray(im)


def belief_timeline(size, colours, acting, jem, bayes, t: int, dt: float):
    """The acting expert and both beliefs over the whole demonstration, filled in up to step
    t, with a cursor there and seconds below."""
    width = size[0]
    im = Image.new("RGB", size, PAPER)
    draw = ImageDraw.Draw(im)
    small = ImageFont.load_default(size=15)
    steps = len(acting)
    left, right = 100, width - 24
    rows = (("expert", 12, 22), ("JEM", 44, 50), ("Bayes", 102, 50))
    xs = [round(left + s / steps * (right - left)) for s in range(steps + 1)]
    for label, y, h in rows:
        draw.text((16, y + h // 2 - 9), label, fill=INK, font=small)
        _span(draw, left, right, y, y + h, SHADE)
    for s in range(min(t, steps)):
        _span(draw, xs[s], xs[s + 1], 12, 34, colours[acting[s]])
        for (_, y, h), belief in zip(rows[1:], (jem, bayes), strict=True):
            edges = np.round(y + h - h * np.concatenate([[0.0], np.cumsum(belief[s])])).astype(int)
            for r in range(len(colours)):
                _span(draw, xs[s], xs[s + 1], edges[r + 1], edges[r], colours[r])
    axis = rows[-1][1] + rows[-1][2] + 2
    for sec in range(int(steps * dt + 1e-9) + 1):
        x = round(left + sec / (steps * dt) * (right - left))
        _span(draw, x, x + 1, axis, axis + 6, INK)
        draw.text((x - 9, axis + 6), f"{sec} s", fill=MUTED, font=small)
    if t > 0:
        _span(draw, xs[min(t, steps)] - 1, xs[min(t, steps)] + 1, 6, axis, INK)
    return np.asarray(im)


@torch.no_grad()
def record_recognition(
    lib, out: Path, seconds: float = 1.5, seed: int = 0, fmt: str = "webp"
) -> Path:
    """recognition: a demonstration from the recordings' start that switches expert every
    `seconds` (TOUR), beside JEM's online belief over the options after each step and the
    Bayes filter's, and a timeline of both under the acting expert."""
    world = lib.world
    dt = world.cfg.dt
    names = list(world.expert_names)
    gen = torch.Generator(device=world.device).manual_seed(seed)
    start = world.reset(1, gen)
    steps = max(1, round(seconds / dt))
    experts = torch.tensor(
        [names.index(n) for n in TOUR for _ in range(steps)], device=start.device
    )
    obs, act = switching_demo(world, start, experts, gen)
    jem, bayes = (b.cpu().numpy() for b in online_beliefs(lib, obs, act, gen))
    obs = obs.cpu().numpy()
    acting = [lib.names.index(names[e]) for e in experts.tolist()]  # library index per step
    colours = [_rgb(SERIES[names.index(n)]) for n in lib.names]
    trail = np.array([_rgba(SERIES[e]) for e in experts.tolist()])
    right_jem = np.cumsum(jem.argmax(-1) == acting)
    right_bayes = np.cumsum(bayes.argmax(-1) == acting)
    uniform = np.full(len(colours), 1 / len(colours))
    width, height = RECOGNITION_VIEW
    stage = Stage(render_model(), width, height, CAMERAS["tour"], obs[0, 10:13])
    frames = []
    for t in range(len(obs)):
        img = stage.frame(obs[t, :7], obs[t, 10:13], obs[: t + 1, 7:10], trail[:t])
        view = _caption(img, f"a demonstration that switches expert every {steps * dt:g} s", 19)
        shown = (jem[t - 1], bayes[t - 1]) if t else (uniform, uniform)  # after t steps
        footer = (
            f"top option right so far: JEM {right_jem[t - 1]} of {t} steps, "
            f"Bayes {right_bayes[t - 1]}"
            if t
            else "no step seen yet"
        )
        bars = belief_panel(
            (1080 - width, height),
            lib.names,
            colours,
            *shown,
            acting[t - 1] if t else None,
            f"online belief after {t * dt:.2f} s",
            footer,
        )
        line = belief_timeline((1080, 180), colours, acting, jem, bayes, t, dt)
        frames.append(np.concatenate([np.concatenate([view, bars], 1), line], 0))
    stage.close()
    out.mkdir(parents=True, exist_ok=True)
    return save_animation(frames, out / f"recognition.{fmt}", 1 / dt, hold=2.5)
