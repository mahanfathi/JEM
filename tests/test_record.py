"""Arm recordings: the render model, frames with overlays, tiling, animation files."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", exc_type=ImportError)
pytest.importorskip("mujoco_warp", exc_type=ImportError)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from jem import record  # noqa: E402
from jem.config import make_config  # noqa: E402
from jem.energy import build_model  # noqa: E402
from jem.envs import make_world  # noqa: E402
from jem.evaluate import expert_rollout, hmm_filter, step_loglik  # noqa: E402
from jem.figures import SERIES  # noqa: E402
from jem.inference import Proposals  # noqa: E402
from jem.options import OptionLibrary  # noqa: E402


@pytest.fixture(scope="module")
def stage():
    model = record.render_model()
    try:
        stage = record.Stage(model, 160, 120, record.CAMERAS["free"], np.array([0.5, 0.0, 0.3]))
    except Exception as err:  # no headless OpenGL (EGL) here
        pytest.skip(f"cannot render offscreen: {err}")
    yield stage
    stage.close()


def test_render_model_adds_a_see_through_obstacle():
    arm = record.render_model()
    for kind, extra in (("bin", 5), ("pillar", 1)):  # pedestal and four walls; a cylinder
        model = record.render_model(kind)
        assert model.ngeom == arm.ngeom + extra
        mine = model.geom_bodyid == model.body("obstacle").id
        assert mine.sum() == extra
        assert np.allclose(model.geom_rgba[mine, 3], 0.35)
        assert np.allclose(model.geom_rgba[~mine], arm.geom_rgba)


def test_frame_draws_the_target_and_the_trail(stage):
    q = stage.model.key("home").qpos[:7]
    target = np.array([0.5, 0.0, 0.3])
    trail = np.linspace([0.3, -0.25, 0.45], [0.3, 0.25, 0.45], 12)
    img = stage.frame(q, target, trail, record._rgba(SERIES[0]))
    assert img.shape == (120, 160, 3) and img.dtype == np.uint8
    r, g, b = (img[..., i].astype(int) for i in range(3))
    assert ((r > 150) & (g < 90) & (b < 90)).sum() > 5  # the red target
    assert ((b > 120) & (b - r > 60)).sum() > 5  # the blue trail
    bare = stage.frame(q, target, trail[:1], record._rgba(SERIES[0]))
    assert not ((bare[..., 2].astype(int) - bare[..., 0] > 60) & (bare[..., 2] > 120)).any()


def test_frame_colours_each_segment(stage):
    q = stage.model.key("home").qpos[:7]
    trail = np.linspace([0.3, -0.25, 0.45], [0.3, 0.25, 0.45], 12)
    per = np.array([record._rgba(SERIES[0])] * 5 + [record._rgba(SERIES[1])] * 6)
    stage.frame(q, np.array([0.5, 0.0, 0.3]), trail, per)
    scene = stage.renderer.scene  # the segments are the last geoms added, in order
    drawn = np.array([scene.geoms[i].rgba for i in range(scene.ngeom - len(per), scene.ngeom)])
    assert np.allclose(drawn, per)
    with pytest.raises(ValueError):  # one colour per point is one too many
        stage.frame(q, np.array([0.5, 0.0, 0.3]), trail, np.vstack([per, per[:1]]))


def test_tile_pads_short_panels_and_empty_cells():
    a = [np.full((2, 3, 3), v, np.uint8) for v in (10, 20)]
    b = [np.full((2, 3, 3), v, np.uint8) for v in (30, 40, 50)]
    frames = record.tile([a, b], cols=3)
    assert len(frames) == 3 and frames[0].shape == (2, 9, 3)
    assert (frames[2][:, :3] == 20).all()  # a holds its last frame
    assert (frames[2][:, 3:6] == 50).all() and (frames[2][:, 6:] == 255).all()
    rows = record.tile([a, b, a], cols=2)
    assert rows[0].shape == (4, 6, 3) and (rows[0][2:, 3:] == 255).all()


def test_caption_draws_a_box_and_keeps_the_frame():
    img = np.zeros((120, 160, 3), np.uint8)
    out = record._caption(img, "expert: toward")  # needs Pillow >= 10.1 (sized default font)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[:30, :120].max() == max(record.PAPER)  # the light box behind the text
    assert out[60:].max() == 0  # nothing below it


@pytest.mark.parametrize("suffix", [".webp", ".gif"])
def test_save_animation_writes_every_frame(tmp_path, suffix):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(5)]
    path = record.save_animation(frames, tmp_path / f"clip{suffix}", fps=20)
    with Image.open(path) as im:
        assert im.format == suffix[1:].upper() and im.n_frames == 5
        assert im.size == (32, 24)


def test_save_animation_holds_the_last_frame(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(3)]
    path = record.save_animation(frames, tmp_path / "clip.gif", fps=20, hold=1.0)
    with Image.open(path) as im:  # a GIF reports each frame's duration
        durations = []
        for k in range(im.n_frames):
            im.seek(k)
            durations.append(im.info["duration"])
    assert durations == [50, 50, 1050]


def _particle_library(names=None) -> OptionLibrary:
    """An untrained library on the particle (CPU), optionally with its names reordered."""
    cfg = make_config("smoke", ["train.device=cpu"])
    world = make_world(cfg.env, torch.device("cpu"))
    model = build_model(cfg.model, world.obs_dim, world.act_dim)
    names = list(world.expert_names) if names is None else names
    library = torch.randn(
        len(names), cfg.model.option_dim, generator=torch.Generator().manual_seed(0)
    )
    return OptionLibrary(model, world, cfg, library, names)


def test_switching_demo_follows_the_schedule():
    world = _particle_library().world
    start = world.reset(1, torch.Generator().manual_seed(0))
    experts = torch.tensor([0] * 5 + [2] * 5)
    obs, act = record.switching_demo(world, start, experts, torch.Generator().manual_seed(1))
    assert obs.shape == (11, world.obs_dim) and act.shape == (10, world.act_dim)
    gen = torch.Generator().manual_seed(1)  # the same draws, one expert after the other
    ref_obs, ref_act = expert_rollout(world, 0, start, 5, gen)
    ref2_obs, ref2_act = expert_rollout(world, 2, ref_obs[:, -1], 5, gen)  # obs is the state
    assert torch.equal(obs[:6], ref_obs[0]) and torch.equal(act[:5], ref_act[0])
    assert torch.equal(obs[5:], ref2_obs[0]) and torch.equal(act[5:], ref2_act[0])


def test_online_beliefs_come_in_library_order():
    base = _particle_library()
    world, ec = base.world, base.cfg.eval
    names = ["away", "cw", "ccw", "toward"]  # a 4-cycle of the world's order: not self-inverse
    perm = [list(world.expert_names).index(n) for n in names]
    lib = OptionLibrary(base.model, world, base.cfg, base.library[perm], names)  # relabelled
    for acting in ("cw", "toward"):
        gen = torch.Generator().manual_seed(0)
        experts = torch.full((24,), list(world.expert_names).index(acting))
        obs, act = record.switching_demo(world, world.reset(1, gen), experts, gen)
        state = gen.get_state()
        jem, bayes = record.online_beliefs(lib, obs, act, gen)
        jem_w, bayes_w = record.online_beliefs(base, obs, act, torch.Generator().set_state(state))
        assert jem.shape == bayes.shape == (24, 4)
        assert bayes[-1].argmax() == names.index(acting)  # the true densities find the actor
        assert torch.allclose(jem, jem_w[:, perm], atol=1e-6)
        assert torch.allclose(bayes, bayes_w[:, perm], atol=1e-6)
        props = Proposals(
            world.act_dim, world.act_low, world.act_high, ec.context_proposals, "cpu", seed=11
        )
        gen = torch.Generator().set_state(state)
        ll = step_loglik(lib.model, obs[None, :-1], act[None], lib.library, props, gen)
        assert torch.allclose(jem, hmm_filter(ll, ec.track_hazard)[0], atol=1e-6)  # as tracked


def _count(img, colour) -> int:
    return int(np.all(img == np.array(colour), axis=-1).sum())


def _runs(line, colours) -> list[int]:
    """Index of each run of a test colour along a scanline; other pixels are skipped."""
    runs = []
    for pixel in line:
        hit = [i for i, c in enumerate(colours) if np.array_equal(pixel, c)]
        if hit and (not runs or runs[-1] != hit[0]):
            runs.append(hit[0])
    return runs


def test_belief_panel_draws_bars_ticks_and_the_acting_row():
    colours = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    jem, bayes = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    args = ((480, 420), ["a", "b", "c"], colours, jem, bayes)
    img = record.belief_panel(*args, 1, "head", "foot")
    plain = record.belief_panel(*args, None, "head", "foot")
    assert img.shape == (420, 480, 3)
    assert _count(img, colours[0]) == _count(img, colours[2]) == 14 * 14  # swatches only
    assert _count(img, colours[1]) > 20 * 14 * 14  # swatch and a full bar
    rows = {i: np.flatnonzero(np.all(img == c, -1).any(1)) for i, c in enumerate(colours)}
    added = np.flatnonzero(
        (np.all(img == record.SHADE, -1) & ~np.all(plain == record.SHADE, -1)).any(1)
    )
    assert added.min() <= rows[1].min() and added.max() >= rows[1].max()  # row b is shaded
    assert not set(added) & (set(rows[0]) | set(rows[2]))  # and only row b
    mid = rows[1][len(rows[1]) // 2]
    bar_end = np.flatnonzero(np.all(img[mid] == colours[1], -1)).max()
    ink_a = np.flatnonzero(np.all(img[rows[0]] == record.INK, -1).all(0))
    assert ink_a.size and abs(ink_a.mean() - bar_end) <= 3  # a's Bayes tick at 1.0: a full bar


def test_belief_timeline_fills_rows_in_order_up_to_the_cursor():
    colours = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    acting = [0, 0, 1, 2]
    jem, bayes = np.eye(3)[[1, 0, 1, 2]], np.eye(3)[acting]  # JEM wrong at step 0

    def draw(t):
        return record.belief_timeline((400, 180), colours, acting, jem, bayes, t, 0.5)

    empty, half, full = draw(0), draw(2), draw(4)
    assert full.shape == (180, 400, 3)
    assert all(_count(empty, c) == 0 for c in colours)
    shade = np.flatnonzero(np.all(empty == record.SHADE, -1).sum(1) > 100)
    bands = np.split(shade, np.flatnonzero(np.diff(shade) > 1) + 1)  # expert, JEM, Bayes
    mids = [band[len(band) // 2] for band in bands]
    assert [_runs(full[y], colours) for y in mids] == [[0, 1, 2], [1, 0, 1, 2], [0, 1, 2]]
    assert [_runs(half[y], colours) for y in mids] == [[0], [1, 0], [0]]
    last = np.flatnonzero(np.all(half[mids[0]] == colours[0], -1)).max()
    ink = np.flatnonzero(np.all(half[bands[0][0] : bands[2][-1] + 1] == record.INK, -1).all(0))
    assert ink.size and ink.min() == last + 1  # the cursor, right after the last filled step


def test_recognition_frame_t_shows_the_belief_after_t_steps(tmp_path, monkeypatch):
    lib = _particle_library(["away", "cw", "ccw", "toward"])
    tour = ("toward", "cw", "away")
    monkeypatch.setattr(record, "TOUR", tour)
    monkeypatch.setattr(record, "render_model", lambda kind=None: None)

    class FakeStage:  # no OpenGL
        def __init__(self, model, width, height, camera, target):
            self.shape = (height, width, 3)

        def frame(self, q, target, trail, rgba, mocap=None):
            assert len(rgba) == len(trail) - 1  # one colour per segment
            return np.zeros(self.shape, np.uint8)

        def close(self):
            pass

    beliefs, shown = [], []
    online, panel = record.online_beliefs, record.belief_panel

    def spy_online(*args):
        beliefs.append(online(*args))
        return beliefs[-1]

    def spy_panel(size, names, colours, jem, bayes, acting, header, footer):
        shown.append((np.asarray(jem), np.asarray(bayes), acting))
        return panel(size, names, colours, jem, bayes, acting, header, footer)

    monkeypatch.setattr(record, "Stage", FakeStage)
    monkeypatch.setattr(record, "online_beliefs", spy_online)
    monkeypatch.setattr(record, "belief_panel", spy_panel)
    path = record.record_recognition(lib, tmp_path, seconds=0.3)
    jem, bayes = (b.cpu().numpy() for b in beliefs[0])
    per = round(0.3 / lib.world.cfg.dt)
    acting = [lib.names.index(n) for n in tour for _ in range(per)]
    with Image.open(path) as im:
        assert im.n_frames == len(shown) == len(acting) + 1
    assert shown[0][2] is None and np.allclose(shown[0][0], 1 / len(lib.names))
    for t in range(1, len(acting) + 1):
        assert np.array_equal(shown[t][0], jem[t - 1]) and np.array_equal(shown[t][1], bayes[t - 1])
        assert shown[t][2] == acting[t - 1]
