"""Arm recordings: the render model, frames with overlays, tiling, animation files."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", exc_type=ImportError)
pytest.importorskip("mujoco_warp", exc_type=ImportError)

from PIL import Image  # noqa: E402

from jem import record  # noqa: E402
from jem.figures import SERIES  # noqa: E402


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


def test_tile_pads_short_panels_and_empty_cells():
    a = [np.full((2, 3, 3), v, np.uint8) for v in (10, 20)]
    b = [np.full((2, 3, 3), v, np.uint8) for v in (30, 40, 50)]
    frames = record.tile([a, b], cols=3)
    assert len(frames) == 3 and frames[0].shape == (2, 9, 3)
    assert (frames[2][:, :3] == 20).all()  # a holds its last frame
    assert (frames[2][:, 3:6] == 50).all() and (frames[2][:, 6:] == 255).all()
    rows = record.tile([a, b, a], cols=2)
    assert rows[0].shape == (4, 6, 3) and (rows[0][2:, 3:] == 255).all()


@pytest.mark.parametrize("suffix", [".webp", ".gif"])
def test_save_animation_writes_every_frame(tmp_path, suffix):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(5)]
    path = record.save_animation(frames, tmp_path / f"clip{suffix}", fps=20)
    with Image.open(path) as im:
        assert im.format == suffix[1:].upper() and im.n_frames == 5
        assert im.size == (32, 24)
