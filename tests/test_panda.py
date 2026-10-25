import pytest
import torch

pytest.importorskip("mujoco_warp")
if not torch.cuda.is_available():
    pytest.skip("MuJoCo Warp needs a CUDA device", allow_module_level=True)

from jem.config import make_config  # noqa: E402
from jem.envs import make_world  # noqa: E402

DEV = torch.device("cuda")


@pytest.fixture(scope="module")
def world():
    return make_world(make_config("panda").env, DEV)


def test_reset_obs_step_shapes(world):
    gen = torch.Generator(device=DEV).manual_seed(0)
    state = world.reset(32, gen)
    assert state.shape == (32, world.state_dim)
    obs = world.obs(state)
    assert obs.shape == (32, world.obs_dim)
    assert torch.allclose(obs[:, 13:], obs[:, 10:13] - obs[:, 7:10])
    act = world.expert_act(obs, torch.zeros(32, dtype=torch.long, device=DEV), gen)
    assert act.shape == (32, 7) and act.abs().max() <= 1
    nxt = world.step(state, act)
    assert nxt.shape == state.shape and torch.equal(nxt[:, 14:], state[:, 14:])
    assert (nxt[:, :7] - state[:, :7]).abs().max() > 0


def test_experts_move_the_end_effector_where_they_say(world):
    """Over a few steps each expert's realised displacement points along its velocity."""
    gen = torch.Generator(device=DEV).manual_seed(1)
    state = world.reset(256, gen)
    for k, name in enumerate(world.expert_names):
        s = state
        ex = torch.full((256,), k, device=DEV)
        v = world.expert_velocities(world.obs(s)[:, 7:10], s[:, 14:])[:, k]
        x0 = world.obs(s)[:, 7:10]
        for _ in range(4):
            s = world.step(s, world.expert_act(world.obs(s), ex, gen))
        moved = world.obs(s)[:, 7:10] - x0
        cos = torch.nn.functional.cosine_similarity(moved, v, dim=-1)
        assert cos.median() > 0.8, name


def test_log_prob_broadcasts_and_products_are_exact(world):
    gen = torch.Generator(device=DEV).manual_seed(2)
    obs = world.obs(world.reset(8, gen))
    k = world.num_experts
    props = torch.rand(64, 7, generator=gen, device=DEV) * 2 - 1
    experts = torch.arange(k, device=DEV).expand(8, k)[..., None]
    lp = world.expert_log_prob(obs[:, None, None, :], experts, props[None, None])
    assert lp.shape == (8, k, 64) and lp.isfinite().all()
    # The product of experts 0 and 2 (weights 1 and 2), up to a constant over actions.
    prod = world.product_log_prob(obs[:, None, :], (0, 2), (1.0, 2.0), props[None])
    gap = lp[:, 0] + 2 * lp[:, 2] - prod
    assert (gap - gap.mean(-1, keepdim=True)).abs().max() < 1e-3
    # A zero weight drops an expert entirely.
    single = world.product_log_prob(obs[:, None, :], (1, 3), (1.0, 0.0), props[None])
    assert torch.allclose(single, lp[:, 1], atol=1e-4)


def test_expert_pairs(world):
    adjacent, antipodal = world.expert_pairs()
    names = world.expert_names
    assert {(names[i], names[j]) for i, j in antipodal} == {
        ("toward", "away"),
        ("cw", "ccw"),
        ("up", "down"),
    }
    assert len(adjacent) == 12


@pytest.mark.parametrize("kind", ["bin", "pillar"])
def test_obstacles_block_the_straight_path(world, kind):
    gen = torch.Generator(device=DEV).manual_seed(3)
    task = world.make_task(kind, 16, gen)
    start_x = world.chain.site(task.start[:, :7])
    assert task.distance(task.start).min() > 0.25
    state = task.start
    toward = torch.full((16,), world.expert_names.index("toward"), device=DEV)
    down = torch.full((16,), world.expert_names.index("down"), device=DEV)
    for _ in range(60):  # toward + down heads straight for a bin target; toward for the pillar
        obs = world.obs(state)
        if kind == "bin":
            act = world.product_act(obs, torch.stack([toward, down], -1), torch.ones(16, 2), gen)
        else:
            act = world.expert_act(obs, toward, gen)
        state = task.step(state, act)
    assert (task.distance(state) > 0.06).float().mean() > 0.8
    assert (world.chain.site(state[:, :7]) - start_x).norm(dim=-1).mean() > 0.02
