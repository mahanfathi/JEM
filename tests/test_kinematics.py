from pathlib import Path

import numpy as np
import pytest
import torch

from jem.envs.kinematics import SerialChain

mujoco = pytest.importorskip("mujoco")

XML = Path(__file__).resolve().parents[1] / "jem" / "envs" / "assets" / "panda.xml"
HOME_EE = (0.5545, 0.0, 0.5245)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(XML))


def mujoco_reference(m, q: np.ndarray, site: str = "ee") -> dict[str, np.ndarray]:
    """Site pose, its positional Jacobian, joint anchors and axes from CPU MuJoCo."""
    d = mujoco.MjData(m)
    sid = m.site(site).id
    out = {k: [] for k in ("site", "rot", "jac", "anchor", "axis")}
    for qi in q:
        d.qpos[:] = qi
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)  # mj_jac reads cdof and subtree_com
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jac, None, sid)
        out["site"].append(d.site_xpos[sid].copy())
        out["rot"].append(d.site_xmat[sid].reshape(3, 3).copy())
        out["jac"].append(jac)
        out["anchor"].append(d.xanchor.copy())
        out["axis"].append(d.xaxis.copy())
    return {k: np.stack(v) for k, v in out.items()}


def random_configs(m, n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(m.jnt_range[:, 0], m.jnt_range[:, 1], (n, m.nq))


@pytest.mark.parametrize("dtype, tol", [(torch.float64, 1e-9), (torch.float32, 1e-4)])
def test_chain_matches_mujoco(model, dtype, tol):
    q = random_configs(model, 64)
    ref = mujoco_reference(model, q)
    chain = SerialChain(model, "ee")
    assert chain.qpos_adr == chain.dof_adr == tuple(range(7))
    qt = torch.as_tensor(q, dtype=dtype)

    def close(x, y):
        assert x.dtype == dtype
        np.testing.assert_allclose(x.double().numpy(), y, rtol=0, atol=tol)

    x, jac = chain.site_jacobian(qt)
    close(x, ref["site"])
    close(jac, ref["jac"])
    close(chain.site(qt), ref["site"])
    close(chain.jacobian(qt), ref["jac"])
    pos, rot = chain.pose(qt)
    close(pos, ref["site"])
    close(rot, ref["rot"])
    anchor, axis = chain.frames(qt)
    close(anchor, ref["anchor"])
    close(axis, ref["axis"])
    pts = chain.points(qt)
    assert pts.shape == (64, 9, 3)
    close(pts[:, 0], np.zeros((64, 3)))
    close(pts[:, 1:-1], ref["anchor"])
    close(pts[:, -1], ref["site"])


def test_chain_shapes_autograd_and_devices(model):
    chain = SerialChain(model, "ee")
    q = torch.as_tensor(random_configs(model, 6, seed=1)).reshape(2, 3, 7)
    assert chain.site(q).shape == (2, 3, 3)
    assert chain.jacobian(q).shape == (2, 3, 3, 7)
    assert chain.points(q).shape == (2, 3, 9, 3)
    assert chain.site(q[0, 0]).shape == (3,)

    q1 = q.reshape(6, 7).requires_grad_()
    (auto,) = torch.autograd.grad(chain.site(q1)[:, 2].sum(), q1)
    torch.testing.assert_close(auto, chain.jacobian(q1.detach())[:, 2], rtol=0, atol=1e-12)

    if not torch.cuda.is_available():
        return
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training; the chain must not use it
    try:
        x, jac = SerialChain(model, "ee", "cuda").site_jacobian(q.float().cuda())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.testing.assert_close(x.cpu().double(), chain.site(q), rtol=0, atol=1e-4)
    torch.testing.assert_close(jac.cpu().double(), chain.jacobian(q), rtol=0, atol=1e-4)


def test_robot_collides_with_world_only(model):
    world = model.geom_bodyid == 0
    assert world.sum() == 1 and model.geom(int(np.flatnonzero(world)[0])).name == "floor"
    assert (model.geom_contype[world] == 0).all() and (model.geom_conaffinity[world] == 1).all()
    assert (model.geom_contype[~world] == 1).all() and (model.geom_conaffinity[~world] == 0).all()

    d = mujoco.MjData(model)
    d.qpos[:] = [0.0, 1.6, 0.0, -0.6, 0.0, 2.2, 0.0]  # reaches through the floor
    mujoco.mj_forward(model, d)
    assert d.site_xpos[model.site("ee").id][2] < 0
    robot = np.flatnonzero(~world)
    overlapping = [
        (a, b)
        for i, a in enumerate(robot)
        for b in robot[i + 1 :]
        if mujoco.mj_geomDistance(model, d, int(a), int(b), 0.01, None) < 0
    ]
    assert len(overlapping) > 0  # neighbouring links interpenetrate at every joint
    pairs = [(c.geom1, c.geom2) for c in d.contact[: d.ncon]]
    assert len(pairs) > 0
    assert all(world[g1] != world[g2] for g1, g2 in pairs)


def test_model_holds_home_under_gravity(model):
    assert model.nmesh == 0 and model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("home").id)
    mujoco.mj_forward(model, d)
    ee = model.site("ee").id
    np.testing.assert_allclose(d.site_xpos[ee], HOME_EE, atol=1e-4)
    home = d.qpos.copy()
    for _ in range(round(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, d)
    assert np.abs(d.qpos - home).max() < 0.01  # servo sag under gravity: ~6 mrad
    assert np.abs(d.qvel).max() < 1e-3 and d.ncon == 0
    assert np.linalg.norm(d.site_xpos[ee] - HOME_EE) < 0.01
