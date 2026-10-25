"""MjwSim: agreement with CPU MuJoCo, statelessness, batching, mocap obstacles, streams."""

import math
import time

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco", exc_type=ImportError)
mjw = pytest.importorskip("mujoco_warp", exc_type=ImportError)
if not torch.cuda.is_available():
    pytest.skip("MuJoCo Warp needs a CUDA device", allow_module_level=True)

from jem.envs.mjw import MjwSim  # noqa: E402

# A yaw-shoulder-elbow arm with position servos, above a floor it cannot reach, and a mocap
# box that collides with it (default contype/conaffinity) wherever mocap_pos puts it.
ARM = """
<mujoco model="arm3">
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.8">
      <joint name="yaw" axis="0 0 1" damping="0.5"/>
      <geom type="cylinder" size="0.04 0.05" mass="1"/>
      <body name="upper">
        <joint name="shoulder" axis="0 1 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
        <body name="fore" pos="0.3 0 0">
          <joint name="elbow" axis="0 1 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
        </body>
      </body>
    </body>
    <body name="obstacle" mocap="true" pos="2 2 0.8">
      <geom type="box" size="0.05 0.05 0.2"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="yaw" kp="60" kv="6" ctrlrange="-3 3"/>
    <position joint="shoulder" kp="100" kv="10" ctrlrange="-3 3"/>
    <position joint="elbow" kp="60" kv="6" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""
FRAME_SKIP = 10
DEV = torch.device("cuda")
BOX_IN_PATH = (0.45 * math.cos(0.6), 0.45 * math.sin(0.6), 0.8)  # on the forearm's sweep
BOX_AWAY = (2.0, 2.0, 0.8)


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_string(ARM)


@pytest.fixture(scope="module")
def sim(mjm):
    return MjwSim(mjm, DEV, FRAME_SKIP)


@pytest.fixture(scope="module")
def pressed(sim):
    """Worlds 0 and 2 swing their forearm into the box; worlds 1 and 3 swing freely."""
    mocap = torch.tensor([BOX_IN_PATH, BOX_AWAY, BOX_IN_PATH, BOX_AWAY], device=DEV)[:, None]
    q = v = torch.zeros(4, 3, device=DEV)
    ctrl = torch.tensor([1.2, 0.0, 0.0], device=DEV).repeat(4, 1)
    for _ in range(60):
        q, v = sim.step(q, v, ctrl, mocap)
    return q, v, ctrl, mocap


def test_matches_cpu_mujoco(mjm, sim):
    n, steps = 4, 50
    gen = torch.Generator(device=DEV).manual_seed(0)
    q0 = torch.rand(n, 3, generator=gen, device=DEV) - 0.5
    v0 = torch.rand(n, 3, generator=gen, device=DEV) - 0.5
    ctrl = 2 * torch.rand(steps, n, 3, generator=gen, device=DEV) - 1
    q, v, traj = q0, v0, []
    for t in range(steps):
        q, v = sim.step(q, v, ctrl[t])
        traj.append(q)
    traj, ctrl = torch.stack(traj).double().cpu().numpy(), ctrl.double().cpu().numpy()
    d = mujoco.MjData(mjm)
    for i in range(n):
        mujoco.mj_resetData(mjm, d)
        d.qpos, d.qvel = q0[i].cpu().numpy(), v0[i].cpu().numpy()
        for t in range(steps):
            d.ctrl = ctrl[t, i]
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(mjm, d)
            assert d.ncon == 0
            np.testing.assert_allclose(traj[t, i], d.qpos, rtol=0, atol=1e-4)


def test_mocap_obstacle_blocks_only_where_placed(sim, pressed):
    yaw = pressed[0][:, 0]
    assert (yaw[[0, 2]] < 0.5).all(), yaw  # the box, centred at yaw 0.6, stops them at ~0.39
    assert torch.allclose(yaw[[1, 3]], torch.tensor(1.2, device=DEV), atol=0.01), yaw
    assert not sim.overflow()


def test_step_is_a_pure_function(sim, pressed):
    q, v, ctrl, mocap = pressed  # the forearm is in contact: the solver's warm start matters
    first = sim.step(q, v, ctrl, mocap)
    kept = [x.clone() for x in first]
    sim.step(q.flip(0) - 1, 3 * v + 1, -ctrl, mocap.flip(0))  # unrelated call, same batch size
    again = sim.step(q, v, ctrl, mocap)
    # One contact per world fixes the constraint row order, so equality is exact; a warm
    # start carried over from the unrelated call shows up at ~1e-8.
    for a, b, c in zip(first, kept, again, strict=True):
        assert torch.equal(a, b)  # outputs are not views a later call overwrites
        assert torch.equal(a, c)


def test_inputs_needing_grad_give_plain_outputs(sim, pressed):
    q, v, ctrl, mocap = pressed
    out = sim.step(q.clone().requires_grad_(), v, ctrl, mocap)
    assert not out[0].requires_grad and not out[1].requires_grad
    assert torch.equal(out[0], sim.step(q, v, ctrl, mocap)[0])


def test_batch_size_does_not_change_results(sim, pressed):
    q, v, ctrl, mocap = pressed
    ref = sim.step(q, v, ctrl, mocap)
    for n in (1, 3, 64):
        rows = torch.arange(n, device=DEV) % 4
        out = sim.step(q[rows], v[rows], ctrl[rows], mocap[rows])
        for a, b in zip(out, ref, strict=True):
            torch.testing.assert_close(a, b[rows], rtol=0, atol=1e-6)
    away = torch.tensor(BOX_AWAY, device=DEV).expand(4, 1, 3)
    for a, b in zip(sim.step(q, v, ctrl), sim.step(q, v, ctrl, away), strict=True):
        assert torch.equal(a, b)  # mocap_pos=None means the model's own box position


@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_step_never_syncs_the_host(sim, pressed):
    q, v, ctrl, mocap = pressed
    torch.cuda.set_sync_debug_mode("error")  # torch ops that synchronise now raise
    try:
        for _ in range(5):
            q, v = sim.step(q, v, ctrl, mocap)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    # Nor does Warp: queued behind ~0.4 s of GPU work, a call returns at once.
    torch.cuda.synchronize()
    torch.cuda._sleep(1_000_000_000)
    start = time.perf_counter()
    sim.step(q, v, ctrl, mocap)
    elapsed = time.perf_counter() - start
    busy = not torch.cuda.current_stream().query()
    torch.cuda.synchronize()
    assert busy and elapsed < 0.1


def test_runs_in_order_on_torch_current_stream(sim, pressed):
    q, v, ctrl, mocap = pressed
    ref = sim.step(q, v, ctrl, mocap)
    sim.step(-q, v, -ctrl, mocap.flip(0))  # leave other values in the simulator
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(200_000_000)  # a slow producer: the input is ready only later
        out = sim.step(q.clone(), v, ctrl, mocap)
        consumed = out[0] + 0
    torch.cuda.synchronize()
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    assert torch.equal(consumed, ref[0])


def test_overflow_is_flagged_and_cleared(mjm, pressed):
    tiny = MjwSim(mjm, DEV, FRAME_SKIP, njmax=2)  # a condim-3 contact needs 4 rows
    tiny.step(*pressed)
    assert tiny.overflow() & mjw.OverflowType.NEFC
    assert tiny.overflow() == mjw.OverflowType.NONE  # reading clears the flags


def test_rejects_state_outside_qpos_qvel():
    xml = """<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
      <actuator><general joint="j" dyntype="filter" dynprm="0.1"/></actuator></mujoco>"""
    with pytest.raises(ValueError, match="activations"):
        MjwSim(mujoco.MjModel.from_xml_string(xml), DEV)
