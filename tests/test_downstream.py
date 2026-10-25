import math

import pytest
import torch

from jem.config import make_config
from jem.downstream import (
    PLANNERS,
    DownstreamConfig,
    disc_tasks,
    evaluate_downstream,
    step_with_obstacles,
)
from jem.envs import make_world
from jem.options import OptionLibrary
from jem.train import train


@pytest.mark.parametrize("kind", ["circle", "cup"])
def test_tasks_start_outside_obstacles_and_projection_keeps_them_out(kind):
    obs, centers, radii = disc_tasks(64, torch.Generator().manual_seed(0), "cpu", kind)
    gap = (obs[:, None, :2] - centers).norm(dim=-1) - radii
    assert (gap > 0).all()
    world = make_world(make_config("jem").env)
    toward = obs[:, 2:] - obs[:, :2]
    for _ in range(60):  # push straight at the target: nobody may end up inside a disc
        obs = step_with_obstacles(
            world, obs, toward / toward.norm(dim=-1, keepdim=True), centers, radii
        )
        toward = obs[:, 2:] - obs[:, :2]
        assert ((obs[:, None, :2] - centers).norm(dim=-1) >= radii - 1e-5).all()


def test_planners_run_end_to_end(tmp_path):
    lib = OptionLibrary.load(
        train(make_config("smoke", ["train.device=cpu"]), tmp_path, quiet=True), "cpu"
    )
    cfg = DownstreamConfig(
        episodes=2, steps=6, horizon=3, replan=2, candidates=8, iters=2, task="cup"
    )
    results = evaluate_downstream(lib, cfg)
    assert set(results) == set(PLANNERS)
    for res in results.values():
        assert 0.0 <= res["success"] <= 1.0
        assert res["traj"].shape == (2, 7, 4)
        assert math.isnan(res["steps_to_target"]) or res["steps_to_target"] <= 6


def test_planners_run_with_option_sequences(tmp_path):
    lib = OptionLibrary.load(
        train(make_config("smoke", ["train.device=cpu"]), tmp_path, quiet=True), "cpu"
    )
    cfg = DownstreamConfig(
        episodes=2, steps=6, horizon=4, replan=2, candidates=8, iters=2, task="cup", segments=2
    )
    results = evaluate_downstream(lib, cfg)
    assert set(results) == set(PLANNERS)
    for res in results.values():
        assert res["traj"].shape == (2, 7, 4)
