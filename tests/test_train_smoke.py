import json

import pytest

from jem.config import make_config
from jem.train import train


@pytest.mark.parametrize(
    "preset, extra",
    [
        ("smoke", []),
        ("smoke", ["model.arch=joint"]),
        ("smoke", ["infer.implicit_grad=false", "objective.symmetric=false"]),
        (
            "smoke",
            [
                "model.kind=mlp",
                "model.squared=true",
                "infer.method=langevin",
                "objective.name=legacy",
                "objective.action_chains=4",
                "objective.klz_chains=2",
                "data.context_len=2",
                "data.target_len=2",
            ],
        ),
    ],
)
def test_train_smoke(tmp_path, preset, extra):
    cfg = make_config(preset, ["train.device=cpu", *extra])
    path = train(cfg, root=tmp_path, quiet=True)
    status = json.loads((path / "status.json").read_text())
    assert status["state"] == "finished"
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert any("train/loss" in r for r in rows)
    evals = [r for r in rows if "eval/nmi" in r]
    assert evals and 0.0 <= evals[-1]["eval/nmi"] <= 1.0
    assert (path / "checkpoints" / "last.pt").exists()
    assert {p.name for p in (path / "figures").iterdir()} >= {"options", "rollouts"}
