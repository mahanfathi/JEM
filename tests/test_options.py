import torch

from jem.config import make_config
from jem.data import collect
from jem.options import OptionLibrary
from jem.train import train


def test_option_library_roundtrip(tmp_path):
    run = train(make_config("smoke", ["train.device=cpu"]), root=tmp_path, quiet=True)
    lib = OptionLibrary.load(run, device="cpu")
    assert lib.library.shape == (4, lib.cfg.model.option_dim)
    assert torch.equal(lib["cw"], lib.library[lib.names.index("cw")])

    obs = lib.world.reset(5)
    a = lib.act(obs, lib["toward"] + lib["cw"])
    assert a.shape == (5, 2) and a.abs().max() <= 1

    traj = lib.rollout(lib["away"], obs, steps=7)
    assert traj.shape == (5, 8, 4)

    demos = collect(lib.world, 3, 6, torch.Generator().manual_seed(0))
    post = lib.recognize(demos.obs, demos.act)
    assert post.shape == (3, 4) and torch.allclose(post.sum(-1), torch.ones(3))
    online = lib.recognize(demos.obs, demos.act, hazard=0.1)
    assert online.shape == (3, 6, 4)
    assert lib.infer(demos.obs, demos.act).shape == (3, lib.cfg.model.option_dim)
