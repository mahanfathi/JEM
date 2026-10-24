"""A trained option library: recognise, execute and compose options.

    lib = OptionLibrary.load("runs/<run_id>")
    spiral = lib["toward"] + lib["cw"]                  # composing = adding options
    traj = lib.rollout(spiral, lib.world.reset(8), steps=60)
    belief = lib.recognize(obs, act)                    # (B, K) posterior over the library
    z = lib.infer(obs, act)                             # MAP option for a new snippet

The library holds the k-means centroids of options inferred from held-out windows.
Their names were attached after the fact by matching to expert labels during evaluation;
training never saw labels.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from jem.config import from_dict
from jem.energy import build_model
from jem.envs import make_world
from jem.evaluate import hmm_filter, option_logits, rollout, step_loglik
from jem.inference import Proposals, infer_option
from jem.train import resolve_device


class OptionLibrary:
    def __init__(self, model, world, cfg, library: Tensor, names: list[str]):
        self.model, self.world, self.cfg = model.eval(), world, cfg
        self.library, self.names = library, list(names)
        device = library.device
        self.proposals = Proposals(world.act_dim, world.act_low, world.act_high, 1024, device, 23)
        self.gen = torch.Generator(device=device).manual_seed(0)

    @classmethod
    def load(cls, run: str | Path, device: str = "auto") -> OptionLibrary:
        ckpt = torch.load(
            Path(run) / "checkpoints" / "last.pt", map_location="cpu", weights_only=False
        )
        cfg = from_dict(ckpt["config"])
        dev = resolve_device(device)
        world = make_world(cfg.env, dev)
        model = build_model(cfg.model, world.obs_dim, world.act_dim).to(dev)
        model.load_state_dict(ckpt["model"])
        if "library" not in ckpt:  # older checkpoints: rebuild the library by evaluation
            from jem.evaluate import evaluate
            from jem.train import build_data

            _, _, val = build_data(cfg, dev)
            library = evaluate(model, world, val, cfg)["artifacts"]["library"]
            return cls(model, world, cfg, library, list(world.expert_names))
        return cls(model, world, cfg, ckpt["library"].to(dev), ckpt["names"])

    def __getitem__(self, name: str) -> Tensor:
        return self.library[self.names.index(name)]

    @torch.no_grad()
    def logits(self, obs: Tensor, z: Tensor, acts: Tensor) -> Tensor:
        """-E(s, z, a) for states (S, ds) and candidate actions (S|1, K, da) -> (S, K)."""
        return option_logits(self.model, obs, z.reshape(1, -1), acts)[:, 0]

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, greedy: bool = False) -> Tensor:
        """Sample (or take the mode of) a ~ p(a | s, z) over 1024 fresh proposals."""
        props = self.proposals.sample(self.gen)
        u = self.logits(obs, z, props[None])
        idx = (
            u.argmax(-1)
            if greedy
            else torch.multinomial(u.softmax(-1), 1, generator=self.gen)[:, 0]
        )
        return props[idx]

    def rollout(self, z: Tensor, start: Tensor, steps: int) -> Tensor:
        """Closed-loop trajectory (R, steps+1, ds) under option z."""
        traj, _ = rollout(
            self.model, self.world, z.reshape(1, -1), start, steps, self.proposals, self.gen
        )
        return traj

    def infer(self, obs: Tensor, act: Tensor) -> Tensor:
        """MAP option (B, d) for snippets obs (B, N, ds), act (B, N, da)."""
        with torch.no_grad():
            return infer_option(self.model, obs, act, self.cfg, self.proposals.sample(self.gen))

    @torch.no_grad()
    def recognize(self, obs: Tensor, act: Tensor, hazard: float | None = None) -> Tensor:
        """Posterior over library options: (B, K) for a fixed option, or online beliefs
        (B, N, K) under a switching prior with the given per-step hazard."""
        ll = step_loglik(self.model, obs, act, self.library, self.proposals, self.gen)
        if hazard is None:
            return ll.sum(1).softmax(-1)
        return hmm_filter(ll, hazard)
