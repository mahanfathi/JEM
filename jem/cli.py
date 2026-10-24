"""Command line: `jem train | eval | dash | presets`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _train(args) -> None:
    from jem.config import make_config
    from jem.train import train

    overrides = list(args.set or [])
    if args.name:
        overrides.append(f"name={args.name}")
    train(make_config(args.preset, overrides), root=args.root)


def _eval(args) -> None:
    import matplotlib.pyplot as plt
    import torch

    from jem.config import from_dict
    from jem.energy import build_model
    from jem.evaluate import evaluate
    from jem.figures import make_figures
    from jem.train import build_data, resolve_device

    run = Path(args.run)
    ckpt = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(ckpt["config"])
    device = resolve_device(cfg.train.device)
    world, _, val = build_data(cfg, device)
    model = build_model(cfg.model, world.obs_dim, world.act_dim).to(device)
    model.load_state_dict(ckpt["model"])
    result = evaluate(model, world, val, cfg)
    print(json.dumps(result["metrics"], indent=2))
    if args.figures:
        out = Path(args.figures)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in make_figures(model, world, result).items():
            fig.savefig(out / f"{name}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
        print(f"figures -> {out}")


def _dash(args) -> None:
    from jem.dashboard import serve

    serve(args.root, host=args.host, port=args.port)


def _presets(_args) -> None:
    from jem.config import PRESETS

    for name, overrides in PRESETS.items():
        print(f"{name}: {json.dumps(overrides) if overrides else '(defaults)'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jem", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train a model (see `jem presets`)")
    p.add_argument("--preset", default="jem")
    p.add_argument("--name", default=None, help="run name (defaults to the preset)")
    p.add_argument(
        "--set", nargs="*", metavar="KEY=VALUE", help="config overrides, e.g. model.option_dim=2"
    )
    p.add_argument("--root", default="runs")
    p.set_defaults(fn=_train)

    p = sub.add_parser("eval", help="re-evaluate a run's last checkpoint")
    p.add_argument("run", help="run directory")
    p.add_argument("--figures", default=None, help="directory to write figures to")
    p.set_defaults(fn=_eval)

    p = sub.add_parser("dash", help="serve the local dashboard")
    p.add_argument("--root", default="runs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=_dash)

    p = sub.add_parser("presets", help="list presets")
    p.set_defaults(fn=_presets)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
