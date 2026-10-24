"""Command line: `jem train | eval | table | downstream | dash | presets`."""

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

    from jem.config import _set, from_dict
    from jem.energy import build_model
    from jem.evaluate import evaluate
    from jem.figures import make_figures
    from jem.train import build_data, resolve_device

    run = Path(args.run)
    ckpt = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(ckpt["config"])
    for item in args.set or []:  # e.g. env.switch_prob=0 to score on non-switching demos
        key, _, value = item.partition("=")
        _set(cfg, key.strip(), value.strip())
    device = resolve_device(cfg.train.device)
    world, _, val = build_data(cfg, device)
    model = build_model(cfg.model, world.obs_dim, world.act_dim).to(device)
    model.load_state_dict(ckpt["model"])
    result = evaluate(model, world, val, cfg)
    print(json.dumps(result["metrics"], indent=2))
    if args.log:  # append to the run's own record, at the checkpoint's step
        from jem.runlog import RunLog

        log = RunLog(run)
        log.log(ckpt["step"], result["metrics"])
        log.close()
        if "library" not in ckpt:  # backfill checkpoints that predate the stored library
            ckpt["library"] = result["artifacts"]["library"].cpu()
            ckpt["names"] = list(world.expert_names)
            log.save_checkpoint(ckpt)
    if args.figures:
        out = Path(args.figures)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in make_figures(model, world, result).items():
            fig.savefig(out / f"{name}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
        print(f"figures -> {out}")


def _downstream(args) -> None:
    import dataclasses as dc

    import matplotlib.pyplot as plt

    from jem.downstream import DownstreamConfig, evaluate_downstream
    from jem.figures import downstream_figure
    from jem.options import OptionLibrary
    from jem.runlog import RunLog

    run = Path(args.run)
    lib = OptionLibrary.load(run)
    cfg = DownstreamConfig(
        episodes=args.episodes, seed=args.seed, task=args.task, horizon=args.horizon
    )
    results = evaluate_downstream(lib, cfg)
    print("| planner | success | steps to target |\n|---|---|---|")
    for name, res in results.items():
        print(f"| {name} | {100 * res['success']:.1f}% | {res['steps_to_target']:.1f} |")
    step = json.loads((run / "status.json").read_text())["step"]
    log = RunLog(run)
    tag = f"{cfg.task}_h{cfg.horizon}"
    metrics = {}
    for name, res in results.items():
        metrics[f"down/{tag}/{name}_success"] = res["success"]
        metrics[f"down/{tag}/{name}_steps"] = res["steps_to_target"]
    log.log(step, metrics)
    log.figure(step, f"downstream_{tag}", fig := downstream_figure(results))
    plt.close(fig)
    log.close()
    summary = {
        k: {m: v for m, v in r.items() if m in ("success", "steps_to_target")}
        for k, r in results.items()
    }
    (run / f"downstream_{tag}.json").write_text(
        json.dumps({"config": dc.asdict(cfg), "results": summary}, indent=2)
    )


def _dash(args) -> None:
    from jem.dashboard import serve

    serve(args.root, host=args.host, port=args.port)


def _table(args) -> None:
    """Markdown table of the last eval of each run, seeds (`<name>-s<k>`) pooled as mean ± std."""
    import re
    import statistics

    keys = args.keys.split(",")
    groups: dict[str, list[dict]] = {}
    for run in sorted(Path(args.root).iterdir()):
        metrics = run / "metrics.jsonl"
        if not metrics.exists():
            continue
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if '"eval/' in line]
        if args.step:
            rows = [r for r in rows if r["step"] == args.step]
        if not rows:
            continue
        name = json.loads((run / "config.json").read_text())["meta"]["name"]
        if args.match and not re.search(args.match, name):
            continue
        groups.setdefault(re.sub(r"-s\d+$", "", name), []).append(rows[-1])

    def cell(values: list[float]) -> str:
        if not values:
            return "–"
        if len(values) == 1:
            return f"{values[0]:.3g}"
        return f"{statistics.mean(values):.3g} ± {statistics.stdev(values):.2g}"

    print("| run | n | " + " | ".join(k.removeprefix("eval/") for k in keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for name, rows in groups.items():
        cells = [cell([r[k] for r in rows if r.get(k) is not None]) for k in keys]
        print(f"| {name} | {len(rows)} | " + " | ".join(cells) + " |")


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
    p.add_argument(
        "--log", action="store_true", help="append the metrics to the run's metrics.jsonl"
    )
    p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="config overrides for evaluation")
    p.set_defaults(fn=_eval)

    p = sub.add_parser("downstream", help="obstacle-room MPC with a run's options vs raw actions")
    p.add_argument("run", help="run directory")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", default="circle", choices=["circle", "cup"])
    p.add_argument("--horizon", type=int, default=20, help="planning horizon (steps)")
    p.set_defaults(fn=_downstream)

    p = sub.add_parser("dash", help="serve the local dashboard")
    p.add_argument("--root", default="runs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=_dash)

    p = sub.add_parser("table", help="markdown table of final eval metrics per run")
    p.add_argument("--root", default="runs")
    p.add_argument("--match", default=None, help="regex on run names")
    p.add_argument("--step", type=int, default=None, help="use the eval at this step")
    p.add_argument(
        "--keys",
        default="eval/nmi,eval/policy_kl,eval/heading_err_deg,eval/compose_err_deg,"
        "eval/algebra_err_deg,eval/antipodal_cos,eval/rollout_agree,eval/track_acc",
    )
    p.set_defaults(fn=_table)

    p = sub.add_parser("presets", help="list presets")
    p.set_defaults(fn=_presets)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
