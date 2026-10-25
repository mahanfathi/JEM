"""Training loop: fixed offline demonstrations, periodic evaluation, local run logging."""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from jem.config import Config
from jem.data import collect, sample_windows
from jem.energy import build_model
from jem.envs import make_world
from jem.evaluate import evaluate, kmeans
from jem.figures import make_figures
from jem.inference import Proposals, infer_newton
from jem.objectives import candidates, jem_loss, legacy_loss
from jem.prior import BehaviourPrior
from jem.runlog import RunLog


def resolve_device(name: str) -> torch.device:
    """Pick the device and the matmul precision. Every entry point (training, `jem eval`,
    `jem downstream`, OptionLibrary) goes through here, so all of them compute alike.
    TF32 is ~15% faster here and leaves the Newton residual unchanged (1.4e-6)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_data(cfg: Config, device):
    world = make_world(cfg.env, device)
    gen = torch.Generator(device=device).manual_seed(cfg.data.seed)
    train = collect(world, cfg.data.num_traj, cfg.data.traj_len, gen, cfg.env.switch_prob)
    val = collect(world, cfg.data.val_traj, cfg.data.traj_len, gen, cfg.env.switch_prob)
    return world, train, val


def build_prior(cfg: Config, world, device) -> BehaviourPrior | None:
    oc = cfg.objective
    if oc.proposal == "uniform":
        return None
    if oc.proposal != "prior":
        raise KeyError(f"unknown proposal {oc.proposal!r}")
    return BehaviourPrior(
        world.obs_dim,
        world.act_dim,
        world.act_low,
        world.act_high,
        oc.prior_components,
        oc.prior_hidden,
    ).to(device)


def fit_prior(prior: BehaviourPrior, demos, cfg: Config, device) -> float:
    """Maximum likelihood on single (s, a) pairs of the unlabelled demonstrations, then
    freeze: a fixed proposal (and base measure) for the whole run. Returns the final NLL."""
    gen = torch.Generator(device=device).manual_seed(cfg.train.seed + 1)
    obs = demos.obs.reshape(-1, demos.obs.shape[-1])
    act = demos.act.reshape(-1, demos.act.shape[-1])
    opt = torch.optim.Adam(prior.parameters(), lr=cfg.train.lr)
    steps = cfg.objective.prior_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps))
    nll = torch.zeros((), device=device)
    for _ in range(steps):
        idx = torch.randint(len(obs), (4096,), generator=gen, device=device)
        nll = -prior.log_prob(obs[idx], act[idx]).mean()
        opt.zero_grad(set_to_none=True)
        nll.backward()
        opt.step()
        sched.step()
    prior.requires_grad_(False)
    return prior.eval().log_prob(obs[:65536], act[:65536]).mean().neg().item()


@torch.no_grad()
def init_prototypes(model, demos, cfg: Config, proposals, prior, gen) -> None:
    """Prototype options start as k-means centroids of options inferred from contexts."""
    nc, length = cfg.data.context_len, cfg.data.context_len + cfg.data.target_len
    zs = []
    for _ in range(16):
        win = sample_windows(demos, cfg.train.batch_size, length, gen)
        phi, offs = candidates(model, win, proposals.sample(gen), cfg, prior, gen)
        offs = None if offs is None else offs[:, :nc]
        U = model.scale * phi[:, :nc].float()
        zs.append(infer_newton(U, cfg.infer.prior_std**-2, cfg.infer.iters, False, None, offs)[0])
    centroids, _ = kmeans(torch.cat(zs), model.prototypes.shape[0], gen)
    model.prototypes.copy_(centroids)


def _schedule(cfg: Config):
    tc = cfg.train

    def factor(step: int) -> float:
        if step < tc.warmup:
            return (step + 1) / tc.warmup
        progress = (step - tc.warmup) / max(1, tc.steps - tc.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return factor


def run_eval(model, world, val, cfg: Config, run: RunLog, step: int) -> dict:
    result = evaluate(model, world, val, cfg)
    run.log(step, result["metrics"])
    if cfg.eval.figures:
        for name, fig in make_figures(model, world, result).items():
            run.figure(step, name, fig)
            plt.close(fig)
    return result


def train(cfg: Config, root: str | Path = "runs", quiet: bool = False) -> Path:
    tc = cfg.train
    device = resolve_device(tc.device)
    torch.manual_seed(tc.seed)

    world, train_demos, val_demos = build_data(cfg, device)
    model = build_model(cfg.model, world.obs_dim, world.act_dim).to(device)
    prior = build_prior(cfg, world, device)
    prior_nll = None if prior is None else fit_prior(prior, train_demos, cfg, device)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _schedule(cfg))
    proposals = Proposals(
        world.act_dim, world.act_low, world.act_high, cfg.objective.num_proposals, device, tc.seed
    )
    gen = torch.Generator(device=device).manual_seed(tc.seed)
    length = cfg.data.context_len + cfg.data.target_len

    run = RunLog.create(
        root,
        cfg.name,
        cfg.to_dict(),
        {"device": str(device), "params": sum(p.numel() for p in model.parameters())},
    )
    say = (lambda *a: None) if quiet else print
    say(f"[jem] run {run.path}  ({sum(p.numel() for p in model.parameters()):,} params, {device})")
    if prior_nll is not None:
        run.log(0, {"train/prior_nll": prior_nll})

    step, state, message = 0, "finished", ""
    proto_pending = getattr(model, "prototypes", None) is not None
    tick, tick_step = time.time(), 0
    try:
        for step in range(1, tc.steps + 1):
            win = sample_windows(
                train_demos, tc.batch_size, length, gen, cfg.data.context_len, cfg.data.target_gap
            )
            if cfg.objective.name == "jem":
                if proto_pending and step > cfg.objective.prototype_start:
                    init_prototypes(model, train_demos, cfg, proposals, prior, gen)
                    proto_pending = False
                active = getattr(model, "prototypes", None) is not None and not proto_pending
                loss, stats = jem_loss(
                    model, win, proposals.sample(gen), cfg, prior, gen, prototypes=active
                )
            elif cfg.objective.name == "legacy":
                loss, stats = legacy_loss(model, win, cfg, gen)
            else:
                raise KeyError(f"unknown objective {cfg.objective.name!r}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
            opt.step()
            sched.step()

            if step % tc.log_every == 0 or step == tc.steps:
                now = time.time()
                row = {"train/loss": loss.detach(), "train/grad_norm": grad_norm, **stats}
                row["train/lr"] = sched.get_last_lr()[0]
                row["perf/steps_per_s"] = (step - tick_step) / max(now - tick, 1e-9)
                tick, tick_step = now, step
                run.log(step, row)
                if not (math.isfinite(loss.item()) and math.isfinite(grad_norm.item())):
                    state, message = "diverged", f"non-finite loss/grad at step {step}"
                    break
                if step % (tc.log_every * 10) == 0:
                    say(f"[jem] step {step:>6}  loss {loss.item():.4f}")
                run.set_status("running", step)

            if step % tc.eval_every == 0 or step == tc.steps:
                result = run_eval(model, world, val_demos, cfg, run, step)
                metrics = result["metrics"]
                headline = {k: round(v, 3) for k, v in metrics.items() if k.count("@") == 0}
                say(f"[jem] eval {step}: {headline}")
                run.save_checkpoint(
                    {
                        "model": model.state_dict(),
                        "prior": None if prior is None else prior.state_dict(),
                        "config": cfg.to_dict(),
                        "step": step,
                        # Discovered options; names are attached post hoc by evaluation.
                        "library": result["artifacts"]["library"].cpu(),
                        "names": list(world.expert_names),
                    }
                )
    except Exception as exc:  # keep a record of the failure in the run directory
        run.set_status("failed", step, f"{type(exc).__name__}: {exc}")
        run.close()
        raise
    run.set_status(state, step, message)
    run.close()
    say(f"[jem] {state}: {run.path}")
    return run.path
