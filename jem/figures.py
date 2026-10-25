"""Evaluation figures (matplotlib, saved as PNG into the run directory).

Experts use fixed categorical slots, with a marker shape as a second channel because
four hues exceed what stays distinguishable for colour-blind readers in a scatter.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from jem.envs import ParticleWorld  # noqa: E402
from jem.evaluate import option_logits, rollout  # noqa: E402
from jem.inference import Proposals  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
DENSITY = LinearSegmentedColormap.from_list(
    "jem_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
# Agent positions for probe states; the target sits at the origin.
PROBES = ((0.6, 0.0), (-0.35, 0.5), (0.2, -0.7))

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def _probe(device) -> torch.Tensor:
    return torch.tensor([[x, y, 0.0, 0.0] for x, y in PROBES], device=device)


def _density(model, obs: torch.Tensor, zs: torch.Tensor, res: int = 96) -> torch.Tensor:
    """p(a | s, sum of options) on a res x res grid over [-1, 1]^2 -> (S, res, res)."""
    lin = torch.linspace(-1, 1, res, device=obs.device)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    grid = torch.stack([gx, gy], -1).reshape(-1, 2)
    u = option_logits(model, obs, zs, grid[None]).sum(1)
    return u.softmax(-1).reshape(-1, res, res)


def _action_panel(ax, dens, title: str) -> None:
    ax.imshow(
        dens.cpu(), origin="lower", extent=(-1, 1, -1, 1), cmap=DENSITY, interpolation="bilinear"
    )
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=GRID, lw=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, pad=3)


def _arrow(ax, vec, color, style="-", width=1.6) -> None:
    x, y = (float(v) for v in vec)
    ax.annotate(
        "",
        xy=(0.85 * x, 0.85 * y),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=width, linestyle=style, shrinkA=0, shrinkB=0
        ),
    )


def options_figure(z, labels, library, names) -> plt.Figure:
    zc = z - z.mean(0)
    if z.shape[1] == 1:
        proj = torch.cat([z, 0.05 * torch.randn_like(z)], 1)
        lib = torch.cat([library, torch.zeros_like(library)], 1)
        xlabel, ylabel = "z", "jitter"
    elif z.shape[1] == 2:
        proj, lib, xlabel, ylabel = z, library, "z1", "z2"
    else:
        _, s, v = torch.linalg.svd(zc, full_matrices=False)
        proj, lib = zc @ v[:2].T, (library - z.mean(0)) @ v[:2].T
        frac = (s[:2] ** 2).sum() / (s**2).sum()
        xlabel, ylabel = "PC1", f"PC2  ({100 * frac:.0f}% var. in 2 PCs)"
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    proj, lib, labels = proj.cpu(), lib.cpu(), labels.cpu()
    for e, name in enumerate(names):
        pts = proj[labels == e]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=9,
            alpha=0.35,
            color=SERIES[e],
            marker=MARKERS[e],
            lw=0,
            label=name,
        )
    for e, name in enumerate(names):
        ax.scatter(
            *lib[e], s=90, color=SERIES[e], marker=MARKERS[e], edgecolor=SURFACE, lw=2, zorder=3
        )
        ax.annotate(name, lib[e], xytext=(7, 6), textcoords="offset points", color=INK, fontsize=8)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Inferred options, coloured by the (hidden) expert")
    ax.legend(loc="best", fontsize=7, markerscale=1.6, handletextpad=0.3)
    return fig


def policies_figure(model, world: ParticleWorld, library, names) -> plt.Figure:
    obs = _probe(library.device)
    k = len(names)
    fig, axes = plt.subplots(len(PROBES), k, figsize=(1.75 * k, 1.85 * len(PROBES)), squeeze=False)
    for e in range(k):
        dens = _density(model, obs, library[e : e + 1])
        mean_dir = world.expert_mean_dir(obs, torch.full((len(obs),), e, device=obs.device))
        for r in range(len(PROBES)):
            ax = axes[r, e]
            _action_panel(ax, dens[r], names[e] if r == 0 else "")
            _arrow(ax, mean_dir[r].cpu(), SERIES[e])
            if e == 0:
                ax.set_ylabel(f"agent at {PROBES[r]}", fontsize=7)
    fig.suptitle(
        "p(a | s, z) per discovered option  (arrow: matched expert's mean heading)", fontsize=9
    )
    fig.tight_layout()
    return fig


def compositions_figure(model, world: ParticleWorld, library, names, pairs) -> plt.Figure:
    obs = _probe(library.device)[:1]
    fig, axes = plt.subplots(1, len(pairs), figsize=(1.9 * len(pairs), 2.2), squeeze=False)
    for c, (i, j) in enumerate(pairs):
        ax = axes[0, c]
        dens = _density(model, obs, library[[i, j]])
        _action_panel(ax, dens[0], f"{names[i]} + {names[j]}")
        for e in (i, j):
            _arrow(
                ax,
                world.expert_mean_dir(obs, torch.tensor([e], device=obs.device))[0].cpu(),
                SERIES[e],
                width=1.2,
            )
        ideal = world.ideal_composition_dir(obs, i, j)[0]
        if ideal.norm() > 1e-3:
            _arrow(ax, (ideal / ideal.norm()).cpu(), INK, style="--", width=1.2)
    fig.suptitle(
        "Zero-shot composition: p(a|s,z_i) p(a|s,z_j)  (dashed: product-of-experts bisector)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def rollouts_figure(rollouts: dict, names) -> plt.Figure:
    keys = list(rollouts)
    cols = min(4, len(keys))
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.35 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for c, key in enumerate(keys):
        ax = axes.flat[c]
        ax.set_visible(True)
        traj = rollouts[key].cpu()
        rel = traj[..., :2] - traj[:, :1, 2:]  # each episode centred on its own target
        parts = key.split("+")
        color = SERIES[names.index(parts[0])] if len(parts) == 1 else INK2
        for r in range(rel.shape[0]):
            ax.plot(rel[r, :, 0], rel[r, :, 1], color=color, lw=1.0, alpha=0.65)
            ax.scatter(rel[r, 0, 0], rel[r, 0, 1], s=8, color=color, lw=0)
        ax.scatter([0], [0], marker="*", s=90, color=INK, zorder=3)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(key)
    fig.suptitle(
        "Closed-loop rollouts (relative to target ★): options and sums of options", fontsize=9
    )
    fig.tight_layout()
    return fig


def recognition_figure(recog: dict, bayes: dict, num_experts: int) -> plt.Figure:
    ns = sorted(recog)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.plot(
        ns,
        [bayes[n] for n in ns],
        color=MUTED,
        lw=1.2,
        marker="o",
        ms=3,
        label="Bayes (true densities)",
    )
    ax.plot(ns, [recog[n] for n in ns], color=SERIES[0], marker="o", ms=5, label="model")
    ax.axhline(1 / num_experts, color=MUTED, lw=1, ls="--", label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="center right", fontsize=7)
    ax.set_xlabel("context length N (steps)")
    ax.set_ylabel("accuracy")
    ax.set_title("Option recognition by energy: argmax_k Σ_t log p(a_t|s_t,z_k)")
    return fig


def algebra_figure(model, world: ParticleWorld, library, names, pair) -> plt.Figure:
    """Rollouts of z_i + w z_j: the heading should turn by atan(w) away from expert i."""
    i, j = pair
    weights = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    gen = torch.Generator(device=library.device).manual_seed(5)
    angles = torch.linspace(0, 2 * math.pi, 7, device=library.device)[:-1] + 0.3
    start = torch.stack(
        [0.9 * angles.cos(), 0.9 * angles.sin(), 0 * angles, 0 * angles], -1
    )  # agents on a ring, target at the origin
    sir = Proposals(world.act_dim, world.act_low, world.act_high, 1024, library.device, seed=19)
    fig, axes = plt.subplots(1, len(weights) + 1, figsize=(1.9 * (len(weights) + 1), 2.3))
    panels = [(f"w={w:g}", [1.0, w]) for w in weights] + [(names[j] + " only", [0.0, 1.0])]
    for ax, (title, wts) in zip(axes, panels, strict=True):
        traj, _ = rollout(model, world, library[[i, j]], start, 60, sir, gen, weights=wts)
        xy = traj[..., :2].cpu()
        for r in range(xy.shape[0]):
            ax.plot(xy[r, :, 0], xy[r, :, 1], color=SERIES[0], lw=1.0, alpha=0.8)
            ax.scatter(xy[r, 0, 0], xy[r, 0, 1], s=8, color=SERIES[0], lw=0)
        ax.scatter([0], [0], marker="*", s=80, color=INK, zorder=3)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if title.startswith("w="):  # product of Gaussian-heading experts: 90° w / (1 + w)
            title += f"  ({90 * float(title[2:]) / (1 + float(title[2:])):.0f}°)"
        ax.set_title(title, fontsize=8)
    fig.suptitle(
        f"Behaviour algebra: z_{names[i]} + w·z_{names[j]}  (true product turns 90°·w/(1+w))",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def tracking_figure(track: dict, names) -> plt.Figure:
    expert, belief, bayes = (track[k].cpu() for k in ("expert", "belief", "bayes"))
    t = torch.arange(len(expert))
    fig, axes = plt.subplots(
        3, 1, figsize=(6.4, 3.4), sharex=True, gridspec_kw={"height_ratios": [0.35, 1, 1]}
    )
    axes[0].bar(t, 1.0, width=1.0, color=[SERIES[e] for e in expert.tolist()], lw=0)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_ylabel("true", rotation=0, ha="right", va="center")
    for ax, b, label in ((axes[1], belief, "model"), (axes[2], bayes, "Bayes")):
        ax.stackplot(t, b.T, colors=SERIES[: len(names)], labels=names, step="mid", lw=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    axes[2].set_xlabel("time step")
    fig.suptitle(
        "Online option tracking under hidden switches (HMM filter over the library)", fontsize=9
    )
    fig.tight_layout()
    return fig


def downstream_figure(results: dict, episodes: int = 6) -> plt.Figure:
    """Paths of each planner (columns) on the same obstacle tasks (rows)."""
    names = list(results)
    fig, axes = plt.subplots(episodes, len(names), figsize=(1.9 * len(names), 1.9 * episodes))
    for c, name in enumerate(names):
        res = results[name]
        traj = res["traj"].cpu()
        centers, radii = res["task"].centers.cpu(), res["task"].radii.cpu()
        for r in range(episodes):
            ax = axes[r, c]
            for o in range(centers.shape[1]):
                ax.add_patch(plt.Circle(centers[r, o].tolist(), radii[r, o].item(), color=GRID))
            ax.plot(traj[r, :, 0], traj[r, :, 1], color=SERIES[c], lw=1.2)
            ax.scatter(traj[r, 0, 0], traj[r, 0, 1], s=10, color=INK2, zorder=3)
            ax.scatter(traj[r, 0, 2], traj[r, 0, 3], marker="*", s=70, color=INK, zorder=3)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{name}\nsuccess {100 * res['success']:.0f}%", fontsize=8)
    fig.suptitle(
        "Downstream: reach ★ around an unseen obstacle (MPC over options vs actions)", fontsize=9
    )
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- the arm


def _relative(traj: torch.Tensor) -> torch.Tensor:
    """End-effector position relative to the target, from arm observations (..., 16)."""
    return (traj[..., 7:10] - traj[..., 10:13]).cpu()


def arm_rollouts_figure(rollouts: dict, names) -> plt.Figure:
    """Each option from the same starts: top view (around the target) and side view."""
    fig, axes = plt.subplots(2, len(names), figsize=(1.9 * len(names), 3.9), squeeze=False)
    for c, name in enumerate(names):
        rel = _relative(rollouts[name])
        radius = rel[..., :2].norm(dim=-1)
        top, side = axes[0, c], axes[1, c]
        for r in range(rel.shape[0]):
            top.plot(rel[r, :, 0], rel[r, :, 1], color=SERIES[c], lw=1.0, alpha=0.7)
            side.plot(radius[r], rel[r, :, 2], color=SERIES[c], lw=1.0, alpha=0.7)
            top.scatter(rel[r, 0, 0], rel[r, 0, 1], s=6, color=SERIES[c], lw=0)
            side.scatter(radius[r, 0], rel[r, 0, 2], s=6, color=SERIES[c], lw=0)
        top.scatter([0], [0], marker="*", s=70, color=INK, zorder=3)
        side.scatter([0], [0], marker="*", s=70, color=INK, zorder=3)
        top.set_xlim(-0.6, 0.6)
        top.set_ylim(-0.6, 0.6)
        side.set_xlim(0, 0.7)
        side.set_ylim(-0.5, 0.5)
        for ax in (top, side):
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        top.set_title(name)
    axes[0, 0].set_ylabel("top view")
    axes[1, 0].set_ylabel("side view\n(height vs radius)")
    fig.suptitle("Closed-loop end-effector paths of each option around the target ★", fontsize=9)
    fig.tight_layout()
    return fig


# Each composition with the 3-D view (elevation, azimuth) that shows its shape best.
ARM_COMPOSITIONS = {
    "toward+cw": (62, -60),  # an inward spiral, seen from above
    "cw+up": (16, -58),  # a helix around the target's axis
    "toward+up": (16, -58),  # a diagonal onto the axis
    "toward+away": (35, -60),  # cancellations: staying put
    "up+down": (35, -60),
}


def arm_compositions_figure(rollouts: dict) -> plt.Figure:
    """Sums of options in 3-D: an inward spiral, a helix, a diagonal, and cancellations.

    Paths are relative to the target (★); the grey line is the target's vertical axis.
    """
    keys = [k for k in ARM_COMPOSITIONS if k in rollouts]
    fig = plt.figure(figsize=(3.3 * len(keys), 3.6))
    for c, key in enumerate(keys):
        ax = fig.add_subplot(1, len(keys), c + 1, projection="3d")
        ax.view_init(*ARM_COMPOSITIONS[key])
        ax.plot([0, 0], [0, 0], [-0.45, 0.45], color=GRID, lw=1.2)
        rel = _relative(rollouts[key])
        for r in range(rel.shape[0]):
            ax.plot(rel[r, :, 0], rel[r, :, 1], rel[r, :, 2], color=SERIES[c], lw=1.0, alpha=0.85)
            ax.scatter(*rel[r, 0].tolist(), s=6, color=SERIES[c])
        ax.scatter([0], [0], [0], marker="*", s=70, color=INK)
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(-0.45, 0.45)
        ax.set_zlim(-0.45, 0.45)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(key.replace("+", " + "), fontsize=9)
    fig.suptitle("Zero-shot composition on the arm: z_i + z_j (paths relative to the target ★)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


BIN_FLOOR = 0.3  # jem.envs.panda.BIN_FLOOR (not imported: it needs the simulator)


def arm_downstream_figure(results: dict, kind: str, episodes: int = 4) -> plt.Figure:
    """Planner paths (columns) on the same episodes (rows) with the obstacle in grey.

    bin: side view along the start-target line (go up and over the rim);
    pillar: top view (go around).
    """
    names = list(results)
    fig, axes = plt.subplots(
        episodes, len(names), figsize=(1.9 * len(names), 1.9 * episodes), squeeze=False
    )
    task = results[names[0]]["task"]
    for c, name in enumerate(names):
        res = results[name]
        traj = res["traj"].cpu()
        for r in range(episodes):
            ax = axes[r, c]
            x, g = traj[r, :, 7:10], traj[r, 0, 10:13]
            mocap = task.mocap[r, 0].cpu()
            if kind == "bin":  # the square bin cut along the start-target line
                u = (traj[r, 0, 7:9] - g[:2]) / (traj[r, 0, 7:9] - g[:2]).norm()
                along = (x[:, :2] - g[:2]) @ u
                stretch = 1 / u.abs().max().item()
                inner, outer = 0.09 * stretch, 0.12 * stretch
                ax.add_patch(plt.Rectangle((-outer, 0), 2 * outer, BIN_FLOOR, color=GRID))
                for side in (-1, 1):
                    edge = min(side * inner, side * outer)
                    wall = plt.Rectangle((edge, BIN_FLOOR), outer - inner, 0.2, color=GRID)
                    ax.add_patch(wall)
                ax.plot(along, x[:, 2], color=SERIES[c], lw=1.2)
                ax.scatter(along[0], x[0, 2], s=10, color=INK2, zorder=3)
                ax.scatter([0], [g[2]], marker="*", s=70, color=INK, zorder=3)
                ax.set_xlim(-0.25, 0.4)
                ax.set_ylim(0.15, 0.8)
            else:
                ax.add_patch(plt.Circle(mocap[:2].tolist(), 0.05, color=GRID))
                ax.plot(x[:, 0], x[:, 1], color=SERIES[c], lw=1.2)
                ax.scatter(x[0, 0], x[0, 1], s=10, color=INK2, zorder=3)
                ax.scatter([g[0]], [g[1]], marker="*", s=70, color=INK, zorder=3)
                ax.set_xlim(g[0] - 0.45, g[0] + 0.45)
                ax.set_ylim(g[1] - 0.45, g[1] + 0.45)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{name}\nsuccess {100 * res['success']:.0f}%", fontsize=8)
    view = "side view" if kind == "bin" else "top view"
    fig.suptitle(f"Downstream on the arm ({kind}, {view}): MPC over options vs actions", fontsize=9)
    fig.tight_layout()
    return fig


def make_figures(model, world, result: dict) -> dict[str, plt.Figure]:
    art = result["artifacts"]
    names = list(world.expert_names)
    with torch.no_grad():
        figs = {
            "options": options_figure(art["z"], art["labels"], art["library"], names),
            "recognition": recognition_figure(art["recog"], art["recog_bayes"], len(names)),
        }
        if world.name == "panda":
            figs["rollouts"] = arm_rollouts_figure(art["rollouts"], names)
            figs["compositions"] = arm_compositions_figure(art["rollouts"])
        if world.name == "particle":
            figs["policies"] = policies_figure(model, world, art["library"], names)
            figs["compositions"] = compositions_figure(
                model, world, art["library"], names, art["adjacent"] + art["antipodal"]
            )
            figs["rollouts"] = rollouts_figure(art["rollouts"], names)
            if art["adjacent"]:
                figs["algebra"] = algebra_figure(
                    model, world, art["library"], names, art["adjacent"][0]
                )
        if art["track"]:
            figs["tracking"] = tracking_figure(art["track"], names)
    return figs
