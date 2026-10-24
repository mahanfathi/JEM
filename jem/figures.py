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
from jem.evaluate import option_logits  # noqa: E402

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
    ax.plot(ns, [bayes[n] for n in ns], color=MUTED, lw=1.2, marker="o", ms=3)
    ax.plot(ns, [recog[n] for n in ns], color=SERIES[0], marker="o", ms=5)
    ax.annotate(
        "model", (ns[0], recog[ns[0]]), xytext=(4, -12), textcoords="offset points", color=INK2
    )
    ax.annotate(
        "Bayes (true densities)",
        (ns[0], bayes[ns[0]]),
        xytext=(4, 6),
        textcoords="offset points",
        color=MUTED,
    )
    ax.axhline(1 / num_experts, color=MUTED, lw=1, ls="--")
    ax.annotate(
        "chance",
        (ns[-1], 1 / num_experts),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=MUTED,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xlabel("context length N (steps)")
    ax.set_ylabel("accuracy")
    ax.set_title("Option recognition by energy: argmax_k Σ_t log p(a_t|s_t,z_k)")
    return fig


def make_figures(model, world: ParticleWorld, result: dict) -> dict[str, plt.Figure]:
    art = result["artifacts"]
    names = list(world.expert_names)
    pairs = art["adjacent"] + art["antipodal"]
    with torch.no_grad():
        return {
            "options": options_figure(art["z"], art["labels"], art["library"], names),
            "policies": policies_figure(model, world, art["library"], names),
            "compositions": compositions_figure(model, world, art["library"], names, pairs),
            "rollouts": rollouts_figure(art["rollouts"], names),
            "recognition": recognition_figure(art["recog"], art["recog_bayes"], world.num_experts),
        }
