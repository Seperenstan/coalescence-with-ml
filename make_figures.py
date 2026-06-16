"""Regenerate the project figures from the saved results in ``results/``.

Every figure is built from the curated ``.txt`` data the thesis experiments
produced -- no training or evaluation is run here. The styling matches the
portfolio site (dark background, teal/violet/amber accents) so the figures sit
coherently on the project page.

    python make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# -- portfolio palette (src/index.css tokens) --------------------------------
BG = "#0f1420"
TEXT = "#e6edf6"
MUTED = "#9aa7b8"
FAINT = "#64748b"
GRID = "#1c2433"
TEAL = "#2dd4bf"
VIOLET = "#a78bfa"
AMBER = "#fbbf24"
RAIL = "#2a3344"
EDGE = "#243042"

AGENT_COLORS = {
    "Choose action": TEAL,
    "Choose particle": VIOLET,
    "Choose action + particle": AMBER,
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": TEXT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "font.size": 12,
    "figure.dpi": 130,
})


def _style(ax):
    """Trim the top/right spines and soften the rest."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(length=0)


def load(*parts):
    return np.loadtxt(RESULTS.joinpath(*parts))


def smooth(data, window=5):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def theory(n):
    t = np.arange(1, n + 1)
    return t, 1.0 / np.sqrt(np.pi * t)


# -- density-vs-time control figures -----------------------------------------

VARIANTS = {  # label -> results/performance filename stem
    "Choose action": "action",
    "Choose particle": "1L",
    "Choose action + particle": "2L",
}


def _plot_density(ax, mode, title):
    for label, stem in VARIANTS.items():
        data = load("performance", f"{mode}_{stem}.txt")
        t = np.arange(1, len(data) + 1)
        ax.plot(t, data, color=AGENT_COLORS[label], lw=2.2, label=label)

    rnd = load("performance", "random_agent.txt")
    t = np.arange(1, len(rnd) + 1)
    ax.plot(t, rnd, color=FAINT, lw=1.8, label="Random agent")

    tt, th = theory(len(rnd))
    ax.plot(tt, th, color=MUTED, lw=1.5, ls="--", label=r"$1/\sqrt{\pi t}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time  $t$  (sweeps)")
    ax.set_ylabel(r"$N(t)/N_0$")
    ax.set_title(title, fontsize=13, pad=10)
    _style(ax)


def fig_control():
    """Two panels: agents suppress (left) and facilitate (right) coalescence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    _plot_density(ax1, "suppressing", "Suppressing — stay apart, decay slower")
    _plot_density(ax2, "facilitating", "Facilitating — merge fast, decay faster")
    ax1.legend(loc="lower left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "control.png", bbox_inches="tight")
    plt.close(fig)


def fig_cover():
    """Schematic cover: a 1D lattice with hop directions, the observation window
    around one particle, and the policy network that turns it into an action."""
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 67.5)
    ax.set_aspect("equal")
    ax.axis("off")

    def mono(x, y, s, size, color, ha="left", weight="normal"):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
                family="monospace", weight=weight)

    # title
    mono(8, 62, "COALESCING RANDOM WALKS  ·  DEEP Q-LEARNING", 12.5, TEAL)
    ax.text(8, 56.5, "Coalescence with ML", fontsize=31, color=TEXT, weight="bold", va="center")
    mono(8, 51, "an agent learns where each particle should hop", 13, MUTED)

    # lattice
    n, x0, dx, yL = 10, 10, 6.4, 24
    sites = [x0 + i * dx for i in range(n)]
    ax.plot([sites[0] - 2, sites[-1] + 2], [yL, yL], color=RAIL, lw=3, zorder=1, solid_capstyle="round")
    for sx in sites:
        ax.plot([sx, sx], [yL - 1.1, yL + 1.1], color=EDGE, lw=2, zorder=1)
    # dotted continuation: the lattice runs on (periodic) past both ends
    for side in (-1, 1):
        edge = sites[0] - 2 if side < 0 else sites[-1] + 2
        ax.plot([edge + side * k for k in (1.3, 2.4, 3.5)], [yL] * 3, marker="s",
                ms=2.2, ls="none", color=RAIL, zorder=1)
    occ = {1: TEAL, 3: VIOLET, 4: TEAL, 6: VIOLET, 8: TEAL}
    for i, c in occ.items():
        ax.add_patch(Circle((sites[i], yL), 2.5, facecolor=c, alpha=0.18, edgecolor="none", zorder=3))
        ax.add_patch(Circle((sites[i], yL), 1.55, facecolor=c, edgecolor="none", zorder=4))

    def hop(sx, direction, color):
        ax.add_patch(FancyArrowPatch((sx, yL + 2.2), (sx + direction * 4.2, yL + 2.2),
                                     arrowstyle="-|>", mutation_scale=12, lw=2, color=color,
                                     connectionstyle="arc3,rad=-0.45", zorder=5))

    hop(sites[1], +1, MUTED)
    hop(sites[8], -1, MUTED)
    hop(sites[4], -1, TEAL)
    hop(sites[4], +1, TEAL)

    # observation window (radius 2 around the centre particle)
    lo, hi, cx = 2, 6, sites[4]
    wx0, wx1 = sites[lo] - dx / 2, sites[hi] + dx / 2
    ax.add_patch(FancyBboxPatch((wx0, yL - 4.2), wx1 - wx0, 8.4,
                                boxstyle="round,pad=0.1,rounding_size=1.2",
                                facecolor=TEAL, alpha=0.10, edgecolor=TEAL, lw=2,
                                linestyle=(0, (5, 3)), zorder=2))
    mono(cx, 16.5, "observation radius  r = 2", 12, TEAL, ha="center")

    # policy network
    ix, hx, ox = 78, 92, 104
    iy = hy = [16, 22, 28, 34, 40]
    oy = [25, 31]
    for yy in iy:
        for hyy in hy:
            ax.plot([ix, hx], [yy, hyy], color=EDGE, lw=0.7, alpha=0.7, zorder=1)
    for hyy in hy:
        for oyy in oy:
            ax.plot([hx, ox], [hyy, oyy], color=EDGE, lw=0.7, alpha=0.7, zorder=1)
    for yy in iy:
        ax.add_patch(Circle((ix, yy), 1.25, facecolor="#0b0f18", edgecolor=TEAL, lw=1.6, zorder=4))
    for yy in hy:
        ax.add_patch(Circle((hx, yy), 1.25, facecolor="#0b0f18", edgecolor=VIOLET, lw=1.6, zorder=4))
    for yy in oy:
        ax.add_patch(Circle((ox, yy), 1.5, facecolor=VIOLET, edgecolor=TEAL, lw=1.6, zorder=4))
    mono(hx, 45, "policy network", 11.5, MUTED, ha="center")

    # window cells fan into the input layer
    for cxw, yy in zip([sites[i] for i in range(lo, hi + 1)], iy):
        ax.add_patch(FancyArrowPatch((cxw, yL - 4.6), (ix - 1.6, yy), arrowstyle="-",
                                     lw=1.1, color=TEAL, alpha=0.45,
                                     connectionstyle="arc3,rad=-0.18", zorder=1))

    # outputs -> action
    mono(ox + 3.2, 31, "←", 18, TEXT)
    mono(ox + 3.2, 25, "→", 18, TEXT)
    mono(ox + 2.6, 19.5, "action", 11.5, MUTED)

    fig.savefig(FIGURES / "cover.png", facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# -- training curves ----------------------------------------------------------

def fig_training():
    """Particles remaining per training episode for the three formulations."""
    series = {  # label -> training filename
        "Choose action": "particles_per_training_suppressing_action_5d.txt",
        "Choose particle": "particles_per_training_suppressing_1L_15d.txt",
        "Choose action + particle": "particles_per_training_suppressing_2L_15d.txt",
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for label, fname in series.items():
        data = smooth(load("training", fname)[:100] / 15.0)
        ax.plot(np.arange(1, len(data) + 1), data, color=AGENT_COLORS[label], lw=2.2, label=label)
    ax.set_xlabel("training episode")
    ax.set_ylabel(r"$N/N_0$ at episode end")
    ax.set_title("Particles remaining per training episode", fontsize=13, pad=10)
    ax.legend(loc="center left", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "training.png", bbox_inches="tight")
    plt.close(fig)


# -- hyperparameter sweeps ----------------------------------------------------

def fig_hyperparameters():
    """Reward-per-episode learning curves across DQN hyperparameter sweeps."""
    sweeps = [
        ("lr", [0.0001, 0.001, 0.01, 0.1], "learning rate"),
        ("discount", [0, 0.8, 0.9, 0.99], r"discount $\gamma$"),
        ("batch_size", [4, 16, 32, 64, 128], "batch size"),
        ("capacity", [100, 1000, 10000, 100000, 1000000], "replay capacity"),
        ("hidden_layer_size", [4, 16, 32, 128, 256], "hidden width"),
        ("training_environment_size", [5, 11, 20, 40, 100], "lattice size $L$"),
    ]
    palette = [TEAL, VIOLET, AMBER, "#f472b6", "#38bdf8"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for ax, (param, values, title) in zip(axes.ravel(), sweeps):
        for color, value in zip(palette, values):
            try:
                data = smooth(load("hyperparameters", f"reward_per_training_{param}_{value}.txt")[:100])
            except OSError:
                continue
            ax.plot(np.arange(1, len(data) + 1), data, color=color, lw=1.8, label=str(value))
        ax.set_title(title, fontsize=12, pad=6)
        ax.legend(fontsize=8, loc="lower right")
        _style(ax)
    fig.supxlabel("training episode", color=MUTED, fontsize=12)
    fig.supylabel("accumulated reward", color=MUTED, fontsize=12)
    fig.suptitle("DQN hyperparameter study", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "hyperparameters.png", bbox_inches="tight")
    plt.close(fig)


def fig_observation_radius():
    """Density vs. time for local-action agents with different observation radii.

    Evaluation rollouts of the trained agents on a large lattice: a wider window
    lets the agent hold more particles apart, well above the 1/sqrt(pi t) baseline.
    """
    radii = [(1, TEAL), (3, VIOLET), (5, AMBER)]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for r, color in radii:
        data = load("performance", f"obs_radius_r{r}.txt")
        ax.plot(np.arange(1, len(data) + 1), data, color=color, lw=2.4, label=f"$r_o = {r}$")

    rnd = load("performance", "obs_radius_random.txt")
    ax.plot(np.arange(1, len(rnd) + 1), rnd, color=FAINT, lw=1.8, label="Random agent")
    tt, th = theory(len(rnd))
    ax.plot(tt, th, color=MUTED, lw=1.5, ls="--", label=r"$1/\sqrt{\pi t}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time  $t$  (sweeps)")
    ax.set_ylabel(r"$N(t)/N_0$")
    ax.set_title("The more it sees, the more it suppresses", fontsize=13, pad=10)
    ax.legend(loc="lower left", fontsize=9.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "observation-radius.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    for fn in (fig_cover, fig_control, fig_training, fig_hyperparameters, fig_observation_radius):
        fn()
        print(f"wrote figures/{fn.__name__.replace('fig_', '').replace('_', '-')}.png")


if __name__ == "__main__":
    main()
