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
from matplotlib import font_manager

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
    """Single suppression panel — the headline result, used as the cover."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _plot_density(ax, "suppressing", "Learning to suppress coalescence")
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "cover.png", bbox_inches="tight")
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
    """How the local-action agent learns as its observation window grows."""
    radii = [1, 2, 3, 5, 10]
    palette = [TEAL, "#38bdf8", VIOLET, AMBER, FAINT]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for color, r in zip(palette, radii):
        try:
            data = smooth(load("hyperparameters", f"reward_per_training_observation_radius_{r}.txt")[:100])
        except OSError:
            continue
        ax.plot(np.arange(1, len(data) + 1), data, color=color, lw=2.0, label=f"$r_o = {r}$")
    ax.set_xlabel("training episode")
    ax.set_ylabel("accumulated reward")
    ax.set_title("Wider observation, faster learning", fontsize=13, pad=10)
    ax.legend(loc="lower right", fontsize=10, ncol=2)
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
