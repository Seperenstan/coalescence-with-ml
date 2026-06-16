"""Evaluation utilities: measure how fast coalescence happens under a policy.

The headline quantity is the particle density ``N(t)/N0`` as a function of
time, compared against a random policy and the diffusion-limited asymptote
``1/sqrt(pi * t)``. Time is measured in *sweeps* of ``N`` moves (one Monte Carlo
sweep), matching the thesis convention.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_tensor(state, device):
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


def theoretical_asymptote(n_sweeps):
    """The ``1/sqrt(pi * t)`` decay of free coalescing random walks."""
    t = np.arange(1, n_sweeps + 1)
    return 1.0 / np.sqrt(np.pi * t)


def rollout_density(agent, env, n_sweeps):
    """Density ``N(t)/N0`` over ``n_sweeps`` sweeps for a trained ``agent``."""
    was_eval = agent.eval_mode
    agent.eval_mode = True
    env.reset()
    state = _to_tensor(env.observe(), agent.device)
    densities = np.empty(n_sweeps)
    for i in range(n_sweeps):
        densities[i] = env.particles / env.n_particles
        for _ in range(env.particles):
            action = agent.choose_action(state)
            observation, _ = env.step(action.item())
            state = _to_tensor(observation, agent.device)
    agent.eval_mode = was_eval
    return densities


def random_density(env, n_sweeps):
    """Density under a uniform-random policy on the same environment."""
    env.reset()
    densities = np.empty(n_sweeps)
    for i in range(n_sweeps):
        densities[i] = env.particles / env.n_particles
        for _ in range(env.particles):
            env.step(np.random.randint(0, env.action_dim))
    return densities


def average_density(rollout_fn, n_runs):
    """Average a density rollout over ``n_runs`` independent episodes.

    ``rollout_fn`` is a zero-argument callable returning one density array.
    """
    runs = np.vstack([rollout_fn() for _ in range(n_runs)])
    return runs.mean(axis=0)


def coalescence_time(agent, env, max_sweeps=10_000):
    """Number of sweeps until a single particle remains (capped at ``max_sweeps``)."""
    was_eval = agent.eval_mode
    agent.eval_mode = True
    env.reset()
    state = _to_tensor(env.observe(), agent.device)
    for sweep in range(1, max_sweeps + 1):
        if env.done:
            agent.eval_mode = was_eval
            return sweep
        for _ in range(env.particles):
            action = agent.choose_action(state)
            observation, _ = env.step(action.item())
            state = _to_tensor(observation, agent.device)
    agent.eval_mode = was_eval
    return max_sweeps
