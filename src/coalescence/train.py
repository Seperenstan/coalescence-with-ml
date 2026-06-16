"""Training loop for a DQN agent on a coalescence environment.

A single, config-driven loop that replaces the three near-identical ``main.py``
scripts in the original thesis code. Run it from the command line:

    python -m coalescence.train --config configs/suppress_local_action.yaml

or call :func:`train` directly from a notebook.
"""

from __future__ import annotations

import argparse
from itertools import count

import numpy as np
import torch

from .agent import DQNAgent
from .config import ExperimentConfig
from .envs import make_env


def _to_tensor(state, device):
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


def train(config, *, verbose=True):
    """Train an agent from an :class:`ExperimentConfig`.

    Returns ``(agent, history)`` where ``history`` holds the per-episode score
    and the fraction of particles remaining at the end of each episode.
    """
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    env = make_env(
        config.variant,
        config.length,
        config.n_particles,
        facilitating=config.facilitating,
        obs_radius=config.obs_radius,
        rng=config.seed,
    )
    agent = DQNAgent(env, config.agent)

    history = {"score": [], "fraction_remaining": []}
    for episode in range(config.training_episodes):
        state = _to_tensor(env.reset(), agent.device)
        score = 0.0
        for _ in count():
            action = agent.choose_action(state)
            observation, reward = env.step(action.item())
            reward_t = torch.tensor([reward], dtype=torch.float32, device=agent.device)

            terminated = env.done
            truncated = env.steps > config.max_steps
            next_state = None if terminated else _to_tensor(observation, agent.device)

            agent.memory.add(state, action, next_state, reward_t)
            agent.learn()
            state = _to_tensor(observation, agent.device)
            score += reward

            if terminated or truncated:
                agent.sync_target()
                break

        fraction = env.particles / env.n_particles
        history["score"].append(score)
        history["fraction_remaining"].append(fraction)
        if verbose:
            print(
                f"episode {episode:3d}  score {score:8.1f}  "
                f"remaining {fraction:5.1%}  eps {agent.epsilon:.3f}"
            )

    return agent, history


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a DQN coalescence-control agent.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--out", help="Optional path to save the trained network (.pth).")
    args = parser.parse_args(argv)

    config = ExperimentConfig.load(args.config)
    agent, _ = train(config)
    if args.out:
        agent.save(args.out)
        print(f"saved trained network to {args.out}")


if __name__ == "__main__":
    main()
