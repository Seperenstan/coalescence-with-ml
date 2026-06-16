"""Experiment configuration: one dataclass loadable from a YAML file.

A config fully describes a run -- which control formulation, whether to suppress
or facilitate coalescence, the lattice, and the agent hyperparameters -- so the
experiments in ``configs/`` are self-documenting and reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from .agent import AgentConfig


@dataclass
class ExperimentConfig:
    variant: str  # one of envs.ENVIRONMENTS: local_action | global_particle | global_particle_action
    facilitating: bool = False  # False -> suppress coalescence, True -> facilitate it
    length: int = 15  # lattice sites L
    n_particles: int = 15  # initial particles N
    obs_radius: int = 2  # observation window (local_action variant only)
    training_episodes: int = 300
    max_steps: int = 1000  # truncate an episode after this many moves
    seed: int | None = None
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        agent = AgentConfig(**data.pop("agent", {}) or {})
        return cls(agent=agent, **data)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
