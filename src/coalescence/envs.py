"""Coalescing-random-walk environments on a 1D periodic lattice.

A lattice of ``length`` sites holds ``n_particles`` identical particles. When a
particle steps onto an occupied site the two coalesce into one (A + A -> A), so
the particle count only ever decreases. Left to diffuse freely the density
decays as the classic ``N(t)/N0 ~ 1/sqrt(pi * t)`` law.

A reinforcement-learning agent controls the walk to either *suppress* (keep
particles apart, decay slower than the baseline) or *facilitate* (drive them
together, decay faster). The three subclasses below differ only in **what the
agent observes** and **what it acts on** -- the lattice mechanics are shared:

  - :class:`LocalActionEnv`        observe a window around one particle; choose
                                   its direction (2 actions).
  - :class:`GlobalParticleEnv`     observe the whole lattice; choose which
                                   particle moves (it then steps randomly).
  - :class:`GlobalParticleActionEnv` observe the whole lattice; choose a
                                   particle *and* its direction (2*L actions).

The reward magnitudes are the exact values used for the thesis experiments and
are kept as class constants so each formulation stays reproducible.
"""

from __future__ import annotations

import numpy as np


class CoalescenceEnv:
    """Shared mechanics for coalescing random walks on a periodic lattice.

    Subclasses implement :meth:`observe`, :meth:`step`, and the ``state_dim`` /
    ``action_dim`` properties; everything here is common to all three.
    """

    def __init__(self, length, n_particles, *, facilitating=False, rng=None):
        self.length = length
        self.n_particles = n_particles
        self.facilitating = facilitating
        self.rng = np.random.default_rng(rng)
        self.reset()

    # -- lattice bookkeeping --------------------------------------------------

    def reset(self):
        """Scatter the particles at random and return the first observation."""
        self.lattice = np.zeros(self.length, dtype=int)
        occupied = self.rng.choice(self.length, size=self.n_particles, replace=False)
        self.lattice[occupied] = 1
        self.particles = self.n_particles
        self.steps = 0
        self.merges = 0
        self.just_merged = False
        return self.observe()

    @property
    def done(self):
        """A single particle is left; the walk has fully coalesced."""
        return self.particles <= 1

    def _settle(self, target):
        """Place the moving particle on ``target``, coalescing if occupied.

        Returns ``True`` if a coalescence happened.
        """
        merged = self.lattice[target] == 1
        self.lattice[target] = 1
        self.particles = int(np.count_nonzero(self.lattice))
        self.just_merged = merged
        if merged:
            self.merges += 1
        return merged

    # -- subclass API ---------------------------------------------------------

    @property
    def state_dim(self):
        raise NotImplementedError

    @property
    def action_dim(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def step(self, action):
        """Apply ``action``; return ``(next_observation, reward)``."""
        raise NotImplementedError


class LocalActionEnv(CoalescenceEnv):
    """Observe a window of radius ``obs_radius`` around one random particle and
    choose that particle's step direction (0 = left, 1 = right).

    Each step a fresh random particle is presented, so the policy is a local
    rule keyed only on the neighbourhood it can see.
    """

    # (merge, move) rewards for each objective.
    SUPPRESS = (-100, 5)
    FACILITATE = (50, -5)

    def __init__(self, length, n_particles, *, obs_radius=2, facilitating=False, rng=None):
        self.obs_radius = obs_radius
        self.current = 0
        super().__init__(length, n_particles, facilitating=facilitating, rng=rng)

    @property
    def state_dim(self):
        return 2 * self.obs_radius + 1

    @property
    def action_dim(self):
        return 2

    def _pick_particle(self):
        self.current = int(self.rng.choice(np.flatnonzero(self.lattice == 1)))
        return self.current

    def observe(self):
        """Occupancy window centred on the current particle (periodic)."""
        offsets = np.arange(-self.obs_radius, self.obs_radius + 1)
        sites = (self.current + offsets) % self.length
        return self.lattice[sites].astype(float)

    def reset(self):
        out = super().reset()
        self._pick_particle()
        return self.observe()

    def step(self, action):
        self.steps += 1
        self.lattice[self.current] = 0
        direction = -1 if action == 0 else 1
        self.current = (self.current + direction) % self.length

        merged = self._settle(self.current)
        merge_reward, move_reward = self.FACILITATE if self.facilitating else self.SUPPRESS
        reward = merge_reward if merged else move_reward

        self._pick_particle()
        return self.observe(), reward


class GlobalParticleEnv(CoalescenceEnv):
    """Observe the whole lattice and choose *which* particle to move; the chosen
    particle then steps in a random direction.

    Selecting an empty site is an invalid move: it is penalised and the lattice
    is left unchanged.
    """

    EMPTY_PENALTY = -60
    SUPPRESS = (-50, 5)
    FACILITATE = (50, -5)

    @property
    def state_dim(self):
        return self.length

    @property
    def action_dim(self):
        return self.length

    def observe(self):
        return self.lattice.astype(float)

    def step(self, action):
        self.steps += 1
        if self.lattice[action] == 0:
            self.just_merged = False
            return self.observe(), self.EMPTY_PENALTY

        direction = -1 if self.rng.random() < 0.5 else 1
        target = (action + direction) % self.length
        self.lattice[action] = 0
        merged = self._settle(target)

        merge_reward, move_reward = self.FACILITATE if self.facilitating else self.SUPPRESS
        reward = merge_reward if merged else move_reward
        return self.observe(), reward


class GlobalParticleActionEnv(CoalescenceEnv):
    """Observe the whole lattice and choose a particle *and* its direction.

    The action encodes both: ``particle = action // 2`` and a left/right step
    from ``action % 2``. Selecting an empty site is penalised as an invalid move.
    """

    EMPTY_PENALTY = -60
    SUPPRESS = (-50, 5)
    FACILITATE = (100, -40)

    @property
    def state_dim(self):
        return self.length

    @property
    def action_dim(self):
        return 2 * self.length

    def observe(self):
        return self.lattice.astype(float)

    def step(self, action):
        self.steps += 1
        particle = action // 2
        if self.lattice[particle] == 0:
            self.just_merged = False
            return self.observe(), self.EMPTY_PENALTY

        direction = -1 if action % 2 == 0 else 1
        target = (particle + direction) % self.length
        self.lattice[particle] = 0
        merged = self._settle(target)

        merge_reward, move_reward = self.FACILITATE if self.facilitating else self.SUPPRESS
        reward = merge_reward if merged else move_reward
        return self.observe(), reward


#: Maps a config-friendly name to the environment class.
ENVIRONMENTS = {
    "local_action": LocalActionEnv,
    "global_particle": GlobalParticleEnv,
    "global_particle_action": GlobalParticleActionEnv,
}


def make_env(variant, length, n_particles, *, facilitating=False, obs_radius=2, rng=None):
    """Construct an environment by ``variant`` name (see :data:`ENVIRONMENTS`)."""
    cls = ENVIRONMENTS[variant]
    kwargs = dict(facilitating=facilitating, rng=rng)
    if cls is LocalActionEnv:
        kwargs["obs_radius"] = obs_radius
    return cls(length, n_particles, **kwargs)
