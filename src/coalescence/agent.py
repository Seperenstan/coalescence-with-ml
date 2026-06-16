"""A small Deep Q-Network agent for the coalescence environments.

This is a single, deduplicated implementation that replaces the three
near-identical copies in the original thesis code. The network is a plain
two-hidden-layer MLP; the agent uses a target network, an epsilon-greedy policy
with exponential annealing, experience replay, and the Huber loss.
"""

from __future__ import annotations

import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


@dataclass
class AgentConfig:
    """Hyperparameters for :class:`DQNAgent` (defaults match the thesis runs)."""

    hidden_dim: int = 128
    batch_size: int = 32
    gamma: float = 0.99
    lr: float = 1e-3
    capacity: int = 10_000
    update_rate: int = 1200
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 6000.0
    grad_clip: float = 50.0


class ReplayMemory:
    """Fixed-capacity ring buffer of transitions, sampled uniformly at random."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self, *transition):
        self.memory.append(Transition(*transition))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Two-hidden-layer ReLU MLP mapping a state to per-action Q-values."""

    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    """Epsilon-greedy DQN agent with a target network and experience replay."""

    def __init__(self, env, config=None, device=None):
        self.config = config or AgentConfig()
        self.input_dim = env.state_dim
        self.action_dim = env.action_dim
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = DQN(self.input_dim, self.action_dim, self.config.hidden_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.action_dim, self.config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.config.lr, amsgrad=True)
        self.memory = ReplayMemory(self.config.capacity)

        self.steps_done = 0
        self.eval_mode = False
        self.last_action_was_random = False

    # -- acting ---------------------------------------------------------------

    @property
    def epsilon(self):
        c = self.config
        return c.eps_end + (c.eps_start - c.eps_end) * math.exp(-self.steps_done / c.eps_decay)

    def choose_action(self, state):
        """Pick an action greedily (eval) or epsilon-greedily (training)."""
        self.last_action_was_random = False
        if self.eval_mode:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).view(1, 1)

        eps = self.epsilon
        self.steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).view(1, 1)
        self.last_action_was_random = True
        action = np.random.randint(0, self.action_dim)
        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    # -- learning -------------------------------------------------------------

    def learn(self):
        """One gradient step on a replay batch; returns the loss (or ``None``)."""
        if len(self.memory) < self.config.batch_size:
            return None
        batch = Transition(*zip(*self.memory.sample(self.config.batch_size)))

        non_final = torch.tensor(
            [s is not None for s in batch.next_state], device=self.device, dtype=torch.bool
        )
        non_final_next = torch.cat([s for s in batch.next_state if s is not None])
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        q_values = self.policy_net(states).gather(1, actions)
        next_values = torch.zeros(self.config.batch_size, device=self.device)
        with torch.no_grad():
            next_values[non_final] = self.target_net(non_final_next).max(1)[0]
        expected = rewards + self.config.gamma * next_values

        loss = nn.SmoothL1Loss()(q_values, expected.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        """Copy the policy-network weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # -- persistence ----------------------------------------------------------

    def save(self, path):
        torch.save(self.target_net.state_dict(), path)

    def load(self, path):
        weights = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)
