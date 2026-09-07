"""Actor and critic networks for continuous control.

Both follow the DDPG paper's architecture choices (Lillicrap et al., 2015,
section 7 "Experiment Details"):

* hidden layers initialised uniformly from ``+/- 1/sqrt(fan_in)``, which keeps
  the initial output scale independent of layer width;
* the final layer initialised from a deliberately tiny ``+/- 3e-3``, so the
  policy starts near the centre of the action range and the critic starts near
  zero rather than emitting large arbitrary values that the target network then
  has to unlearn;
* the critic takes the action at the *second* layer, not the first, so the
  first layer learns a pure state representation.

The one place we allow deviation is normalisation. The paper uses batch norm;
it helps, but it makes the network behave differently at batch size 1 (acting)
than at batch size 128 (learning), which is a real source of silent bugs. Layer
norm is available as a drop-in that behaves identically either way. The default
is ``none`` so the baseline matches the project's benchmark implementation, and
the alternatives exist to be ablated.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer: nn.Linear) -> tuple[float, float]:
    """Uniform range ``+/- 1/sqrt(fan_in)`` for a hidden layer."""
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


def _norm(kind: str, size: int) -> nn.Module:
    """Normalisation layer by name; ``none`` yields a no-op so forward() stays flat."""
    if kind == "batch":
        return nn.BatchNorm1d(size)
    if kind == "layer":
        return nn.LayerNorm(size)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind!r} (expected none|batch|layer)")


class Actor(nn.Module):
    """Deterministic policy: state -> action in [-1, 1]^action_size."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden: tuple[int, ...] = (256, 128),
        norm: str = "none",
        input_norm: str = "none",
        final_init: float = 3e-3,
    ):
        super().__init__()
        h1, h2 = hidden
        # Normalising the raw observation is a separate decision from
        # normalising the hidden layers, and on this task the more important
        # one: the 33 inputs are raw physics quantities spanning |max| 0 to 28.
        self.n0 = _norm(input_norm, state_size)
        self.fc1 = nn.Linear(state_size, h1)
        self.n1 = _norm(norm, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.n2 = _norm(norm, h2)
        self.fc3 = nn.Linear(h2, action_size)
        self.reset_parameters(final_init)

    def reset_parameters(self, final_init: float = 3e-3) -> None:
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-final_init, final_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.n1(self.fc1(self.n0(state))))
        x = F.relu(self.n2(self.fc2(x)))
        # tanh bounds the output to the torque range Unity accepts; the env
        # clips too, but an unbounded policy would push gradients into a region
        # where every action saturates to the same clipped torque.
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Action-value function: (state, action) -> Q."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden: tuple[int, ...] = (256, 128),
        norm: str = "none",
        input_norm: str = "none",
        final_init: float = 3e-3,
    ):
        super().__init__()
        h1, h2 = hidden
        self.n0 = _norm(input_norm, state_size)
        self.fc1 = nn.Linear(state_size, h1)
        self.n1 = _norm(norm, h1)
        # Actions enter here, after the state has its own representation.
        self.fc2 = nn.Linear(h1 + action_size, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.reset_parameters(final_init)

    def reset_parameters(self, final_init: float = 3e-3) -> None:
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-final_init, final_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xs = F.relu(self.n1(self.fc1(self.n0(state))))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TwinCritic(nn.Module):
    """Two independent critics sharing an interface, for TD3-style min-of-two.

    Kept in one module so the twin pair has a single optimiser and a single
    ``soft_update`` target, which is what makes TD3 a small delta on DDPG here
    rather than a parallel implementation.
    """

    def __init__(self, state_size: int, action_size: int, **kwargs):
        super().__init__()
        self.q1 = Critic(state_size, action_size, **kwargs)
        self.q2 = Critic(state_size, action_size, **kwargs)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.q1(state, action), self.q2(state, action)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """The actor is trained against a single critic; using the min would
        bias the policy gradient toward whichever critic is pessimistic."""
        return self.q1(state, action)
