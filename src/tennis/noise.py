"""Exploration noise for a deterministic policy.

A deterministic actor emits the same torque for the same state every time, so
all exploration has to be injected. Two processes, both ablatable:

* **Ornstein-Uhlenbeck** -- what the DDPG paper used. Temporally correlated, so
  the noise pushes the arm in a consistent direction for a while instead of
  jittering it in place. That matters for a physical system with momentum.
* **Gaussian** -- independent per step. Later work (TD3) found this works as
  well or better while having one fewer hyperparameter, and it is the honest
  baseline for asking whether OU's correlation actually earns its complexity.

Both hold one noise state *per agent*: with 20 arms sharing a policy, giving
them a shared noise vector would make all 20 explore in lockstep and collapse
the diversity that makes the 20-agent build worth using.
"""
from __future__ import annotations

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process, vectorised over agents."""

    def __init__(
        self,
        size: tuple[int, int],
        seed: int = 0,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        sigma_min: float = 0.0,
        sigma_decay: float = 1.0,
    ):
        self.size = size
        self.mu = mu * np.ones(size, dtype=np.float32)
        self.theta = float(theta)
        self.sigma_start = float(sigma)
        self.sigma = float(sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_decay = float(sigma_decay)
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Re-centre the process and decay sigma one episode's worth.

        Decaying here rather than per step ties the exploration schedule to
        episodes, which is the unit the score is measured in.
        """
        self.state = self.mu.copy()
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self) -> np.ndarray:
        # Standard normal, not the uniform [0,1) the original DRLND code used:
        # `random.random()` there gives a strictly positive draw, so the noise
        # has a mean offset and biases every torque in one direction.
        dx = self.theta * (self.mu - self.state) + self.sigma * self._rng.standard_normal(self.size)
        self.state = (self.state + dx).astype(np.float32)
        return self.state


class GaussianNoise:
    """Independent zero-mean Gaussian noise, with the same interface as OUNoise."""

    def __init__(
        self,
        size: tuple[int, int],
        seed: int = 0,
        sigma: float = 0.2,
        sigma_min: float = 0.0,
        sigma_decay: float = 1.0,
        **_ignored,
    ):
        self.size = size
        self.sigma_start = float(sigma)
        self.sigma = float(sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_decay = float(sigma_decay)
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self) -> np.ndarray:
        return (self.sigma * self._rng.standard_normal(self.size)).astype(np.float32)


def make_noise(kind: str, size: tuple[int, int], seed: int, **kwargs):
    if kind == "ou":
        return OUNoise(size, seed=seed, **kwargs)
    if kind == "gaussian":
        return GaussianNoise(size, seed=seed, **kwargs)
    if kind == "none":
        return GaussianNoise(size, seed=seed, sigma=0.0)
    raise ValueError(f"unknown noise kind: {kind!r} (expected ou|gaussian|none)")
