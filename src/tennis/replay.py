"""Experience replay, sized for a CPU trainer.

The reference DRLND buffer is a ``deque`` of namedtuples that samples with
``random.sample`` and rebuilds tensors with ``torch.from_numpy(np.vstack([...]))``
per field, per sample. At 20 agents that runs roughly 500 times per episode and
the per-sample Python overhead becomes a visible fraction of wall-clock on a
machine with no GPU to hide behind.

These buffers preallocate one contiguous ``float32`` array per field and index
them with a single fancy-index gather, which is one C-level copy instead of
``batch_size`` Python-level ones. The interface is otherwise the familiar
``add`` / ``sample`` / ``__len__``.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size uniform-sampling ring buffer over flat vector transitions."""

    def __init__(
        self,
        capacity: int,
        state_size: int,
        action_size: int,
        batch_size: int,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ):
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)

        self.states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_size), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self._pos = 0
        self._full = False

    def add_batch(self, states, actions, rewards, next_states, dones) -> None:
        """Insert one transition per agent in a single call.

        Wrapping is handled by splitting the write at the end of the ring rather
        than looping per row, so a 20-agent step is at most two memcpys.
        """
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1, 1)

        n = states.shape[0]
        if n > self.capacity:
            raise ValueError(f"batch of {n} exceeds buffer capacity {self.capacity}")

        first = min(n, self.capacity - self._pos)
        second = n - first
        for dst, src in (
            (self.states, states),
            (self.actions, actions),
            (self.rewards, rewards),
            (self.next_states, next_states),
            (self.dones, dones),
        ):
            dst[self._pos:self._pos + first] = src[:first]
            if second:
                dst[:second] = src[first:]

        if second or self._pos + first == self.capacity:
            self._full = True
        self._pos = (self._pos + n) % self.capacity

    def sample(self):
        """Uniform sample; returns tensors on ``self.device``."""
        idx = self._rng.integers(0, len(self), size=self.batch_size)
        return self._gather(idx)

    def _gather(self, idx: np.ndarray):
        t = lambda a: torch.from_numpy(a[idx]).to(self.device)  # noqa: E731
        return (
            t(self.states),
            t(self.actions),
            t(self.rewards),
            t(self.next_states),
            t(self.dones),
        )

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritised replay (Schaul et al., 2015).

    Priorities live in a sum tree so that sampling and updating are both
    O(log n); a linear scan over a million-entry buffer would cost more than
    the network update it feeds.

    Note the interaction with a multi-agent step: 20 transitions arrive at once
    and all get max priority, so early on the sampler is effectively uniform
    over the newest experience. That is the intended behaviour -- unseen
    transitions should be tried once before being ranked.
    """

    def __init__(
        self,
        capacity: int,
        state_size: int,
        action_size: int,
        batch_size: int,
        seed: int = 0,
        device: torch.device | str = "cpu",
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-6,
    ):
        super().__init__(capacity, state_size, action_size, batch_size, seed, device)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.eps = float(eps)
        self._beta_step = 0

        # Sum tree over `capacity` leaves, padded to a power of two so the
        # parent/child index arithmetic stays branch-free.
        self._tree_size = 1
        while self._tree_size < self.capacity:
            self._tree_size *= 2
        self._tree = np.zeros(2 * self._tree_size, dtype=np.float64)
        self._max_priority = 1.0

    def _set_priority(self, leaf: int, priority: float) -> None:
        i = leaf + self._tree_size
        delta = priority - self._tree[i]
        self._tree[i] = priority
        i //= 2
        while i >= 1:
            self._tree[i] += delta
            i //= 2

    def add_batch(self, states, actions, rewards, next_states, dones) -> None:
        start = self._pos
        n = np.asarray(states).shape[0]
        super().add_batch(states, actions, rewards, next_states, dones)
        for k in range(n):
            self._set_priority((start + k) % self.capacity, self._max_priority ** self.alpha)

    def _find(self, value: float) -> int:
        """Descend the tree to the leaf whose interval contains ``value``."""
        i = 1
        while i < self._tree_size:
            left = 2 * i
            if value <= self._tree[left]:
                i = left
            else:
                value -= self._tree[left]
                i = left + 1
        return i - self._tree_size

    @property
    def beta(self) -> float:
        """Importance-sampling exponent, annealed 'beta_start' -> 1."""
        frac = min(1.0, self._beta_step / self.beta_frames)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def sample(self):
        n = len(self)
        total = self._tree[1]
        # Stratified sampling: one draw per equal-mass segment. Sampling all
        # `batch_size` draws from the whole range would over-represent a single
        # very-high-priority transition.
        segment = total / self.batch_size
        idx = np.empty(self.batch_size, dtype=np.int64)
        for k in range(self.batch_size):
            v = self._rng.uniform(segment * k, segment * (k + 1))
            leaf = self._find(v)
            idx[k] = min(leaf, n - 1)

        beta = self.beta
        self._beta_step += 1
        probs = self._tree[idx + self._tree_size] / total
        weights = (n * np.maximum(probs, 1e-12)) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        batch = self._gather(idx)
        return (*batch, torch.from_numpy(weights).unsqueeze(1).to(self.device), idx)

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        p = np.abs(np.asarray(td_errors, dtype=np.float64).ravel()) + self.eps
        self._max_priority = max(self._max_priority, float(p.max()))
        for leaf, prio in zip(idx, p):
            self._set_priority(int(leaf), prio ** self.alpha)
