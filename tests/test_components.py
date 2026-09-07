"""Unit tests for the shared components, checkable without launching Unity.

Carried over from the Reacher project along with the modules they cover
(replay buffers, n-step accumulation, noise processes, networks). The
multi-agent pieces are tested separately in test_multiagent.py.

Deep RL fails silently: a buffer that wraps wrong or an n-step return that
splices two agents' trajectories together still produces a training curve, just
a bad one, and the only symptom is an hour of wasted compute. These tests cover
the bookkeeping where that kind of bug hides.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tennis.agent import _NStepAccumulator
from tennis.networks import Actor, Critic, TwinCritic
from tennis.noise import make_noise
from tennis.replay import PrioritizedReplayBuffer, ReplayBuffer

# Tennis: 24 stacked observations per racket, 2 continuous actions, 2 agents.
STATE, ACTION, AGENTS = 24, 2, 2


def _transitions(n: int, state_size: int = STATE, action_size: int = ACTION, offset: float = 0.0):
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((n, state_size)).astype(np.float32) + offset,
        rng.uniform(-1, 1, (n, action_size)).astype(np.float32),
        rng.standard_normal(n).astype(np.float32),
        rng.standard_normal((n, state_size)).astype(np.float32),
        np.zeros(n, dtype=bool),
    )


class TestReplayBuffer:
    def test_length_and_wrap(self):
        buf = ReplayBuffer(50, STATE, ACTION, batch_size=8, seed=0)
        buf.add_batch(*_transitions(20))
        assert len(buf) == 20
        buf.add_batch(*_transitions(20))
        assert len(buf) == 40
        # This write straddles the end of the ring, exercising the split copy.
        buf.add_batch(*_transitions(20))
        assert len(buf) == 50

    def test_wrapped_write_preserves_rows(self):
        """A split write must not shear a transition across the wrap point."""
        buf = ReplayBuffer(10, 2, 1, batch_size=4, seed=0)
        # Fill so the next write wraps, then write rows with a recognisable
        # relationship between state and reward.
        buf.add_batch(np.zeros((8, 2), np.float32), np.zeros((8, 1), np.float32),
                      np.zeros(8, np.float32), np.zeros((8, 2), np.float32),
                      np.zeros(8, bool))
        states = np.arange(8, dtype=np.float32).reshape(4, 2)
        rewards = states[:, 0].copy()
        buf.add_batch(states, np.zeros((4, 1), np.float32), rewards,
                      states, np.zeros(4, bool))
        # Every stored row must still satisfy reward == state[0].
        stored = np.concatenate([buf.states[:, :1].ravel()[:, None],
                                 buf.rewards], axis=1)
        rows = stored[stored[:, 1] != 0]
        assert np.allclose(rows[:, 0], rows[:, 1])

    def test_sample_shapes_and_dtypes(self):
        buf = ReplayBuffer(100, STATE, ACTION, batch_size=16, seed=0)
        buf.add_batch(*_transitions(64))
        s, a, r, ns, d = buf.sample()
        assert s.shape == (16, STATE)
        assert a.shape == (16, ACTION)
        assert r.shape == (16, 1) and ns.shape == (16, STATE) and d.shape == (16, 1)
        assert all(t.dtype == torch.float32 for t in (s, a, r, ns, d))

    def test_capacity_is_a_hard_cap(self):
        buf = ReplayBuffer(30, STATE, ACTION, batch_size=4, seed=0)
        for _ in range(10):
            buf.add_batch(*_transitions(20))
        assert len(buf) == 30

    def test_oversized_batch_rejected(self):
        buf = ReplayBuffer(10, STATE, ACTION, batch_size=4, seed=0)
        with pytest.raises(ValueError):
            buf.add_batch(*_transitions(11))


class TestPrioritizedReplay:
    def test_sample_returns_weights_and_indices(self):
        buf = PrioritizedReplayBuffer(128, STATE, ACTION, batch_size=16, seed=0)
        buf.add_batch(*_transitions(64))
        s, a, r, ns, d, w, idx = buf.sample()
        assert w.shape == (16, 1)
        assert idx.shape == (16,)
        assert (w <= 1.0 + 1e-6).all() and (w > 0).all()

    def test_priority_update_biases_sampling(self):
        """A single high-priority transition should dominate the sample."""
        buf = PrioritizedReplayBuffer(64, 2, 1, batch_size=32, seed=0, alpha=1.0)
        buf.add_batch(*_transitions(64, state_size=2, action_size=1))
        idx = np.arange(64)
        errors = np.full(64, 0.01)
        errors[7] = 100.0
        buf.update_priorities(idx, errors)
        _, _, _, _, _, _, sampled = buf.sample()
        assert (sampled == 7).sum() >= 8

    def test_beta_anneals_to_one(self):
        buf = PrioritizedReplayBuffer(64, 2, 1, batch_size=4, seed=0,
                                      beta_start=0.4, beta_frames=10)
        buf.add_batch(*_transitions(64, state_size=2, action_size=1))
        assert buf.beta == pytest.approx(0.4)
        for _ in range(20):
            buf.sample()
        assert buf.beta == pytest.approx(1.0)


class TestNStep:
    def test_single_agent_discounted_return(self):
        acc = _NStepAccumulator(num_agents=1, n=3, gamma=0.5)
        s = np.arange(4, dtype=np.float32).reshape(4, 1)
        out = None
        for k in range(3):
            out = acc.push(s[k:k + 1], s[k:k + 1], [1.0], s[k + 1:k + 2], [False])
        assert out is not None
        # 1 + 0.5*1 + 0.25*1
        assert out[2][0] == pytest.approx(1.75)
        # The stored state is the *first* of the window, the next_state the last.
        assert out[0][0][0] == pytest.approx(0.0)
        assert out[3][0][0] == pytest.approx(3.0)

    def test_agents_do_not_share_windows(self):
        """Two agents at different rewards must not blend into one return."""
        acc = _NStepAccumulator(num_agents=2, n=2, gamma=1.0)
        s = np.zeros((2, 1), dtype=np.float32)
        acc.push(s, s, [1.0, 10.0], s, [False, False])
        out = acc.push(s, s, [1.0, 10.0], s, [False, False])
        assert sorted(out[2]) == pytest.approx([2.0, 20.0])

    def test_done_truncates_the_return(self):
        acc = _NStepAccumulator(num_agents=1, n=5, gamma=1.0)
        s = np.zeros((1, 1), dtype=np.float32)
        acc.push(s, s, [1.0], s, [False])
        out = acc.push(s, s, [1.0], s, [True])
        # Terminal at step 2: the flush emits windows, none longer than 2 steps.
        assert out is not None
        assert max(out[2]) == pytest.approx(2.0)
        assert out[4].any()


class TestNoise:
    def test_per_agent_state_is_independent(self):
        noise = make_noise("ou", (AGENTS, ACTION), seed=0)
        sample = noise.sample()
        assert sample.shape == (AGENTS, ACTION)
        # All 20 arms exploring identically would defeat the point of 20 arms.
        assert not np.allclose(sample[0], sample[1])

    def test_ou_is_zero_mean_over_time(self):
        noise = make_noise("ou", (1, 1), seed=1, sigma=0.2)
        draws = np.array([noise.sample()[0, 0] for _ in range(20000)])
        # A uniform-[0,1) innovation (the classic DRLND bug) gives a mean far
        # from zero and a one-sided torque bias.
        assert abs(draws.mean()) < 0.05

    def test_sigma_decays_per_reset(self):
        noise = make_noise("gaussian", (2, 2), seed=0, sigma=1.0,
                           sigma_min=0.1, sigma_decay=0.5)
        noise.reset()
        assert noise.sigma == pytest.approx(0.5)
        for _ in range(10):
            noise.reset()
        assert noise.sigma == pytest.approx(0.1)

    def test_none_is_silent(self):
        noise = make_noise("none", (3, 2), seed=0)
        assert np.allclose(noise.sample(), 0.0)


class TestNetworks:
    def test_actor_output_is_bounded(self):
        actor = Actor(STATE, ACTION)
        out = actor(torch.randn(64, STATE))
        assert out.shape == (64, ACTION)
        assert out.abs().max() <= 1.0

    def test_critic_consumes_state_and_action(self):
        critic = Critic(STATE, ACTION)
        q = critic(torch.randn(8, STATE), torch.rand(8, ACTION) * 2 - 1)
        assert q.shape == (8, 1)

    def test_twin_critics_are_independent(self):
        twin = TwinCritic(STATE, ACTION)
        q1, q2 = twin(torch.randn(8, STATE), torch.rand(8, ACTION) * 2 - 1)
        # Identical initialisation would make min(q1, q2) a no-op and silently
        # turn TD3 back into DDPG.
        assert not torch.allclose(q1, q2)

    def test_input_norm_reaches_the_raw_observation(self):
        """The badly-scaled input is what saturated the first baseline's actor,
        so input_norm must act *before* fc1, not on its output."""
        plain = Actor(STATE, ACTION, norm="none", input_norm="none")
        normed = Actor(STATE, ACTION, norm="none", input_norm="layer")
        assert isinstance(plain.n0, torch.nn.Identity)
        assert not isinstance(normed.n0, torch.nn.Identity)

        # Give the two identical weights, so any difference in output comes from
        # the input normalisation alone.
        normed.load_state_dict(plain.state_dict(), strict=False)
        # A wildly-scaled observation, as Reacher actually produces.
        state = torch.randn(8, STATE) * 28.0
        assert not torch.allclose(plain(state), normed(state), atol=1e-4)

    def test_input_norm_off_leaves_state_dict_unchanged(self):
        """nn.Identity adds no entries, so checkpoints written before input_norm
        existed still load."""
        before = set(Actor(STATE, ACTION, norm="none").state_dict())
        after = set(Actor(STATE, ACTION, norm="none", input_norm="none").state_dict())
        assert before == after

    def test_layer_norm_is_batch_size_invariant(self):
        """Unlike batch norm, layer norm gives the same answer for one row."""
        actor = Actor(STATE, ACTION, norm="layer").eval()
        batch = torch.randn(4, STATE)
        with torch.no_grad():
            full = actor(batch)
            single = actor(batch[:1])
        assert torch.allclose(full[0], single[0], atol=1e-6)


