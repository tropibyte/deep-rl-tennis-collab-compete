"""Tests for the parts of Tennis that Reacher's test suite does not cover.

Two of these guard against failures that would produce a plausible-looking
training curve and a silently wrong agent:

* **The agent-centric view.** If the joint state were not reordered per agent,
  the shared critic would be asked to map one input to two different rewards.
  It would fit neither, and the only symptom would be slow learning.
* **The scoring rule.** Tennis scores an episode by the *maximum* of the two
  agents' totals. Using the mean roughly halves every score, so a solved agent
  would never register as solved.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tennis.agent import Agent, agent_centric_views
from tennis.config import Config
from tennis.train import episode_score

OBS, ACT, AGENTS = 24, 2, 2


def _cfg(**overrides) -> Config:
    kwargs = dict(batch_size=8, learn_start=16, update_every=1,
                  updates_per_cycle=1, buffer_size=512)
    kwargs.update(overrides)
    return Config(**kwargs)


def _step_batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((AGENTS, OBS)).astype(np.float32),
        rng.uniform(-1, 1, (AGENTS, ACT)).astype(np.float32),
        rng.standard_normal(AGENTS).astype(np.float32),
        rng.standard_normal((AGENTS, OBS)).astype(np.float32),
        np.zeros(AGENTS, dtype=bool),
    )


class TestAgentCentricViews:
    def test_two_agents_are_swapped(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        out = agent_centric_views(a)
        assert out.shape == (2, 4)
        # Row 0 is agent 0 first; row 1 is agent 1 first.
        assert np.allclose(out[0], [1, 2, 3, 4])
        assert np.allclose(out[1], [3, 4, 1, 2])

    def test_own_observation_always_leads(self):
        """The critic and the actor both rely on the acting agent leading."""
        rng = np.random.default_rng(0)
        obs = rng.standard_normal((AGENTS, OBS)).astype(np.float32)
        out = agent_centric_views(obs)
        for i in range(AGENTS):
            assert np.allclose(out[i][:OBS], obs[i]), f"agent {i} not leading its own row"

    def test_is_a_permutation_not_a_copy(self):
        """Every row must contain all agents' data exactly once."""
        obs = np.arange(AGENTS * OBS, dtype=np.float32).reshape(AGENTS, OBS)
        out = agent_centric_views(obs)
        for row in out:
            assert sorted(row.tolist()) == sorted(obs.ravel().tolist())


class TestScoring:
    def test_max_is_the_rubric_rule(self):
        assert episode_score(np.array([0.1, 0.7]), "max") == pytest.approx(0.7)

    def test_mean_would_understate(self):
        """Recorded because it is the failure mode, not because it is used."""
        totals = np.array([0.0, 0.6])
        assert episode_score(totals, "mean") == pytest.approx(0.3)
        assert episode_score(totals, "max") == pytest.approx(0.6)

    def test_unknown_reduction_rejected(self):
        with pytest.raises(ValueError, match="unknown score_reduce"):
            episode_score(np.array([0.1]), "median")

    def test_config_rejects_bad_reduction(self):
        with pytest.raises(ValueError, match="unknown score_reduce"):
            Config(score_reduce="median")


class TestCentralisedCritic:
    def test_critic_sees_both_agents_actor_sees_one(self):
        agent = Agent(OBS, ACT, AGENTS, _cfg(centralised_critic=True))
        assert agent.critic_state_size == OBS * AGENTS
        assert agent.critic_action_size == ACT * AGENTS
        # Decentralised execution: the policy must be runnable from one racket's
        # own observation alone.
        assert agent.actor_local.fc1.in_features == OBS

    def test_decentralised_critic_is_narrower(self):
        agent = Agent(OBS, ACT, AGENTS, _cfg(centralised_critic=False))
        assert agent.critic_state_size == OBS
        assert agent.critic_action_size == ACT

    def test_each_step_stores_one_row_per_agent(self):
        agent = Agent(OBS, ACT, AGENTS, _cfg(centralised_critic=True))
        agent.step(*_step_batch())
        assert len(agent.memory) == AGENTS

    def test_buffer_rows_are_agent_centric(self):
        agent = Agent(OBS, ACT, AGENTS, _cfg(centralised_critic=True))
        states, actions, rewards, next_states, dones = _step_batch()
        agent.step(states, actions, rewards, next_states, dones)
        stored = agent.memory.states[:AGENTS]
        for i in range(AGENTS):
            assert np.allclose(stored[i][:OBS], states[i], atol=1e-6)

    def test_reward_row_matches_the_leading_agent(self):
        """Row i must carry agent i's reward, or the critic learns the wrong target."""
        agent = Agent(OBS, ACT, AGENTS, _cfg(centralised_critic=True))
        states, actions, _, next_states, dones = _step_batch()
        rewards = np.array([0.0, 0.1], dtype=np.float32)
        agent.step(states, actions, rewards, next_states, dones)
        assert agent.memory.rewards[0, 0] == pytest.approx(0.0)
        assert agent.memory.rewards[1, 0] == pytest.approx(0.1)

    def test_learning_runs_end_to_end(self):
        agent = Agent(OBS, ACT, AGENTS, _cfg(centralised_critic=True))
        for k in range(12):
            agent.step(*_step_batch(k))
        assert agent._updates > 0
        assert np.isfinite(agent.last_critic_loss)
        assert np.isfinite(agent.last_actor_loss)

    def test_td3_path_runs_with_centralised_critic(self):
        agent = Agent(OBS, ACT, AGENTS, _cfg(centralised_critic=True, algo="td3"))
        for k in range(12):
            agent.step(*_step_batch(k))
        assert agent._updates > 0
        assert np.isfinite(agent.last_critic_loss)

    def test_act_is_shaped_and_bounded(self):
        agent = Agent(OBS, ACT, AGENTS, _cfg())
        actions = agent.act(np.random.randn(AGENTS, OBS).astype(np.float32))
        assert actions.shape == (AGENTS, ACT)
        assert np.abs(actions).max() <= 1.0

    def test_checkpoint_roundtrip(self, tmp_path):
        agent = Agent(OBS, ACT, AGENTS, _cfg())
        for k in range(12):
            agent.step(*_step_batch(k))
        path = tmp_path / "ckpt.pt"
        agent.save(path)

        restored = Agent(OBS, ACT, AGENTS, _cfg())
        restored.load(path)
        states = np.random.randn(AGENTS, OBS).astype(np.float32)
        assert np.allclose(agent.act(states, add_noise=False),
                           restored.act(states, add_noise=False), atol=1e-6)


class TestConfigGuards:
    def test_centralised_requires_shared_policy(self):
        """Refusing beats silently doing something the config did not ask for."""
        with pytest.raises(ValueError, match="centralised_critic"):
            Config(centralised_critic=True, shared_policy=False)

    def test_layer_norm_is_the_default(self):
        """Carried from Reacher, where norm:none failed on 2 of 3 seeds."""
        assert Config().norm == "layer"

    def test_solve_criterion_matches_the_project(self):
        cfg = Config()
        assert cfg.solve_score == pytest.approx(0.5)
        assert cfg.solve_window == 100
        assert cfg.score_reduce == "max"
