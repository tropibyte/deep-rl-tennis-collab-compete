"""MADDPG for the two-racket Tennis environment.

The problem this solves that Reacher did not have
-------------------------------------------------
Reacher's 20 arms never interacted, so each one's environment was stationary and
a critic seeing one arm's state could explain everything that happened to it.
Tennis has two agents whose actions change what the other sees. From either
agent's point of view the environment therefore *changes as the other agent
learns*, and a critic that observes only its own agent is regressing on a moving
target with a hidden cause. That non-stationarity is the standard reason naive
independent DDPG is unstable here.

MADDPG (Lowe et al., 2017) answers it with centralised training and
decentralised execution: the critic sees **both** agents' observations and
actions, so the transition it is asked to explain is fully determined by its
inputs. The actor still sees only its own 24 observations, so the learned policy
remains runnable by a single racket that knows nothing about the other.

Shared policy, agent-centric critic
-----------------------------------
The two rackets face a symmetric task, so one actor drives both -- self-play,
and every episode yields two trajectories for the same policy.

A shared critic needs care. Fed the joint state in a fixed ordering, the same
input would map to two different targets (one per agent's reward) and it could
not fit both. So every transition is stored **twice, once from each agent's
point of view**, with that agent's own observation leading:

    agent 0's row:  state = [obs0, obs1]   action = [a0, a1]   reward = r0
    agent 1's row:  state = [obs1, obs0]   action = [a1, a0]   reward = r1

The critic then always answers "what is this state-action worth *to the agent
whose observation comes first*", which is well defined, and both agents'
experience trains the same weights. ``centralised_critic: false`` falls back to
a plain per-agent critic over ``(obs_i, a_i)`` -- the ablation that measures
whether any of this was necessary.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .networks import Actor, Critic, TwinCritic
from .noise import make_noise
from .replay import PrioritizedReplayBuffer, ReplayBuffer


def agent_centric_views(per_agent: np.ndarray) -> np.ndarray:
    """Reorder a ``(num_agents, dim)`` array once per agent, that agent first.

    Returns ``(num_agents, num_agents * dim)``: row *i* is agent *i*'s own vector
    followed by everyone else's, in rotation order. For two agents that is simply
    ``[[a, b], [b, a]]``.
    """
    n = per_agent.shape[0]
    return np.stack([np.roll(per_agent, -i, axis=0).reshape(-1) for i in range(n)])


class _NStepAccumulator:
    """Per-agent n-step return accumulator.

    Each agent needs its own window: the two rackets are at different points in
    the rally, and mixing them would splice unrelated states together.
    """

    def __init__(self, num_agents: int, n: int, gamma: float):
        self.n = int(n)
        self.gamma = float(gamma)
        self._q = [deque() for _ in range(num_agents)]

    def push(self, states, actions, rewards, next_states, dones):
        out_s, out_a, out_r, out_ns, out_d = [], [], [], [], []

        def emit(q):
            out_s.append(q[0][0])
            out_a.append(q[0][1])
            r, ns, d = self._fold(q)
            out_r.append(r)
            out_ns.append(ns)
            out_d.append(d)

        for i, q in enumerate(self._q):
            q.append((states[i], actions[i], float(rewards[i]),
                      next_states[i], bool(dones[i])))
            if len(q) == self.n:
                emit(q)
                q.popleft()
            if dones[i]:
                while q:
                    emit(q)
                    q.popleft()
        if not out_s:
            return None
        return (np.asarray(out_s), np.asarray(out_a), np.asarray(out_r),
                np.asarray(out_ns), np.asarray(out_d))

    def _fold(self, q):
        r = 0.0
        for k, (_, _, rew, _, done) in enumerate(q):
            r += (self.gamma ** k) * rew
            if done:
                return r, q[k][3], True
        return r, q[-1][3], q[-1][4]

    def reset(self) -> None:
        for q in self._q:
            q.clear()


class Agent:
    """Shared-policy MADDPG (or plain DDPG) driving both rackets."""

    def __init__(self, obs_size: int, action_size: int, num_agents: int, cfg: Config):
        self.cfg = cfg
        self.obs_size = obs_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.device = torch.device(cfg.device)
        self.is_td3 = cfg.algo == "td3"
        self.centralised = cfg.centralised_critic

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # The actor always sees one agent's own observation, centralised or not:
        # that is what "decentralised execution" means.
        net_kwargs = dict(norm=cfg.norm, input_norm=cfg.input_norm)
        self.actor_local = Actor(obs_size, action_size, cfg.hidden_actor, **net_kwargs).to(self.device)
        self.actor_target = Actor(obs_size, action_size, cfg.hidden_actor, **net_kwargs).to(self.device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor_local.parameters(), lr=cfg.lr_actor)

        # The critic's input width is the only thing centralisation changes.
        self.critic_state_size = obs_size * num_agents if self.centralised else obs_size
        self.critic_action_size = action_size * num_agents if self.centralised else action_size

        critic_cls = TwinCritic if self.is_td3 else Critic
        self.critic_local = critic_cls(
            self.critic_state_size, self.critic_action_size,
            hidden=cfg.hidden_critic, **net_kwargs).to(self.device)
        self.critic_target = critic_cls(
            self.critic_state_size, self.critic_action_size,
            hidden=cfg.hidden_critic, **net_kwargs).to(self.device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_opt = torch.optim.Adam(
            self.critic_local.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay)

        buf_cls = PrioritizedReplayBuffer if cfg.per.enabled else ReplayBuffer
        buf_kwargs = dict(
            capacity=cfg.buffer_size,
            state_size=self.critic_state_size,
            action_size=self.critic_action_size,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            device=self.device,
        )
        if cfg.per.enabled:
            buf_kwargs.update(alpha=cfg.per.alpha, beta_start=cfg.per.beta_start,
                              beta_frames=cfg.per.beta_frames)
        self.memory = buf_cls(**buf_kwargs)

        self.noise = make_noise(
            cfg.noise.kind, (num_agents, action_size), seed=cfg.seed,
            sigma=cfg.noise.sigma, sigma_min=cfg.noise.sigma_min,
            sigma_decay=cfg.noise.sigma_decay,
            **({"theta": cfg.noise.theta, "mu": cfg.noise.mu}
               if cfg.noise.kind == "ou" else {}),
        )

        self.nstep = (_NStepAccumulator(num_agents, cfg.n_step, cfg.gamma)
                      if cfg.n_step > 1 else None)
        self.gamma_n = cfg.gamma ** cfg.n_step

        self._t = 0
        self._updates = 0
        self._grad_norm_sum = 0.0
        self._grad_norm_max = 0.0
        self._grad_norm_n = 0
        self._grad_clipped_n = 0
        self.last_critic_loss = float("nan")
        self.last_actor_loss = float("nan")

    # -- interaction -----------------------------------------------------
    def act(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """One action per racket, each from that racket's own observation."""
        states_t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states_t).cpu().numpy()
        self.actor_local.train()
        if add_noise:
            actions = actions + self.noise.sample()
        return np.clip(actions, -1.0, 1.0)

    def step(self, states, actions, rewards, next_states, dones) -> None:
        """Store the step from both agents' points of view, then learn on schedule."""
        if self.nstep is not None:
            ready = self.nstep.push(states, actions, rewards, next_states, dones)
            if ready is None:
                self._t += 1
                return
            states, actions, rewards, next_states, dones = ready

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)

        if self.centralised:
            # One row per agent, that agent's own view leading. See the module
            # docstring for why the ordering matters.
            s = agent_centric_views(states)
            a = agent_centric_views(actions)
            ns = agent_centric_views(next_states)
        else:
            s, a, ns = states, actions, next_states

        self.memory.add_batch(s, a, rewards, ns, dones)

        self._t += 1
        if len(self.memory) < max(self.cfg.learn_start, self.cfg.batch_size):
            return
        if self._t % self.cfg.update_every != 0:
            return
        for _ in range(self.cfg.updates_per_cycle):
            self.learn()

    def reset_noise(self) -> None:
        self.noise.reset()
        if self.nstep is not None:
            self.nstep.reset()

    # -- learning --------------------------------------------------------
    def _actor_actions_for_all(self, joint_states: torch.Tensor,
                               actor: torch.nn.Module) -> torch.Tensor:
        """Run the shared actor on every agent's slice of a joint state.

        Used to build the target action: with a shared policy we can predict what
        the *other* racket would do too, which is exactly the joint action the
        centralised critic expects.
        """
        chunks = [actor(joint_states[:, i * self.obs_size:(i + 1) * self.obs_size])
                  for i in range(self.num_agents)]
        return torch.cat(chunks, dim=1)

    def learn(self) -> None:
        cfg = self.cfg
        if cfg.per.enabled:
            states, actions, rewards, next_states, dones, weights, idx = self.memory.sample()
        else:
            states, actions, rewards, next_states, dones = self.memory.sample()
            weights, idx = None, None

        with torch.no_grad():
            if self.centralised:
                next_actions = self._actor_actions_for_all(next_states, self.actor_target)
            else:
                next_actions = self.actor_target(next_states)

            if self.is_td3:
                noise = (torch.randn_like(next_actions) * cfg.td3.target_noise).clamp(
                    -cfg.td3.noise_clip, cfg.td3.noise_clip)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)
                q1_next, q2_next = self.critic_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next)
            else:
                q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma_n * q_next * (1.0 - dones)

        if self.is_td3:
            q1, q2 = self.critic_local(states, actions)
            td_error = q_target - q1
            losses = (F.mse_loss(q1, q_target, reduction="none")
                      + F.mse_loss(q2, q_target, reduction="none"))
        else:
            q_expected = self.critic_local(states, actions)
            td_error = q_target - q_expected
            losses = F.mse_loss(q_expected, q_target, reduction="none")

        critic_loss = (losses * weights).mean() if weights is not None else losses.mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        # clip_grad_norm_ returns the pre-clip norm, so recording it is free when
        # clipping is on; with it off we pass an infinite threshold, which
        # measures without scaling. On Reacher this proved the clip never fired.
        threshold = cfg.grad_clip_critic
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.critic_local.parameters(),
            threshold if threshold is not None else float("inf"))
        self._grad_norm_sum += float(total_norm)
        self._grad_norm_max = max(self._grad_norm_max, float(total_norm))
        self._grad_norm_n += 1
        if threshold is not None and float(total_norm) > threshold:
            self._grad_clipped_n += 1
        self.critic_opt.step()

        if cfg.per.enabled:
            self.memory.update_priorities(idx, td_error.detach().cpu().numpy())

        self._updates += 1
        delay = cfg.td3.policy_delay if self.is_td3 else 1
        if self._updates % delay == 0:
            if self.centralised:
                # Only the acting agent's action is replaced by the policy's
                # current output; the other racket's action stays as it was
                # played. That is MADDPG's actor update -- differentiating
                # through both would credit this policy for the other agent's
                # choice.
                own_pred = self.actor_local(states[:, :self.obs_size])
                others = actions[:, self.action_size:]
                pred_actions = torch.cat([own_pred, others], dim=1)
            else:
                pred_actions = self.actor_local(states)

            if self.is_td3:
                actor_loss = -self.critic_local.q1_only(states, pred_actions).mean()
            else:
                actor_loss = -self.critic_local(states, pred_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            if cfg.grad_clip_actor is not None:
                torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(),
                                               cfg.grad_clip_actor)
            self.actor_opt.step()
            self.last_actor_loss = float(actor_loss.detach())

            self._soft_update(self.actor_local, self.actor_target)
            self._soft_update(self.critic_local, self.critic_target)
        elif not self.is_td3:
            self._soft_update(self.critic_local, self.critic_target)

        self.last_critic_loss = float(critic_loss.detach())

    def _soft_update(self, local: torch.nn.Module, target: torch.nn.Module) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for t, l in zip(target.parameters(), local.parameters()):
                t.mul_(1.0 - tau).add_(tau * l)

    @property
    def grad_norm_stats(self) -> dict:
        """Critic gradient-norm summary: mean, max, and how often the clip bound."""
        if not self._grad_norm_n:
            return {}
        return {
            "mean": self._grad_norm_sum / self._grad_norm_n,
            "max": self._grad_norm_max,
            "clipped_fraction": self._grad_clipped_n / self._grad_norm_n,
            "updates": self._grad_norm_n,
        }

    # -- checkpoints -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor_local.state_dict(),
                "critic": self.critic_local.state_dict(),
                "config": self.cfg.to_dict(),
                "obs_size": self.obs_size,
                "action_size": self.action_size,
                "num_agents": self.num_agents,
            },
            p,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_local.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            self.critic_local.load_state_dict(ckpt["critic"])
            self.critic_target.load_state_dict(ckpt["critic"])
