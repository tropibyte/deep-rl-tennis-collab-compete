"""Measure what this machine can actually do, before committing to a study plan.

Reports three numbers that between them predict wall-clock for any training
configuration:

* **env throughput** -- Unity steps/second with random actions and no learning.
  One step advances all ``num_agents`` arms, so transitions/second is this
  times ``num_agents``.
* **update throughput** -- optimiser steps/second for the actor+critic pair at
  the configured batch size, measured on synthetic data with no environment
  attached.
* **projected episode time** -- the two combined under a given update schedule.

Run it before and after any change that could plausibly cost time.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tennis.env import ReacherEnv
from tennis.networks import Actor, Critic


def bench_env(steps: int, exe: str | None, run_dir: Path) -> dict:
    with ReacherEnv(exe_path=exe, no_graphics=True, seed=0, run_dir=run_dir) as env:
        info = {
            "num_agents": env.num_agents,
            "state_size": env.state_size,
            "action_size": env.action_size,
            "action_type": env.action_type,
            "brain_name": env.brain_name,
        }
        print(f"  brain          : {info['brain_name']}")
        print(f"  agents         : {info['num_agents']}")
        print(f"  state size     : {info['state_size']}")
        print(f"  action size    : {info['action_size']} ({info['action_type']})")

        env.reset()
        # Warm up: the first few steps include JIT/allocation costs that would
        # otherwise be amortised into the measurement.
        for _ in range(10):
            env.step(np.random.uniform(-1, 1, (env.num_agents, env.action_size)))

        rewards_seen = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            actions = np.random.uniform(-1, 1, (env.num_agents, env.action_size))
            _, r, done, _ = env.step(actions)
            rewards_seen += float(r.sum())
            if done.any():
                env.reset()
        elapsed = time.perf_counter() - t0

    info["env_steps_per_sec"] = steps / elapsed
    info["transitions_per_sec"] = steps * info["num_agents"] / elapsed
    info["random_reward_per_step_per_agent"] = rewards_seen / (steps * info["num_agents"])
    return info


def bench_update(state_size: int, action_size: int, batch: int, iters: int) -> float:
    """Optimiser steps/second for one actor update + one critic update."""
    actor = Actor(state_size, action_size)
    critic = Critic(state_size, action_size)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    states = torch.randn(batch, state_size)
    actions = torch.rand(batch, action_size) * 2 - 1
    targets = torch.randn(batch, 1)

    def one_update():
        critic_opt.zero_grad()
        loss = torch.nn.functional.mse_loss(critic(states, actions), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
        critic_opt.step()

        actor_opt.zero_grad()
        actor_loss = -critic(states, actor(states)).mean()
        actor_loss.backward()
        actor_opt.step()

    for _ in range(10):
        one_update()

    t0 = time.perf_counter()
    for _ in range(iters):
        one_update()
    return iters / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=300, help="env steps to time")
    ap.add_argument("--updates", type=int, default=200, help="optimiser steps to time")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--episode-len", type=int, default=None,
                    help="steps per episode for the projection; measured if omitted, "
                         "since Tennis episodes end when the ball drops and lengthen "
                         "as the agents improve")
    ap.add_argument("--update-every", type=int, default=20)
    ap.add_argument("--updates-per-cycle", type=int, default=10)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--out", default="results/bench.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"torch threads    : {torch.get_num_threads()}")

    run_dir = Path("results/_bench_run")
    print("\n[env]")
    info = bench_env(args.steps, args.exe, run_dir)
    print(f"  env steps/s    : {info['env_steps_per_sec']:.1f}")
    print(f"  transitions/s  : {info['transitions_per_sec']:.0f}")
    print(f"  random reward  : {info['random_reward_per_step_per_agent']:.4f} per step per agent")

    print("\n[update]")
    ups = bench_update(info["state_size"], info["action_size"], args.batch, args.updates)
    info["updates_per_sec"] = ups
    info["batch"] = args.batch
    print(f"  updates/s      : {ups:.1f} (batch {args.batch})")

    # Projection: an episode costs its env steps plus the learning triggered
    # along the way. These are sequential in a single-process trainer.
    ep_env = args.episode_len / info["env_steps_per_sec"]
    cycles = args.episode_len / args.update_every
    ep_learn = cycles * args.updates_per_cycle / ups
    info["projected_episode_sec"] = ep_env + ep_learn
    info["projected_episode_env_sec"] = ep_env
    info["projected_episode_learn_sec"] = ep_learn

    print("\n[projection]")
    print(f"  per episode    : {ep_env + ep_learn:.1f}s "
          f"({ep_env:.1f}s env + {ep_learn:.1f}s learning)")
    for n in (150, 300):
        print(f"  {n:>4} episodes : {(ep_env + ep_learn) * n / 3600:.2f} h")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
