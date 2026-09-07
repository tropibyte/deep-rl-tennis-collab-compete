"""Diagnose why a training run is underperforming, without waiting for it to finish.

A plateauing learning curve is a symptom, not a cause. This separates the
candidate causes by measuring them directly on a checkpoint:

* **Input scale.** The 33 observations are raw Unity physics quantities --
  positions, velocities, angular velocities -- on wildly different scales. A
  network with no normalisation on its input has to learn that scaling in its
  first layer, and if some features are orders of magnitude larger than others
  they dominate the gradient.
* **Actor saturation.** A tanh policy that has been pushed into its flat region
  outputs +/-1 for every state, and its gradient vanishes. Exploration noise
  then cannot recover it because the noise is added *after* the tanh.
* **Critic calibration.** If the critic's Q estimate is far from the return the
  policy actually achieves, the policy gradient is pointing somewhere unrelated
  to the objective.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tennis.agent import Agent
from tennis.config import Config
from tennis.env import ReacherEnv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--worker-id", type=int, default=50)
    ap.add_argument("--exe", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    print(f"checkpoint: {args.checkpoint}")
    print(f"  norm={cfg.norm}  hidden={cfg.hidden_actor}  "
          f"updates={cfg.updates_per_cycle}/{cfg.update_every}  noise={cfg.noise.kind}")

    states_log, rewards_log, actions_log = [], [], []

    with ReacherEnv(exe_path=args.exe, worker_id=args.worker_id, no_graphics=True,
                    seed=0, run_dir=Path("results/_diag_run")) as env:
        agent = Agent(env.state_size, env.action_size, env.num_agents, cfg)
        agent.load(args.checkpoint)

        states = env.reset(train_mode=True)
        for _ in range(args.steps):
            actions = agent.act(states, add_noise=False)
            states_log.append(states.copy())
            actions_log.append(actions.copy())
            states, rewards, dones, _ = env.step(actions)
            rewards_log.append(rewards.copy())
            if dones.any():
                states = env.reset(train_mode=True)

    S = np.concatenate(states_log)          # (steps*agents, 33)
    A = np.concatenate(actions_log)         # (steps*agents, 4)
    R = np.concatenate(rewards_log)

    print("\n[1] observation scale")
    absmax = np.abs(S).max(axis=0)
    print(f"  per-feature |max|: min {absmax.min():.3f}  median {np.median(absmax):.3f}  "
          f"max {absmax.max():.3f}")
    print(f"  ratio largest/smallest feature: {absmax.max() / max(absmax.min(), 1e-9):.1f}x")
    print(f"  overall mean {S.mean():.3f}  std {S.std():.3f}")
    big = np.argsort(absmax)[-5:][::-1]
    print(f"  widest features (index: |max|): "
          + ", ".join(f"{i}: {absmax[i]:.1f}" for i in big))

    # Constant features are a trap specifically for batch normalisation, which
    # divides each feature by its own standard deviation: a feature with zero
    # variance is divided by ~sqrt(eps), turning numerical noise into a large
    # input. Layer norm normalises across features within a sample and does not
    # have this failure mode.
    std = S.std(axis=0)
    dead = np.flatnonzero(std < 1e-6)
    print(f"  zero-variance features: {len(dead)} of {S.shape[1]}"
          + (f"  (indices {dead.tolist()})" if len(dead) else ""))
    if len(dead):
        print("  -> batch norm on the INPUT will amplify noise in these; "
              "prefer layer norm, or drop them.")

    print("\n[2] actor saturation (exploration off)")
    sat = float((np.abs(A) > 0.99).mean())
    print(f"  |action| > 0.99 : {sat:.1%} of all torques")
    print(f"  mean |action|   : {np.abs(A).mean():.3f}")
    print(f"  per-dim mean    : {np.round(A.mean(axis=0), 3)}")
    print(f"  per-dim std     : {np.round(A.std(axis=0), 3)}")
    if sat > 0.5:
        print("  -> SATURATED: the tanh is in its flat region and gradients vanish there.")
    elif sat < 0.05:
        print("  -> healthy: the policy is using the interior of the action range.")

    print("\n[3] critic calibration")
    with torch.no_grad():
        s_t = torch.from_numpy(S[:2048])
        a_t = torch.from_numpy(A[:2048])
        q = agent.critic_local(s_t, a_t)
        if isinstance(q, tuple):
            q = q[0]
        q = q.numpy().ravel()
    # What the policy actually earns per step, extrapolated over the remaining
    # episode under a 0.99 discount: an upper bound on a sane Q.
    per_step = float(R.mean())
    horizon_value = per_step * (1 - cfg.gamma ** 1001) / (1 - cfg.gamma)
    print(f"  reward per step per agent : {per_step:.4f}")
    print(f"  implied discounted return : {horizon_value:.2f}")
    print(f"  critic Q: mean {q.mean():.2f}  std {q.std():.2f}  "
          f"range [{q.min():.2f}, {q.max():.2f}]")
    ratio = q.mean() / max(horizon_value, 1e-6)
    print(f"  Q / achieved return       : {ratio:.2f}x")
    if ratio > 3:
        print("  -> OVERESTIMATING: the actor is climbing a value the policy cannot realise.")
    elif ratio < 0.33:
        print("  -> UNDERESTIMATING: the critic is not tracking the policy's own returns.")
    else:
        print("  -> calibrated within a factor of 3.")

    print(f"\n[4] episode score at this checkpoint")
    print(f"  mean reward/step {per_step:.4f} -> ~{per_step * 1001:.1f} per episode")


if __name__ == "__main__":
    main()
