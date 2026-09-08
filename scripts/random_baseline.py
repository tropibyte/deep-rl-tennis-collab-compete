"""Score a uniform-random policy, and confirm the environment is wired correctly.

Two jobs. The obvious one is a floor for the Report: on Tennis a random policy
scores almost nothing, so any learning curve has to be read against roughly zero
rather than against a comfortable baseline.

The less obvious one is a sanity check, and it matters more here than it did on
Reacher. Tennis rewards are sparse — a random racket rarely returns the ball at
all — so a genuinely broken reward channel and a working one both look like
"mostly zeros". This asserts that *some* reward is non-zero across enough
episodes to be conclusive, that episode lengths vary (they are not fixed as in
Reacher), and that the observation and action shapes match what the agent code
assumes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tennis.env import TennisEnv

EXPECTED_OBS, EXPECTED_ACT, EXPECTED_AGENTS = 24, 2, 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=50,
                    help="more than Reacher needs: the reward is sparse enough "
                         "that a handful of episodes can legitimately score zero")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--out", default="results/random_baseline.json")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    scores: list[float] = []
    lengths: list[int] = []
    positive_reward_steps = 0

    with TennisEnv(exe_path=args.exe, no_graphics=True, seed=0,
                   run_dir=Path("results/_random_run")) as env:
        print(f"  agents      : {env.num_agents}")
        print(f"  observation : {env.state_size}")
        print(f"  action      : {env.action_size} ({env.action_type})")

        mismatches = []
        if env.num_agents != EXPECTED_AGENTS:
            mismatches.append(f"agents {env.num_agents} != {EXPECTED_AGENTS}")
        if env.state_size != EXPECTED_OBS:
            mismatches.append(f"observation {env.state_size} != {EXPECTED_OBS}")
        if env.action_size != EXPECTED_ACT:
            mismatches.append(f"action {env.action_size} != {EXPECTED_ACT}")
        if mismatches:
            # Loud, because every shape assumption in agent.py depends on these
            # and a mismatch would surface as bad learning, not as an error.
            print("\n  !! SHAPE MISMATCH: " + "; ".join(mismatches))
            print("     The agent's centralised critic sizes are derived from these.")
        else:
            print("  shapes match the agent's assumptions")

        for ep in range(args.episodes):
            env.reset(train_mode=True)
            totals = np.zeros(env.num_agents)
            steps = 0
            while steps < args.max_steps:
                actions = rng.uniform(-1, 1, (env.num_agents, env.action_size))
                _, rewards, dones, _ = env.step(actions)
                totals += rewards
                positive_reward_steps += int((rewards > 0).sum())
                steps += 1
                if dones.any():
                    break
            scores.append(float(totals.max()))   # the project's scoring rule
            lengths.append(steps)

    assert positive_reward_steps > 0, (
        f"no positive reward in {args.episodes} episodes — either the reward "
        "channel is not being read, or the episode count is too low to be "
        "conclusive on a reward this sparse"
    )
    assert len(set(lengths)) > 1, (
        f"every episode was exactly {lengths[0]} steps — Tennis episodes should "
        "vary in length, so termination is probably not being read correctly"
    )

    result = {
        "episodes": args.episodes,
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "nonzero_episodes": int(sum(s > 0 for s in scores)),
        "mean_episode_length": float(np.mean(lengths)),
        "min_episode_length": int(np.min(lengths)),
        "max_episode_length": int(np.max(lengths)),
        "positive_reward_steps": positive_reward_steps,
        "scores": scores,
    }
    print(f"\n  random policy : {result['mean_score']:.4f} mean "
          f"(max {result['max_score']:.2f}, "
          f"{result['nonzero_episodes']}/{args.episodes} episodes scored above zero)")
    print(f"  episode length: {result['mean_episode_length']:.1f} mean "
          f"({result['min_episode_length']}-{result['max_episode_length']})")
    print(f"  target to solve: 0.5 averaged over 100 episodes")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
