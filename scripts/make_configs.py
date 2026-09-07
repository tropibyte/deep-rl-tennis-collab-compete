"""Generate every ablation config from one baseline, so the arms cannot drift.

An ablation is only interpretable if each arm differs from the baseline in
exactly the way its name claims. Hand-maintained YAML does not hold that
property: when the baseline changes -- as it did here, when the first baseline's
missing normalisation was diagnosed -- every arm silently keeps the old setting
and the whole study measures variations of a broken configuration.

So the arms are generated. Each is the baseline text with one or two named
substitutions applied, and a substitution that fails to match exactly once is an
error rather than a silent no-op.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# name -> (header comment, [(pattern, replacement), ...])
VARIANTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "ddpg_noclip": (
        "# Ablation: the baseline with critic gradient clipping removed.\n"
        "# The project's benchmark write-up identifies clipping as the change that\n"
        "# turned a run that collapsed at episode 100 into one that held together.\n"
        "# This arm tests that claim rather than repeating it.",
        [(r"^grad_clip_critic: 1\.0$", "grad_clip_critic: null")],
    ),
    "ddpg_cadence": (
        "# Ablation: the same number of updates as the baseline, spread differently.\n"
        "#\n"
        "# The baseline does 20 updates every 20 steps; this does 1 update every\n"
        "# step. Both perform 1,001 optimiser steps per episode, so any difference\n"
        "# is caused by the spacing of updates rather than their quantity.\n"
        "#\n"
        "# That distinction matters more now than when this arm was written. The\n"
        "# diagnostics showed update *count* dominates, and the benchmark's\n"
        "# write-up credits its fix to 'getting less aggressive with the number of\n"
        "# updates' -- but that change altered the count and the batching at once.\n"
        "# This arm separates them.",
        [(r"^update_every: 20$", "update_every: 1"),
         (r"^updates_per_cycle: 20$", "updates_per_cycle: 1")],
    ),
    "ddpg_nstep": (
        "# Ablation: 5-step returns. Reacher rewards are dense but tiny (+0.1 per\n"
        "# step in the goal), so a 1-step target propagates credit slowly. Longer\n"
        "# returns trade bootstrap bias for variance.",
        [(r"^n_step: 1$", "n_step: 5")],
    ),
    "ddpg_gaussian": (
        "# Ablation: uncorrelated Gaussian exploration instead of Ornstein-Uhlenbeck.\n"
        "# TD3 found OU's temporal correlation unnecessary; this arm checks whether\n"
        "# its extra hyperparameter earns its place here.",
        [(r"^  kind: ou$", "  kind: gaussian")],
    ),
    "ddpg_per": (
        "# Ablation: prioritised experience replay.\n"
        "#\n"
        "# Not obviously right for this task. PER helps most when reward is sparse\n"
        "# and a few transitions carry nearly all the information; Reacher's reward\n"
        "# is dense, so the question is whether prioritising by TD error just\n"
        "# re-samples noise. It also costs wall-clock: the sum tree is walked once\n"
        "# per sampled transition, on the CPU that is already the bottleneck.",
        [(r"^per:\n  enabled: false$",
          "per:\n  enabled: true\n  alpha: 0.6\n  beta_start: 0.4\n  beta_frames: 100000")],
    ),
    "ddpg_lowupdates": (
        "# Ablation: the benchmark's update budget -- 10 updates per cycle rather\n"
        "# than the baseline's 20, halving the update-to-data ratio from 6.4 to 3.2\n"
        "# samples drawn per transition collected.\n"
        "#\n"
        "# This is the single change that mattered most in the pre-study\n"
        "# diagnostics: at episode 30 the low-update setting scored 2.12 against\n"
        "# 4.63, and at episode 20 it scored 1.01 against 3.00. The arm exists to\n"
        "# measure that properly, across seeds and to convergence, rather than\n"
        "# resting on two points from an 80-episode race.",
        [(r"^updates_per_cycle: 20$", "updates_per_cycle: 10")],
    ),
    "ddpg_nonorm": (
        "# Negative control: no normalisation anywhere -- what the project's\n"
        "# benchmark write-up implies by never mentioning the subject.\n"
        "#\n"
        "# Differs from the baseline in normalisation ONLY, so it is not identical\n"
        "# to this repo's abandoned first attempt, which also used the lower update\n"
        "# ratio. That attempt plateaued between 0.8 and 8.4 by ~120 episodes on\n"
        "# four seeds, with 66.5% of torques pinned at |a|>0.99 and one joint dead\n"
        "# at a constant -1.0 (Report.md section 4); this arm isolates how much of\n"
        "# that was the missing normalisation alone.",
        [(r"^norm: .*$", "norm: none"),
         (r"^input_norm: .*$", "input_norm: none")],
    ),
    "td3": (
        "# TD3 (Fujimoto et al., 2018): the baseline plus twin critics with a min\n"
        "# over targets, noise added to the target action, and an actor updated\n"
        "# every second critic update.\n"
        "#\n"
        "# Exploration switches to Gaussian because that is what the TD3 paper\n"
        "# uses; ddpg_gaussian isolates whether that swap matters on its own.",
        [(r"^algo: ddpg$", "algo: td3"),
         (r"^  kind: ou$", "  kind: gaussian"),
         (r"^solve_score: 30\.0$",
          "td3:\n  policy_delay: 2\n  target_noise: 0.2\n  noise_clip: 0.5\n\nsolve_score: 30.0")],
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="configs/ddpg.yaml")
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--only", nargs="*", default=None, help="generate a subset")
    args = ap.parse_args()

    base_path = Path(args.base)
    base = base_path.read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = args.only or list(VARIANTS)
    for name in wanted:
        header, subs = VARIANTS[name]
        text = base
        for pattern, repl in subs:
            text, n = re.subn(pattern, repl, text, flags=re.M)
            if n != 1:
                raise SystemExit(
                    f"{name}: pattern {pattern!r} matched {n} times in {base_path} "
                    "(the baseline changed shape; fix the substitution)"
                )
        # Swap the name field and replace the baseline's header comment.
        text, n = re.subn(r"^name: .*$", f"name: {name}", text, flags=re.M)
        if n != 1:
            raise SystemExit(f"{name}: could not find a name: line in {base_path}")
        body = text.split("name:", 1)[1]
        (out_dir / f"{name}.yaml").write_text(header.rstrip() + "\nname:" + body,
                                              encoding="utf-8")
        print(f"  wrote {out_dir / f'{name}.yaml'}")

    print(f"\n{len(wanted)} arms generated from {base_path}")


if __name__ == "__main__":
    main()
