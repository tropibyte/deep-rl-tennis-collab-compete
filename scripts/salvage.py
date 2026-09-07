"""Rebuild a training record from a run's log, for runs that were stopped early.

Results are only written when ``train()`` returns, so a run killed part-way
leaves nothing behind but its progress log -- which nonetheless contains the
per-episode score and moving average for every episode it completed. That is
the whole learning curve, and it is worth keeping when a run is stopped because
its *configuration* was wrong rather than because the data was bad.

The reconstructed record is marked ``truncated: true`` and carries no
checkpoint, so nothing downstream can mistake it for a completed run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from tennis.config import Config


def parse_log(path: Path) -> tuple[list[float], list[float]]:
    """Pull (score, avg100) pairs out of the progress lines.

    The trainer writes progress with a carriage return so the terminal shows one
    updating line; in a redirected file that becomes one long line, so split on
    \\r before matching.
    """
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    scores: list[float] = []
    moving: list[float] = []
    for line in text.splitlines():
        parts = line.split()
        # "ep 12/300 score 1.40 avg100 1.29 buffer 240240 42.9s/ep"
        if len(parts) >= 6 and parts[0] == "ep" and parts[2] == "score":
            try:
                scores.append(float(parts[3]))
                moving.append(float(parts[5]))
            except ValueError:
                continue
    return scores, moving


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", help="run logs to rebuild records from")
    ap.add_argument("--config-dir", default="results",
                    help="where the matching <tag>.config.yaml files live")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for log_path in args.logs:
        log = Path(log_path)
        tag = log.stem
        scores, moving = parse_log(log)
        if not scores:
            print(f"  {tag}: no progress lines, skipped")
            continue

        cfg_path = Path(args.config_dir) / f"{tag}.config.yaml"
        if not cfg_path.exists():
            print(f"  {tag}: no config at {cfg_path}, skipped")
            continue
        cfg = Config.from_dict(yaml.safe_load(cfg_path.read_text(encoding="utf-8")))

        window = cfg.solve_window
        solved_episode = None
        for i, avg in enumerate(moving, start=1):
            if i >= window and avg >= cfg.solve_score:
                solved_episode = i - window
                break

        best = max(moving[window - 1:], default=None) if len(moving) >= window else None
        record = {
            "tag": tag,
            "name": cfg.name,
            "algo": cfg.algo,
            "seed": cfg.seed,
            "scores": scores,
            "moving_average": moving,
            "solved_episode": solved_episode,
            "episodes_run": None if solved_episode is None else solved_episode + window,
            "solved": solved_episode is not None,
            "best_moving_average": best,
            "final_moving_average": moving[-1],
            "episodes": len(scores),
            # Reconstructed from a log, so the per-agent detail and timing that
            # train() records are simply not available.
            "truncated": True,
            "agent_scores": None,
            "elapsed_sec": None,
            "sec_per_episode": float("nan"),
            "config": cfg.to_dict(),
        }
        out = out_dir / f"{tag}.json"
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"  {tag}: {len(scores)} episodes, final avg100 {moving[-1]:.2f} -> {out}")


if __name__ == "__main__":
    main()
