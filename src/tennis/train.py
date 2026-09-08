"""The training loop, and the bookkeeping the rubric asks for.

Two things differ from the Reacher project's loop, and both change the numbers
rather than just the plumbing.

**Scoring.** An episode's score is the *maximum* of the two agents' undiscounted
totals, not the mean. That is the project's own definition. Using the mean would
roughly halve every score and the environment would never register as solved, so
``score_reduce`` is explicit in the config and recorded in every result file.

**Episode length.** Reacher episodes are a fixed 1001 steps. A Tennis episode
ends when the ball hits the ground, so early episodes last a dozen steps and a
competent agent produces long rallies. Wall-clock per episode therefore *grows*
as training succeeds, which makes an early throughput estimate an
underestimate -- and makes episode length itself a useful progress signal, so it
is recorded alongside the score.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .agent import Agent
from .config import Config
from .env import TennisEnv


def episode_score(agent_totals: np.ndarray, reduce: str) -> float:
    """Collapse per-agent episode totals into the one number the rubric scores."""
    if reduce == "max":
        return float(np.max(agent_totals))
    if reduce == "mean":
        return float(np.mean(agent_totals))
    raise ValueError(f"unknown score_reduce: {reduce!r}")


def train(
    cfg: Config,
    env: TennisEnv,
    checkpoint_dir: str | Path = "checkpoints",
    results_dir: str | Path = "results",
    tag: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train to convergence (or to ``cfg.episodes``) and return the run record."""
    tag = tag or f"{cfg.name}_seed{cfg.seed}"
    checkpoint_dir = Path(checkpoint_dir)
    results_dir = Path(results_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)

    agent = Agent(env.state_size, env.action_size, env.num_agents, cfg)

    scores: list[float] = []
    agent_scores: list[list[float]] = []
    lengths: list[int] = []
    moving: list[float] = []
    window: deque[float] = deque(maxlen=cfg.solve_window)

    solved_episode: int | None = None
    best_moving = -np.inf
    t_start = time.perf_counter()

    for episode in range(1, cfg.episodes + 1):
        states = env.reset(train_mode=True)
        agent.reset_noise()
        totals = np.zeros(env.num_agents, dtype=np.float64)
        steps = 0

        for _ in range(cfg.max_steps):
            actions = agent.act(states, add_noise=True)
            next_states, rewards, dones, _ = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            totals += rewards
            states = next_states
            steps += 1
            if dones.any():
                break

        score = episode_score(totals, cfg.score_reduce)
        scores.append(score)
        agent_scores.append([float(s) for s in totals])
        lengths.append(steps)
        window.append(score)
        avg = float(np.mean(window))
        moving.append(avg)

        window_full = len(window) == cfg.solve_window
        if window_full and avg > best_moving:
            best_moving = avg
            agent.save(checkpoint_dir / f"{tag}_best.pt")

        if window_full and solved_episode is None and avg >= cfg.solve_score:
            solved_episode = episode - cfg.solve_window
            agent.save(checkpoint_dir / f"{tag}_solved.pt")
            if verbose:
                print(
                    f"\n  solved: {cfg.solve_window}-episode average {avg:.3f} >= "
                    f"{cfg.solve_score} after {episode} episodes "
                    f"(reported as episode {solved_episode})"
                )

        # Every episode, not every tenth: the log is the only record a run
        # stopped early leaves behind, and a sparse prefix misaligns the
        # reconstructed curve against its episode numbers.
        if verbose:
            elapsed = time.perf_counter() - t_start
            print(
                f"  ep {episode:>5}/{cfg.episodes}  score {score:>6.3f}  "
                f"avg{cfg.solve_window} {avg:>6.3f}  len {np.mean(lengths[-10:]):>5.0f}  "
                f"buffer {len(agent.memory):>7}  {elapsed / episode:>5.2f}s/ep",
                end="\r",
                flush=True,
            )

    elapsed = time.perf_counter() - t_start
    agent.save(checkpoint_dir / f"{tag}_final.pt")

    record = {
        "tag": tag,
        "name": cfg.name,
        "algo": cfg.algo,
        "seed": cfg.seed,
        "scores": scores,
        "moving_average": moving,
        "agent_scores": agent_scores,
        "episode_lengths": lengths,
        # solved_episode uses the project's convention (window end minus the
        # window size); episodes_run is the raw count actually executed.
        "solved_episode": solved_episode,
        "episodes_run": None if solved_episode is None else solved_episode + cfg.solve_window,
        "solved": solved_episode is not None,
        "best_moving_average": None if best_moving == -np.inf else float(best_moving),
        "final_moving_average": moving[-1] if moving else None,
        "episodes": len(scores),
        "mean_episode_length": float(np.mean(lengths)) if lengths else None,
        "elapsed_sec": elapsed,
        "sec_per_episode": elapsed / max(1, len(scores)),
        "grad_norm": agent.grad_norm_stats,
        "config": cfg.to_dict(),
    }

    out = results_dir / f"{tag}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if verbose:
        status = (
            f"solved at episode {solved_episode}"
            if solved_episode is not None
            else f"not solved (best avg {record['best_moving_average']})"
        )
        print(f"\n  {tag}: {status} in {elapsed / 60:.1f} min -> {out}")
    return record


def evaluate(
    agent: Agent,
    env: TennisEnv,
    episodes: int = 10,
    add_noise: bool = False,
    max_steps: int = 1000,
) -> dict:
    """Run a trained policy with exploration off, which is how it would be deployed."""
    per_episode: list[float] = []
    lengths: list[int] = []
    reduce = agent.cfg.score_reduce
    for _ in range(episodes):
        states = env.reset(train_mode=True)
        totals = np.zeros(env.num_agents, dtype=np.float64)
        steps = 0
        for _ in range(max_steps):
            actions = agent.act(states, add_noise=add_noise)
            states, rewards, dones, _ = env.step(actions)
            totals += rewards
            steps += 1
            if dones.any():
                break
        per_episode.append(episode_score(totals, reduce))
        lengths.append(steps)
    return {
        "episodes": episodes,
        "mean": float(np.mean(per_episode)),
        "std": float(np.std(per_episode)),
        "min": float(np.min(per_episode)),
        "max": float(np.max(per_episode)),
        "mean_length": float(np.mean(lengths)),
        "scores": per_episode,
    }
