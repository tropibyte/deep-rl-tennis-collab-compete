"""Build the Report's figures from the reconstructed run records.

Written separately from the Reacher project's plotting module because these
records carry an explicit ``episode_index``: the runs were stopped early once
they had solved decisively, and their first 100 episodes were logged every
tenth. Plotting against position in the list rather than against the recorded
episode number would silently compress the early part of every curve.

The third figure has no Reacher equivalent. Rally length is the mechanism behind
the score on this task -- the reward pays for keeping the ball in play -- so
plotting it shows *why* the score rose, and explains why the run got slower as
it improved.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAW = "#a8c4e0"
AVG = "#1f4e79"
SOLVE = "#c44e52"
SEEDC = ["#1f4e79", "#2e8b57", "#b8860b"]


def load(seed: int) -> dict:
    return json.loads(Path(f"results/maddpg_seed{seed}.json").read_text(encoding="utf-8"))


def learning_curve(rec: dict, out: Path) -> None:
    eps = np.array(rec["episode_index"])
    scores = np.array(rec["scores"])
    moving = np.array(rec["moving_average"])
    window = rec["config"]["solve_window"]
    threshold = rec["config"]["solve_score"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eps, scores, color=RAW, lw=0.9, label="Score (max over the 2 agents)")
    full = eps >= window
    ax.plot(eps[full], moving[full], color=AVG, lw=2.0,
            label=f"{window}-episode moving average")
    ax.axhline(threshold, color=SOLVE, ls="--", lw=1.2,
               label=f"Solve threshold (+{threshold})")

    solved = rec.get("solved_episode")
    if solved is not None:
        end = solved + window
        ax.axvline(end, color=SOLVE, ls=":", lw=1.2)
        ax.annotate(f"solved at episode {solved}\n(window ends at {end})",
                    xy=(end, threshold), xytext=(10, 24), textcoords="offset points",
                    color=SOLVE, fontsize=9)

    ax.set_xlabel("Episode #")
    ax.set_ylabel("Score")
    ax.set_title(f"MADDPG on Tennis — seed {rec['seed']}")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def all_seeds(recs: list[dict], out: Path) -> None:
    window = recs[0]["config"]["solve_window"]
    threshold = recs[0]["config"]["solve_score"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, rec in enumerate(recs):
        eps = np.array(rec["episode_index"])
        moving = np.array(rec["moving_average"])
        full = eps >= window
        label = f"seed {rec['seed']}"
        if rec.get("solved_episode") is not None:
            label += f" — solved @ {rec['solved_episode']}"
        ax.plot(eps[full], moving[full], lw=1.8, color=SEEDC[k % 3], label=label)
    ax.axhline(threshold, color=SOLVE, ls="--", lw=1.2,
               label=f"Solve threshold (+{threshold})")
    ax.set_xlabel("Episode #")
    ax.set_ylabel(f"{window}-episode moving average")
    ax.set_title("MADDPG on Tennis — three seeds")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def rally_length(recs: list[dict], out: Path) -> None:
    """Why the score rose, and why the run slowed down."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, rec in enumerate(recs):
        eps = np.array(rec["episode_index"])
        lengths = np.array(rec["episode_lengths"])
        ax.plot(eps, lengths, lw=1.2, color=SEEDC[k % 3], label=f"seed {rec['seed']}")
    ax.axhline(17.4, color="#888", ls=":", lw=1.2, label="random policy (17.4 steps)")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Steps per episode (10-episode mean)")
    ax.set_title("Rally length — the mechanism behind the score, and the cost")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def main() -> None:
    assets = Path("assets")
    assets.mkdir(exist_ok=True)
    recs = [load(s) for s in (0, 1, 2)]
    best = min((r for r in recs if r["solved"]), key=lambda r: r["solved_episode"])
    learning_curve(best, assets / "learning_curve.png")
    all_seeds(recs, assets / "all_seeds.png")
    rally_length(recs, assets / "rally_length.png")


if __name__ == "__main__":
    main()
