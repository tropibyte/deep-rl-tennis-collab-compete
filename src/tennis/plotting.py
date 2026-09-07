"""Figures for the Report.

Deliberately plain matplotlib with no seaborn dependency: these are read as
evidence, not decoration, so the priorities are an honest y-axis, the solve
threshold always drawn, and enough labelling that a figure survives being
pulled out of the document it was written for.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SOLVE_COLOR = "#c44e52"
RAW_COLOR = "#a8c4e0"
AVG_COLOR = "#1f4e79"


def load_records(paths) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]


def learning_curve(record: dict, out: str | Path, title: str | None = None) -> Path:
    """Per-episode score and its 100-episode moving average -- the rubric plot."""
    scores = record["scores"]
    moving = record["moving_average"]
    window = record["config"]["solve_window"]
    threshold = record["config"]["solve_score"]
    episodes = np.arange(1, len(scores) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, scores, color=RAW_COLOR, lw=1.0, label="Score (mean over 20 agents)")

    # The moving average is only meaningful once the window is full; drawing it
    # from episode 1 would show a rising line that is an artefact of the window
    # filling, not of the agent learning.
    if len(moving) >= window:
        ax.plot(episodes[window - 1:], moving[window - 1:], color=AVG_COLOR, lw=2.0,
                label=f"{window}-episode moving average")

    ax.axhline(threshold, color=SOLVE_COLOR, ls="--", lw=1.2,
               label=f"Solve threshold (+{threshold:g})")

    solved = record.get("solved_episode")
    if solved is not None:
        end = solved + window
        ax.axvline(end, color=SOLVE_COLOR, ls=":", lw=1.2)
        ax.annotate(
            f"solved at episode {solved}\n(window ends at {end})",
            xy=(end, threshold),
            xytext=(8, -28),
            textcoords="offset points",
            color=SOLVE_COLOR,
            fontsize=9,
        )

    ax.set_xlabel("Episode #")
    ax.set_ylabel("Score")
    ax.set_title(title or f"{record['name']} (seed {record['seed']})")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def seed_band(records: list[dict], out: str | Path, title: str = "") -> Path:
    """Median and inter-seed range for one variant.

    A single seed's curve says very little in deep RL -- the same code and the
    same hyperparameters can solve in 70 episodes or fail outright. The band is
    the honest summary.
    """
    window = records[0]["config"]["solve_window"]
    threshold = records[0]["config"]["solve_score"]
    n = min(len(r["scores"]) for r in records)
    arr = np.array([r["scores"][:n] for r in records])
    episodes = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(episodes, arr.min(axis=0), arr.max(axis=0),
                    color=RAW_COLOR, alpha=0.5, label=f"range over {len(records)} seeds")
    ax.plot(episodes, np.median(arr, axis=0), color=AVG_COLOR, lw=1.8, label="median")
    ax.axhline(threshold, color=SOLVE_COLOR, ls="--", lw=1.2,
               label=f"Solve threshold (+{threshold:g})")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Score")
    ax.set_title(title or records[0]["name"])
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ablation_curves(groups: dict[str, list[dict]], out: str | Path,
                    title: str = "Variants") -> Path:
    """Median moving-average per variant, all on one axis."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("tab10")
    threshold = None
    window = 100

    for i, (name, records) in enumerate(sorted(groups.items())):
        if not records:
            continue
        window = records[0]["config"]["solve_window"]
        threshold = records[0]["config"]["solve_score"]
        n = min(len(r["moving_average"]) for r in records)
        arr = np.array([r["moving_average"][:n] for r in records])
        episodes = np.arange(1, n + 1)
        med = np.median(arr, axis=0)
        colour = cmap(i % 10)
        ax.plot(episodes[window - 1:], med[window - 1:], lw=1.8, color=colour,
                label=f"{name} (n={len(records)})")
        ax.fill_between(episodes[window - 1:], arr.min(axis=0)[window - 1:],
                        arr.max(axis=0)[window - 1:], color=colour, alpha=0.15)

    if threshold is not None:
        ax.axhline(threshold, color=SOLVE_COLOR, ls="--", lw=1.2,
                   label=f"Solve threshold (+{threshold:g})")
    ax.set_xlabel("Episode #")
    ax.set_ylabel(f"{window}-episode moving average")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def best_average(groups: dict[str, list[dict]], out: str | Path) -> Path:
    """Best 100-episode average reached by each run, with the +30 line drawn.

    Solved/unsolved is a threshold on this quantity, and the threshold hides a
    lot: in this study several runs finished between 27 and 30, which the binary
    reports as an outright failure alongside a run that plateaued at 10. Plotting
    the underlying number keeps a 0.26 miss visually distinct from never getting
    close.
    """
    names = sorted(groups)
    threshold = groups[names[0]][0]["config"]["solve_score"]

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 2.5))
    for row, name in enumerate(names):
        vals = [r["best_moving_average"] for r in groups[name]
                if r["best_moving_average"] is not None]
        solved = [r["best_moving_average"] for r in groups[name]
                  if r["solved"] and r["best_moving_average"] is not None]
        unsolved = [r["best_moving_average"] for r in groups[name]
                    if not r["solved"] and r["best_moving_average"] is not None]
        if solved:
            ax.scatter(solved, [row] * len(solved), s=55, color=AVG_COLOR, zorder=3,
                       label="solved" if row == 0 else None)
        if unsolved:
            ax.scatter(unsolved, [row] * len(unsolved), s=55, facecolors="none",
                       edgecolors=SOLVE_COLOR, zorder=3,
                       label="did not solve" if row == 0 else None)
        if vals:
            ax.plot([min(vals), max(vals)], [row, row], color="#999", lw=1, zorder=2)

    ax.axvline(threshold, color=SOLVE_COLOR, ls="--", lw=1.2,
               label=f"Solve threshold (+{threshold:g})")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Best 100-episode average reached (higher is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def episodes_to_solve(groups: dict[str, list[dict]], out: str | Path,
                      cap: int | None = None) -> Path:
    """Per-variant spread of time-to-solve, with unsolved seeds marked.

    Unsolved runs are drawn at the cap with an open marker rather than dropped:
    silently omitting failures would make an unstable variant look like the
    best one.
    """
    names = sorted(groups)
    if cap is None:
        cap = max((r["episodes"] for rs in groups.values() for r in rs), default=300)

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 2.5))
    for row, name in enumerate(names):
        solved = [r["solved_episode"] for r in groups[name] if r["solved"]]
        unsolved = [r for r in groups[name] if not r["solved"]]
        if solved:
            ax.scatter(solved, [row] * len(solved), s=55, color=AVG_COLOR, zorder=3,
                       label="solved" if row == 0 else None)
            ax.scatter([np.median(solved)], [row], s=140, marker="|",
                       color=SOLVE_COLOR, zorder=4,
                       label="median" if row == 0 else None)
        if unsolved:
            ax.scatter([cap] * len(unsolved), [row] * len(unsolved), s=55,
                       facecolors="none", edgecolors=SOLVE_COLOR, zorder=3,
                       label="did not solve" if row == 0 else None)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Episodes to solve (lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
