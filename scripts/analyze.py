"""Aggregate finished runs into the tables and figures the Report needs.

Two things this deliberately does not do.

It does not report a mean time-to-solve without also reporting the spread. In
deep RL the seed-to-seed variance is frequently larger than the effect being
measured, and a table of means alone will support almost any conclusion the
author already believed.

It does not drop unsolved runs. A variant that solves fast on four seeds and
diverges on the fifth is not better than one that solves steadily on all five,
but that is exactly how it looks once the failure is filtered out. Unsolved
runs are counted, and ranked at the episode cap.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from tennis.plotting import (ablation_curves, best_average, episodes_to_solve,
                              learning_curve, seed_band)


SKIP = {"study.json", "bench.json", "random_baseline.json", "summary.json"}


def load_group(results_dirs: list[Path]) -> dict[str, list[dict]]:
    """Group result files by variant name (every seed of one config together).

    Takes several directories because the baseline and the ablation runs are
    written separately, and the interesting comparisons need them side by side.
    Grouping is by the ``name`` recorded *inside* each file rather than by
    filename, so a run cannot be filed under the wrong variant by a typo in a
    ``--tag``.
    """
    groups: dict[str, list[dict]] = {}
    seen: set[tuple[str, int]] = set()
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("*.json")):
            if path.name in SKIP:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            # Evaluation files also carry a "scores" list, so match on the full
            # training-record signature rather than one field.
            if not all(k in record for k in ("name", "seed", "scores", "moving_average")):
                continue
            key = (record["name"], record["seed"])
            if key in seen:
                # The same variant+seed appearing twice means one directory is
                # a stale copy; counting it twice would silently weight that
                # seed double in every median.
                print(f"  warning: duplicate {key[0]} seed {key[1]}, skipping {path}")
                continue
            seen.add(key)
            groups.setdefault(record["name"], []).append(record)
    return groups


def prob_superiority(a: list[float], b: list[float]) -> float:
    """P(a random run of A beats a random run of B), ties counting as half.

    The Vargha-Delaney A statistic. With three to five seeds per arm a t-test
    is not credible, but "across all seed pairings, how often did A actually
    win?" is a statement the data can support.
    """
    if not a or not b:
        return float("nan")
    wins = sum((x < y) + 0.5 * (x == y) for x, y in itertools.product(a, b))
    return wins / (len(a) * len(b))


def summarise(groups: dict[str, list[dict]], cap: int, baseline: str) -> dict:
    rows = []
    baseline_solve = None

    for name, records in sorted(groups.items()):
        # Unsolved runs are ranked at the cap rather than excluded.
        solve = [
            r["solved_episode"] if r["solved"] else cap
            for r in records
        ]
        solved_only = [r["solved_episode"] for r in records if r["solved"]]
        finals = [r["final_moving_average"] for r in records
                  if r["final_moving_average"] is not None]
        bests = [r["best_moving_average"] for r in records
                 if r["best_moving_average"] is not None]

        row = {
            "name": name,
            "seeds": len(records),
            "solved": sum(1 for r in records if r["solved"]),
            "median_episodes_to_solve": float(np.median(solve)) if solve else None,
            "mean_episodes_to_solve": float(np.mean(solve)) if solve else None,
            "sd_episodes_to_solve": float(np.std(solve)) if len(solve) > 1 else 0.0,
            "fastest_solve": min(solved_only) if solved_only else None,
            "best_moving_average": max(bests) if bests else None,
            "median_final_moving_average": float(np.median(finals)) if finals else None,
            "sec_per_episode": float(np.median([r["sec_per_episode"] for r in records])),
            "_solve_vector": solve,
        }
        rows.append(row)
        if name == baseline:
            baseline_solve = solve

    if baseline_solve is not None:
        for row in rows:
            row["prob_faster_than_baseline"] = prob_superiority(
                row.pop("_solve_vector"), baseline_solve
            )
    else:
        for row in rows:
            row.pop("_solve_vector", None)

    return {"baseline": baseline, "cap": cap, "variants": rows}


def markdown_table(summary: dict) -> str:
    header = (
        "| Variant | Seeds solved | Median episodes to solve | Mean ± SD | "
        "Best 100-ep avg | P(faster than baseline) | s/episode |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in summary["variants"]:
        p = row.get("prob_faster_than_baseline")
        p_str = "—" if p is None or np.isnan(p) else f"{p:.0%}"
        if row["name"] == summary["baseline"]:
            p_str = "— (baseline)"
        med = row["median_episodes_to_solve"]
        med_str = "—" if med is None else f"{med:.0f}"
        best = row["best_moving_average"]
        best_str = "—" if best is None else f"{best:.2f}"
        lines.append(
            f"| `{row['name']}` | {row['solved']}/{row['seeds']} | {med_str} | "
            f"{row['mean_episodes_to_solve']:.0f} ± {row['sd_episodes_to_solve']:.0f} | "
            f"{best_str} | {p_str} | {row['sec_per_episode']:.0f} |"
        )
    return header + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", default=["results", "results/ablation"],
                    help="directories of result JSONs; later ones may add variants")
    ap.add_argument("--out-dir", default="results",
                    help="where summary.json is written")
    ap.add_argument("--assets", default="assets", help="where figures are written")
    ap.add_argument("--baseline", default="maddpg")
    ap.add_argument("--cap", type=int, default=None,
                    help="episodes to rank unsolved runs at; defaults to the longest run")
    args = ap.parse_args()

    results_dirs = [Path(r) for r in args.results if Path(r).is_dir()]
    out_dir = Path(args.out_dir)
    assets = Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = load_group(results_dirs)
    if not groups:
        raise SystemExit(f"no result files in {[str(d) for d in results_dirs]}")

    cap = args.cap or max(r["episodes"] for rs in groups.values() for r in rs)
    print(f"{sum(len(v) for v in groups.values())} runs across {len(groups)} variants")

    # The headline figure: the single best run of the baseline.
    baseline_runs = groups.get(args.baseline, [])
    if baseline_runs:
        best = max(baseline_runs,
                   key=lambda r: (r["solved"], -(r["solved_episode"] or cap)))
        learning_curve(best, assets / "learning_curve.png",
                       title=f"MADDPG on Tennis (2 agents) — seed {best['seed']}")
        print(f"  learning_curve.png <- {best['tag']}")
        if len(baseline_runs) > 1:
            seed_band(baseline_runs, assets / "baseline_seeds.png",
                      title=f"DDPG baseline across {len(baseline_runs)} seeds")
            print("  baseline_seeds.png")

    if len(groups) > 1:
        ablation_curves(groups, assets / "ablation_curves.png",
                        title="Variants: 100-episode moving average")
        episodes_to_solve(groups, assets / "episodes_to_solve.png", cap=cap)
        best_average(groups, assets / "best_average.png")
        print("  ablation_curves.png, episodes_to_solve.png, best_average.png")

    summary = summarise(groups, cap, args.baseline)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = markdown_table(summary)
    (assets / "summary.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"wrote {out_dir / 'summary.json'} and {assets / 'summary.md'}")


if __name__ == "__main__":
    main()
