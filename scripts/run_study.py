"""Run a multi-variant, multi-seed study with a bounded pool of parallel workers.

Each run is a separate process, for three reasons that are specific to this
environment rather than to Python:

* **The Unity log.** Every player opens ``unity-environment.log`` in its own
  working directory and holds it. Two runs sharing a cwd silently serialise --
  one makes progress, the other sits connected and idle with no error and no
  timeout. Each run therefore gets a private ``run_dir``.
* **Ports.** ``worker_id`` maps onto TCP port 5005+id with no collision
  handling, so ids are assigned explicitly and spaced, not left to chance.
* **Thread oversubscription.** Unity's physics and torch's intra-op pool
  compete for the same four cores. Each worker is pinned to a small thread
  count so N runs do not each try to use the whole machine.

Runs are dispatched from a queue rather than in fixed batches, so a fast run
finishing frees its slot immediately.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue


@dataclass
class Run:
    config: Path
    seed: int
    tag: str
    worker_id: int


def build_runs(configs: list[Path], seeds: list[int], worker_stride: int) -> list[Run]:
    runs: list[Run] = []
    wid = 0
    for cfg in configs:
        name = cfg.stem
        for seed in seeds:
            runs.append(Run(cfg, seed, f"{name}_seed{seed}", wid))
            wid += worker_stride
    return runs


def run_one(run: Run, args, log_dir: Path) -> dict:
    log_path = log_dir / f"{run.tag}.log"
    cmd = [
        sys.executable, "-m", "reacher.cli", "train",
        "--config", str(run.config),
        "--seed", str(run.seed),
        "--tag", run.tag,
        "--worker-id", str(run.worker_id),
        "--torch-threads", str(args.threads),
        "--results", args.results,
        "--checkpoints", args.checkpoints,
    ]
    if args.episodes:
        cmd += ["--episodes", str(args.episodes)]

    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=Path.cwd())
    elapsed = time.perf_counter() - t0

    result_file = Path(args.results) / f"{run.tag}.json"
    record = json.loads(result_file.read_text()) if result_file.exists() else None
    return {
        "tag": run.tag,
        "seed": run.seed,
        "config": str(run.config),
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "log": str(log_path),
        "solved": bool(record and record["solved"]),
        "solved_episode": record["solved_episode"] if record else None,
        "best_moving_average": record["best_moving_average"] if record else None,
        "result_file": str(result_file) if record else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", required=True, help="YAML config files")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--parallel", type=int, default=2, help="concurrent runs")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per run")
    ap.add_argument("--episodes", type=int, default=None, help="override config")
    ap.add_argument("--results", default="results/ablation")
    ap.add_argument("--checkpoints", default="checkpoints/ablation")
    ap.add_argument("--worker-stride", type=int, default=4,
                    help="gap between worker ids, so a leaked process cannot collide")
    ap.add_argument("--out", default="results/ablation/study.json")
    args = ap.parse_args()

    configs = [Path(c) for c in args.configs]
    missing = [c for c in configs if not c.exists()]
    if missing:
        raise SystemExit(f"missing config(s): {missing}")

    Path(args.results).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.results) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    runs = build_runs(configs, args.seeds, args.worker_stride)
    print(f"{len(runs)} runs, {args.parallel} at a time, {args.threads} torch threads each")

    queue: Queue[Run] = Queue()
    for r in runs:
        queue.put(r)

    results: list[dict] = []
    lock = threading.Lock()
    started = time.perf_counter()

    def worker() -> None:
        while True:
            try:
                run = queue.get_nowait()
            except Exception:
                return
            with lock:
                print(f"  -> {run.tag} (worker_id {run.worker_id})", flush=True)
            try:
                res = run_one(run, args, log_dir)
            except Exception as exc:  # a crashed run must not sink the study
                res = {"tag": run.tag, "seed": run.seed, "error": repr(exc),
                       "solved": False, "solved_episode": None}
            with lock:
                results.append(res)
                done = len(results)
                status = ("solved @ %s" % res["solved_episode"]) if res.get("solved") else "unsolved"
                print(f"  <- {run.tag}: {status} "
                      f"[{done}/{len(runs)}, {(time.perf_counter() - started) / 60:.0f} min]",
                      flush=True)
            queue.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.parallel)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.perf_counter() - started
    summary = {
        "runs": sorted(results, key=lambda r: r["tag"]),
        "total_runs": len(runs),
        "solved": sum(1 for r in results if r.get("solved")),
        "parallel": args.parallel,
        "threads_per_run": args.threads,
        "elapsed_sec": elapsed,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n{summary['solved']}/{len(runs)} solved in {elapsed / 3600:.2f} h")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
