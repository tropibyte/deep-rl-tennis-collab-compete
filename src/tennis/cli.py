"""Command-line entry point: train, evaluate, or replay a saved policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import Agent
from .config import Config
from .env import TennisEnv
from .train import evaluate, train


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--exe", default=None, help="path to the Tennis build")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--run-dir", default=None,
                   help="private cwd for the Unity log; required for parallel runs")


def cmd_train(args: argparse.Namespace) -> int:
    overrides = {
        k: v
        for k, v in (
            ("seed", args.seed),
            ("episodes", args.episodes),
            ("name", args.name),
            ("torch_threads", args.torch_threads),
        )
        if v is not None
    }
    cfg = (
        Config.from_yaml(args.config, **overrides)
        if args.config
        else Config(**overrides)
    )

    tag = args.tag or f"{cfg.name}_seed{cfg.seed}"
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.results) / "_runs" / tag

    print(f"[{tag}] {cfg.algo} | {cfg.episodes} episodes | seed {cfg.seed}")
    with TennisEnv(
        exe_path=args.exe,
        worker_id=args.worker_id,
        no_graphics=not args.graphics,
        seed=cfg.seed,
        run_dir=run_dir,
    ) as env:
        print(f"[{tag}] {env.num_agents} agents, state {env.state_size}, "
              f"action {env.action_size}")
        cfg.save(Path(args.results) / f"{tag}.config.yaml")
        record = train(
            cfg,
            env,
            checkpoint_dir=args.checkpoints,
            results_dir=args.results,
            tag=tag,
        )
    return 0 if record["solved"] else 1


def cmd_eval(args: argparse.Namespace) -> int:
    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    run_dir = Path(args.run_dir) if args.run_dir else Path("results/_runs/eval")

    with TennisEnv(
        exe_path=args.exe,
        worker_id=args.worker_id,
        no_graphics=not args.graphics,
        seed=args.seed,
        run_dir=run_dir,
    ) as env:
        agent = Agent(env.state_size, env.action_size, env.num_agents, cfg)
        agent.load(args.checkpoint)
        result = evaluate(agent, env, episodes=args.episodes, add_noise=args.noise)

    result["checkpoint"] = str(args.checkpoint)
    print(f"{Path(args.checkpoint).name}: {result['mean']:.2f} +/- {result['std']:.2f} "
          f"over {args.episodes} episodes (min {result['min']:.2f}, max {result['max']:.2f})")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from .record import contact_sheet, record_gif

    print("Screen capture is about to take over the foreground window.")
    print("Close anything you would not want in a published GIF.")
    record_gif(
        args.checkpoint,
        args.out,
        episodes=args.episodes,
        fps=args.fps,
        env_path=args.exe,
        max_frames=args.max_frames,
        scale=args.scale,
        every=args.every,
        frames_dir=args.frames_dir,
        warmup_sec=args.warmup_sec,
    )
    if args.contact_sheet and args.frames_dir:
        contact_sheet(args.frames_dir, args.contact_sheet)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tennis-train", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train an agent")
    _add_common(t)
    t.add_argument("--config", default=None, help="YAML config; omit for defaults")
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--episodes", type=int, default=None)
    t.add_argument("--name", default=None)
    t.add_argument("--tag", default=None, help="output basename; defaults to name_seedN")
    t.add_argument("--torch-threads", type=int, default=None)
    t.add_argument("--checkpoints", default="checkpoints")
    t.add_argument("--results", default="results")
    t.add_argument("--graphics", action="store_true", help="show the Unity window")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="score a saved checkpoint")
    _add_common(e)
    e.add_argument("checkpoint")
    e.add_argument("--episodes", type=int, default=10)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--noise", action="store_true", help="keep exploration noise on")
    e.add_argument("--graphics", action="store_true")
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("record", help="record a GIF of a trained policy")
    _add_common(r)
    r.add_argument("checkpoint")
    r.add_argument("--out", default="assets/trained_agent.gif")
    r.add_argument("--episodes", type=int, default=1)
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--max-frames", type=int, default=450)
    r.add_argument("--every", type=int, default=3, help="capture every Nth step")
    r.add_argument("--scale", type=float, default=0.5)
    r.add_argument("--warmup-sec", type=float, default=3.0,
                   help="settle time after the window is focused, before capturing")
    r.add_argument("--frames-dir", default=None,
                   help="also write every frame as a PNG, for review before publishing")
    r.add_argument("--contact-sheet", default=None,
                   help="tile the captured frames into one reviewable image")
    r.set_defaults(func=cmd_record)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
