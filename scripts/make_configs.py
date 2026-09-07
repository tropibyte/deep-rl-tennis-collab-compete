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
    # Deliberately empty until a Tennis baseline actually solves.
    #
    # On the Reacher project the arms were written against the first baseline,
    # that baseline turned out to be broken, and every arm silently inherited
    # the broken setting. The generator caught it only because a substitution
    # stopped matching. Defining arms before the baseline is known is the same
    # mistake with the safety net removed.
    #
    # The arms this project is likely to want, once there is something to vary:
    #
    #   maddpg_decentralised  centralised_critic: false  -- the headline question
    #   maddpg_nonorm         norm: none                 -- does Reacher's finding transfer?
    #   maddpg_nstep          n_step: 5                  -- best ceiling on Reacher
    #   maddpg_per            per.enabled: true          -- strongest arm on Reacher
    #   td3                   algo: td3
    #   maddpg_gaussian       noise.kind: gaussian
    #   maddpg_nonoisedecay   noise.sigma_decay: 1.0     -- sparse reward needs exploration
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="configs/maddpg.yaml")
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--only", nargs="*", default=None, help="generate a subset")
    args = ap.parse_args()

    base_path = Path(args.base)
    base = base_path.read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = args.only or list(VARIANTS)
    if not wanted:
        raise SystemExit(
            "No variants defined yet. Arms are written once a baseline solves "
            "-- see the note in VARIANTS."
        )
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
