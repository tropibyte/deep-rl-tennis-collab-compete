"""Run configuration: one dataclass, loadable from YAML, saved beside every result.

Every knob an experiment can turn lives here and nowhere else, so each run
writes its resolved config next to its scores and any curve can be traced back
to the settings that produced it -- including defaults never mentioned in the
YAML file.

Defaults carry two findings from the Reacher project rather than re-deriving
them. ``norm`` is ``layer``, because unnormalised raw Unity observations drove
that actor's ``tanh`` into saturation and killed two seeds in three; and the
noise process draws from a standard normal rather than the reference
implementation's strictly-positive uniform.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class NoiseConfig:
    kind: str = "ou"              # ou | gaussian | none
    sigma: float = 0.2
    sigma_min: float = 0.01
    # Tennis rewards are sparse until an agent can return the ball at all, so
    # exploration has to stay alive far longer than in Reacher -- but it must
    # eventually get out of the way, because the scoring rewards long rallies
    # that noisy actions keep interrupting.
    sigma_decay: float = 0.999    # per episode
    theta: float = 0.15
    mu: float = 0.0


@dataclass
class PERConfig:
    enabled: bool = False
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100_000


@dataclass
class TD3Config:
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5


@dataclass
class Config:
    # -- identity --------------------------------------------------------
    name: str = "maddpg"
    algo: str = "ddpg"            # ddpg | td3
    seed: int = 0

    # -- multi-agent -----------------------------------------------------
    # The two rackets face a symmetric task, so one policy can drive both and
    # every episode yields two trajectories for it. That is self-play, and it
    # is the cheapest thing that can work. `shared_policy: false` gives each
    # agent its own actor and critic instead.
    shared_policy: bool = True
    # A centralised critic sees both agents' observations and actions, which is
    # what makes MADDPG stable in a non-stationary two-agent setting: each
    # agent's environment includes the other's changing policy, and a critic
    # that cannot see it is regressing on a moving target it cannot explain.
    centralised_critic: bool = True

    # -- training loop ---------------------------------------------------
    # Tennis is slow to start and then takes off sharply; budgets under ~1500
    # episodes routinely stop during the flat period and report failure.
    episodes: int = 2500
    max_steps: int = 1000         # episodes end on their own when the ball drops
    update_every: int = 1
    updates_per_cycle: int = 1
    learn_start: int = 1024

    # -- agent -----------------------------------------------------------
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.01             # faster target tracking than Reacher's 1e-3
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    weight_decay: float = 0.0
    n_step: int = 1

    grad_clip_critic: float | None = 1.0
    grad_clip_actor: float | None = None

    # -- networks --------------------------------------------------------
    hidden_actor: tuple[int, int] = (256, 128)
    hidden_critic: tuple[int, int] = (256, 128)
    # Layer norm, not none: see the Reacher report. This is the setting that
    # decided whether that project solved at all.
    norm: str = "layer"
    input_norm: str = "none"

    # -- sub-configs -----------------------------------------------------
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    per: PERConfig = field(default_factory=PERConfig)
    td3: TD3Config = field(default_factory=TD3Config)

    # -- solve criterion -------------------------------------------------
    # Tennis scores an episode by the MAXIMUM of the two agents' undiscounted
    # totals, not the mean. That is the project's definition and it matters:
    # the mean would halve every score and the environment would never register
    # as solved.
    score_reduce: str = "max"     # max | mean
    solve_score: float = 0.5
    solve_window: int = 100

    # -- runtime ---------------------------------------------------------
    torch_threads: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.hidden_actor = tuple(self.hidden_actor)
        self.hidden_critic = tuple(self.hidden_critic)
        if self.algo not in ("ddpg", "td3"):
            raise ValueError(f"unknown algo: {self.algo!r}")
        if self.score_reduce not in ("max", "mean"):
            raise ValueError(f"unknown score_reduce: {self.score_reduce!r}")
        if self.n_step < 1:
            raise ValueError("n_step must be >= 1")
        if self.centralised_critic and not self.shared_policy:
            # Supportable, but it is not what this code implements, and silently
            # doing something else would be worse than refusing.
            raise ValueError(
                "centralised_critic currently requires shared_policy=True"
            )

    # -- serialisation ---------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw.update(overrides)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = dict(raw)
        sub = {"noise": NoiseConfig, "per": PERConfig, "td3": TD3Config}
        kwargs: dict[str, Any] = {}
        for key, klass in sub.items():
            kwargs[key] = klass(**(raw.pop(key, None) or {}))
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        kwargs.update(raw)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
