"""A vectorised gym-style wrapper around the Unity ML-Agents v0.4 Tennis environment.

Two rackets, each controlled by its own agent, hitting a ball over a net. Unlike
Reacher's 20 independent arms, these two agents *interact*: one's action changes
what the other observes on the next step. They still share this wrapper's shape
-- ``(2, 24)`` states in, ``(2, 2)`` actions out -- and the training code decides
whether they also share a policy.

Two differences from Reacher that matter more than they look:

* **Episodes have no fixed length.** A Reacher episode is always 1001 steps; a
  Tennis episode ends when the ball hits the ground or goes out. Early training
  gives episodes of a dozen steps, and a competent agent produces long rallies,
  so wall-clock per episode *grows* as learning succeeds. Throughput measured on
  a random policy badly underestimates the cost of a solved one.
* **The observation is already stacked.** Each agent sees 3 frames of 8
  variables concatenated into 24, so velocity information is present without any
  frame-stacking here.

Carried over from the Navigation and Reacher projects (see docs/PORTING.md),
because these sharp edges belong to the v0.4 client rather than to any one
environment:

1. **Leaked processes.** The gRPC server thread is non-daemon, so an exception
   escaping before ``close()`` hangs Python forever holding a live
   ``Tennis.exe``. This is a context manager and also closes on ``__del__``.
2. **Port collisions.** ``worker_id`` maps straight onto TCP port 5005+id with
   no collision handling. We probe for a free port by *connecting* -- see
   ``_port_is_free``.
3. **The shared log file.** Every Unity player opens ``unity-environment.log``
   in its process working directory and holds it. Parallel runs sharing a cwd
   silently serialise: one makes progress, the rest sit connected and idle with
   no error and no timeout. ``run_dir`` gives each run a private cwd.
"""
from __future__ import annotations

import contextlib
import os
import socket
import sys
from pathlib import Path

import numpy as np

# The vendored ml-agents client lives in vendor/ so it can be pip-installed as a
# top-level package; when running from a source checkout without installing,
# put it on the path here.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR = _REPO_ROOT / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from unityagents import UnityEnvironment  # noqa: E402

BASE_PORT = 5005

_CANDIDATE_EXES = [
    "Tennis_Windows_x86_64/Tennis.exe",
    "Tennis_Windows_x86/Tennis.exe",
    "Tennis_Linux/Tennis.x86_64",
    "Tennis_Linux_NoVis/Tennis.x86_64",
    "Tennis.app",
]


def find_executable(explicit: str | None = None) -> str:
    """Locate the Unity build, preferring an explicit path or $TENNIS_ENV_PATH."""
    for cand in filter(None, [explicit, os.environ.get("TENNIS_ENV_PATH")]):
        if Path(cand).exists():
            return str(cand)
        raise FileNotFoundError(f"Unity environment not found at: {cand}")
    for rel in _CANDIDATE_EXES:
        p = _REPO_ROOT / rel
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "Could not find the Tennis Unity environment. Download it (see README), "
        "unzip it into the repo root, or set TENNIS_ENV_PATH."
    )


def _port_is_free(port: int) -> bool:
    """True if nothing is listening on ``port``.

    Deliberately probes by *connecting*, not by binding. Two Windows-specific
    traps make the obvious bind-test silently useless:

    * ``SO_REUSEADDR`` on Windows permits binding a port that is already bound
      (the opposite of its BSD/Linux meaning), so the bind always "succeeds".
    * gRPC binds ``[::]`` (IPv6 dual-stack) while a naive probe binds IPv4
      ``0.0.0.0``; on Windows those need not collide, so the probe passes and
      the gRPC server then dies with WSAEADDRINUSE.

    A successful connect is unambiguous proof that something is listening.
    """
    for family, addr in ((socket.AF_INET, ("127.0.0.1", port)),
                         (socket.AF_INET6, ("::1", port))):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex(addr) == 0:
                    return False
        except OSError:
            continue
    return True


def pick_worker_id(preferred: int = 0, span: int = 64) -> int:
    """First worker_id whose port is actually free, starting at ``preferred``."""
    for offset in range(span):
        wid = preferred + offset
        if _port_is_free(BASE_PORT + wid):
            return wid
    raise RuntimeError(
        f"No free port in {BASE_PORT + preferred}..{BASE_PORT + preferred + span}"
    )


@contextlib.contextmanager
def _chdir(path: Path | None):
    """Temporarily run in ``path`` so the Unity log lands there. See (3) above."""
    if path is None:
        yield
        return
    path.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class TennisEnv:
    """Vectorised view of the Tennis environment.

    ``num_agents`` is read back from the environment rather than assumed, so
    nothing here hard-codes 2 and the wrapper would not silently mis-shape
    its arrays if the build changed.
    """

    def __init__(
        self,
        exe_path: str | None = None,
        worker_id: int | None = None,
        no_graphics: bool = True,
        seed: int = 0,
        train_mode: bool = True,
        run_dir: str | Path | None = None,
    ):
        self.exe_path = str(Path(find_executable(exe_path)).resolve())
        self.worker_id = pick_worker_id(0 if worker_id is None else worker_id)
        self.train_mode = train_mode
        self._closed = False

        # Unity opens its log in the *current* working directory and holds it,
        # so a run with a private run_dir must be launched from inside it.
        with _chdir(Path(run_dir) if run_dir else None):
            self._env = UnityEnvironment(
                file_name=self.exe_path,
                worker_id=self.worker_id,
                no_graphics=no_graphics,
                seed=seed,
            )

        self.brain_name = self._env.brain_names[0]
        brain = self._env.brains[self.brain_name]
        self.action_size = int(brain.vector_action_space_size)
        self.action_type = brain.vector_action_space_type  # 'continuous'
        info = self._env.reset(train_mode=train_mode)[self.brain_name]
        states = np.asarray(info.vector_observations, dtype=np.float32)
        self.num_agents = int(states.shape[0])
        self.state_size = int(states.shape[1])

    # -- gym-style API ---------------------------------------------------
    def reset(self, train_mode: bool | None = None) -> np.ndarray:
        """Return states, shape ``(num_agents, state_size)``."""
        mode = self.train_mode if train_mode is None else train_mode
        info = self._env.reset(train_mode=mode)[self.brain_name]
        return np.asarray(info.vector_observations, dtype=np.float32)

    def step(self, actions: np.ndarray):
        """Apply torques, shape ``(num_agents, action_size)``.

        Every entry of the action vector must lie in [-1, 1]; Unity does not
        validate this and out-of-range torques quietly distort the dynamics, so
        we clip here rather than trusting each caller to remember.
        """
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        actions = actions.reshape(self.num_agents, self.action_size)
        info = self._env.step(actions)[self.brain_name]
        states = np.asarray(info.vector_observations, dtype=np.float32)
        rewards = np.asarray(info.rewards, dtype=np.float32)
        dones = np.asarray(info.local_done, dtype=bool)
        return states, rewards, dones, {}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._env.close()
            except Exception:
                pass

    # -- lifecycle safety ------------------------------------------------
    def __enter__(self) -> "TennisEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
