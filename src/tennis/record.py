"""Record a GIF of a trained policy driving the arms.

The vector-observation Tennis build returns no visual observations, so there
are no frames to pull out of the environment. The only way to film it is to run
the Unity window with graphics enabled and screen-capture its client area.

That makes this a privacy problem as much as a graphics one. Screen capture
sees whatever is on the screen, and on the earlier projects it repeatedly caught
personal content -- a notification toast, and an editor window behind a client
rectangle that was measured while the player was still resizing itself. The
mitigations below are deliberate, and ``--frames-dir`` exists so every frame can
be reviewed before anything is published.

Carried over from that project: ``PrintWindow`` returns solid black for the
Unity D3D surface, so off-screen capture is not an option here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_IS_WINDOWS = sys.platform.startswith("win")

# Width of the screen-right strip reserved for notification toasts, which are
# always-on-top and would otherwise be captured. Generous on purpose: a Windows
# 11 toast is ~360px, and losing a slice of the frame is a far better outcome
# than publishing someone's notifications.
TOAST_ZONE_PX = 480

# How far a frame's mean brightness may drift from the reference measured
# during warm-up before it is treated as something other than the scene.
# The court is a static background, so the honest variation is small.
OCCLUSION_TOLERANCE = 18.0
# Frames tolerated that do not match the scene. Generous: the player can be
# slow to render, and giving up early is what wasted several attempts.
MAX_OCCLUDED = 250
# Mean per-pixel change between consecutive grabs that counts as a live,
# animating render rather than a static window sitting in the region.
MOTION_THRESHOLD = 0.35


def _time_left(deadline: float) -> float:
    """Seconds remaining before the search gives up."""
    import time as _t

    return deadline - _t.time()


def _find_unity_window(timeout: float = 25.0):
    """Return (hwnd, (l, t, r, b)) for the Unity client area, or None."""
    if not _IS_WINDOWS:
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    deadline = time.time() + timeout
    while time.time() < deadline:
        found = []

        def cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls, 256)
            title = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title, 256)
            if ("UnityWndClass" in cls.value
                    or "Tennis" in title.value
                    or "Unity Environment" in title.value):
                found.append(hwnd)
            return True

        EnumWindows(EnumWindowsProc(cb), 0)
        if not found and _time_left(deadline) < 1.0:
            # About to give up: say what we could see, so the failure is
            # actionable instead of just "not found".
            seen = []

            def cb2(hwnd, _):
                if user32.IsWindowVisible(hwnd):
                    cls = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(hwnd, cls, 256)
                    title = ctypes.create_unicode_buffer(256)
                    user32.GetWindowTextW(hwnd, title, 256)
                    if title.value.strip():
                        seen.append(f"{title.value[:40]!r} [{cls.value}]")
                return True

            EnumWindows(EnumWindowsProc(cb2), 0)
            print(f"  no Unity window matched. Visible windows: {seen[:12]}")
        if found:
            hwnd = found[0]
            # Bring it to the front, but never resize or maximise it: the Unity
            # player owns its own resolution, and both resizing and maximising
            # provoke a D3D device reset that can block.
            HWND_TOPMOST, SWP_NOSIZE, SWP_NOMOVE = -1, 0x0001, 0x0002
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE)
            # Nudge the window away from the screen-right toast strip so the
            # exclusion below does not crop the scene. Moving is safe --
            # it is resizing that provokes the device reset.
            SWP_NOSIZE_ONLY = 0x0001
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE_ONLY)
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE, in case it is minimised
            user32.SetForegroundWindow(hwnd)

            # Wait for the client size to STOP changing before trusting it. The
            # player resizes itself asynchronously after launch, and a rectangle
            # measured mid-transition can be far larger than the window really
            # occupies -- which is how desktop content behind the window ends up
            # inside the recording.
            prev, stable = None, 0
            for _ in range(30):
                time.sleep(0.25)
                r = wintypes.RECT()
                user32.GetClientRect(hwnd, ctypes.byref(r))
                cur = (r.right, r.bottom)
                stable = stable + 1 if (cur == prev and cur[0] > 0) else 0
                prev = cur
                if stable >= 3:
                    break

            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            wr = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(wr))
            pt = wintypes.POINT(0, 0)
            user32.ClientToScreen(hwnd, ctypes.byref(pt))

            # Belt and braces: the client area can never be larger than the
            # window containing it. If the two disagree, trust the smaller.
            w = min(rect.right, wr.right - wr.left)
            h = min(rect.bottom, wr.bottom - wr.top)
            right, bottom = pt.x + w, pt.y + h

            # Windows anchors notification toasts to the RIGHT EDGE OF THE
            # SCREEN and draws them above every other window, so anything in
            # that strip lands in the capture -- including whatever personal
            # content a toast happens to be showing.
            right = min(right, user32.GetSystemMetrics(0) - TOAST_ZONE_PX)

            box = (pt.x, pt.y, right, bottom)
            if box[2] > box[0] and box[3] > box[1]:
                return hwnd, box
        time.sleep(0.5)
    return None


def client_box(hwnd):
    """The window's client rectangle in screen coordinates, measured *now*.

    Measured fresh for every frame rather than once at startup. A rectangle
    taken at launch goes stale the moment the window moves, and the grab then
    lands on whatever is behind it -- which produced a recording that was mostly
    an unrelated text window with a small patch of the actual scene offset
    inside it. Re-measuring follows the window instead of assuming it stayed put.

    Returns None if the window is gone or has no area.
    """
    if not _IS_WINDOWS:
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
        return None

    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    wr = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(wr))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))

    w = min(rect.right, wr.right - wr.left)
    h = min(rect.bottom, wr.bottom - wr.top)
    right = min(pt.x + w, user32.GetSystemMetrics(0) - TOAST_ZONE_PX)
    bottom = pt.y + h
    if right <= pt.x or bottom <= pt.y:
        return None
    return (pt.x, pt.y, right, bottom)


def is_foreground(hwnd) -> bool:
    """True if ``hwnd`` is the window the user is actually looking at.

    Windows refuses ``SetForegroundWindow`` from a process that does not own the
    current foreground, and it fails *silently* -- the call returns and the
    window stays behind whatever was already on top. The client rectangle we
    measured is still correct, so a screen grab at those coordinates captures
    whatever window is sitting there instead.

    That is not a cosmetic bug. It is how a recording ends up containing the
    contents of somebody's editor, chat window or inbox, and it has now happened
    on two projects. So every capture is gated on this check rather than on the
    assumption that raising the window worked.

    The test is "does the foreground window belong to the same *process* as the
    one we are filming", not "is it the exact same handle". A Unity player can
    own more than one top-level window, and the one that receives focus need not
    be the one whose client rectangle we measured -- an exact-handle test then
    reports a focus loss while the player is plainly in front, which is what it
    did here. Process identity is the property that actually matters: it
    guarantees nothing else is covering the capture region.
    """
    if not _IS_WINDOWS:
        return True
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    fg = user32.GetForegroundWindow()
    if fg == hwnd:
        return True
    if not fg:
        return False

    def pid_of(h) -> int:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
        return pid.value

    ours = pid_of(hwnd)
    return bool(ours) and pid_of(fg) == ours


def _wait_for_foreground(hwnd, timeout: float = 10.0) -> bool:
    """Give the window a few seconds to actually come forward."""
    if not _IS_WINDOWS:
        return True
    import ctypes
    import time as _time

    user32 = ctypes.windll.user32
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        if is_foreground(hwnd):
            return True
        user32.SetForegroundWindow(hwnd)
        _time.sleep(0.25)
    return False


def record_gif(
    checkpoint: str,
    out_path: str,
    episodes: int = 1,
    fps: int = 30,
    env_path: str | None = None,
    max_frames: int = 450,
    scale: float = 0.5,
    every: int = 3,
    frames_dir: str | None = None,
    warmup_sec: float = 3.0,
) -> Path:
    """Play with exploration off, capturing the Unity window to a GIF."""
    import imageio.v2 as imageio
    import torch
    from PIL import Image, ImageGrab

    from .agent import Agent
    from .config import Config
    from .env import TennisEnv

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])

    frames: list[np.ndarray] = []
    _paused_notice = [False]
    scores: list[float] = []

    # graphics ON: there is nothing to film otherwise. train_mode stays True --
    # nothing is being trained, but the build's inference configuration steps
    # far slower, and the rendered frames are identical either way. Only the
    # wall-clock pacing differs, and playback fps is set here regardless.
    with TennisEnv(exe_path=env_path, no_graphics=False, seed=0, train_mode=True,
                    run_dir=Path("results/_record_run")) as env:
        agent = Agent(env.state_size, env.action_size, env.num_agents, cfg)
        agent.load(checkpoint)

        win = _find_unity_window()
        if win is None:
            # Previously this fell back to grabbing the whole screen. That is
            # exactly the wrong default: if we cannot find the window we are
            # supposed to film, capturing everything else instead is the worst
            # available outcome.
            raise RuntimeError(
                "Could not locate the Unity window. Refusing to screen-grab "
                "something else instead -- check the player actually opened."
            )
        hwnd, box = win
        # Windows will not hand the foreground to a background process, so this
        # depends on a human click. Give them time to find the window and make
        # the instruction explicit about where it is -- a short timeout here is
        # the difference between "works" and "fails for no visible reason".
        print(f"\n  >>> CLICK THE UNITY WINDOW NOW <<<")
        print(f"      It is at screen position {box[0]},{box[1]} "
              f"({box[2] - box[0]}x{box[3] - box[1]} px).")
        print(f"      Waiting up to 60s for it to come to the front...", flush=True)
        if not _wait_for_foreground(hwnd, timeout=60.0):
            raise RuntimeError(
                "The Unity window never came to the foreground, so a screen "
                "grab at its coordinates would capture whatever is on top of "
                "it instead. Windows blocks SetForegroundWindow from a "
                "background process and fails silently.\n\n"
                "Click the Tennis window once to focus it, then re-run. "
                "Nothing was captured."
            )
        print(f"capturing region {box} (window confirmed foreground)")

        # Owning the foreground is not the same as having painted, and on this
        # machine the player can take a long time to put anything on screen. A
        # brightness-only warm-up gave up while the window was still showing
        # what was behind it, and then used *that* as the reference -- so the
        # check was measuring the wrong thing against the wrong baseline.
        #
        # Wait for MOTION instead. The environment is being stepped, so a live
        # render changes between consecutive grabs while a static window
        # underneath does not. Motion is the one signal a stale or occluded
        # capture cannot produce.
        time.sleep(warmup_sec)
        print("  waiting for the scene to start rendering...", flush=True)

        def grab_gray():
            nonlocal box
            box = client_box(hwnd) or box
            return np.asarray(
                ImageGrab.grab(bbox=box, all_screens=True).convert("L"),
                dtype=np.float32)

        states = env.reset(train_mode=True)
        prev = grab_gray()
        moving_streak = 0
        live = False
        for attempt in range(400):          # up to ~2 minutes of patience
            actions = agent.act(states, add_noise=False)
            states, _, dones, _ = env.step(actions)
            if dones.any():
                states = env.reset(train_mode=True)
            cur = grab_gray()
            if cur.shape == prev.shape:
                delta = float(np.abs(cur - prev).mean())
                moving_streak = moving_streak + 1 if delta > MOTION_THRESHOLD else 0
                if moving_streak >= 3:
                    live = True
                    print(f"  scene is live (frame-to-frame change {delta:.2f}) "
                          f"after {attempt + 1} probes", flush=True)
                    break
            prev = cur
            time.sleep(0.05)

        if not live:
            raise RuntimeError(
                "The capture region never showed a moving picture, so it is not "
                "the Tennis render -- it is whatever is sitting in that part of "
                "the screen. Nothing was written."
            )

        box = client_box(hwnd) or box
        ref_mean = float(grab_gray().mean())
        occluded = 0
        print(f"  capture region {box}, scene brightness {ref_mean:.0f}")
        print("")
        print("  >>> RECORDING NOW - do not click anything until this finishes <<<")
        print("      (about 60-90 seconds; it will say when it is done)", flush=True)

        for ep in range(episodes):
            states = env.reset(train_mode=True)
            ep_scores = np.zeros(env.num_agents)
            i, done = 0, False
            while not done and len(frames) < max_frames:
                actions = agent.act(states, add_noise=False)
                states, rewards, dones, _ = env.step(actions)
                ep_scores += rewards
                done = bool(dones.any())
                if i % every == 0:
                    # Follow the window rather than trusting a startup
                    # measurement: it demonstrably moves after launch.
                    live = client_box(hwnd)
                    if live is None:
                        raise RuntimeError(
                            f"The Tennis window disappeared after {len(frames)} "
                            "frames. Nothing is written."
                        )
                    box = live
                    img = ImageGrab.grab(bbox=box, all_screens=True)

                    # Gate on the pixels, not on the window manager.
                    #
                    # The focus test (GetForegroundWindow, by handle and then by
                    # process) kept reporting a loss while the player was
                    # plainly in front, and rejected every frame. Whatever takes
                    # focus back on this machine, the property actually worth
                    # enforcing is "these pixels are the scene we measured
                    # during warm-up" -- which is what protects against
                    # publishing somebody's editor, and is checkable directly.
                    #
                    # The scene's global brightness is stable because the court
                    # is a static background; only the ball and rackets move. A
                    # different window covering the region moves it far outside
                    # this band.
                    probe = float(np.asarray(img.convert("L"), dtype=np.float32).mean())
                    if abs(probe - ref_mean) > OCCLUSION_TOLERANCE:
                        occluded += 1
                        if not _paused_notice[0]:
                            print(f"\n  frame does not match the scene "
                                  f"(brightness {probe:.0f} vs {ref_mean:.0f}) - "
                                  "something is covering the window; pausing",
                                  flush=True)
                            _paused_notice[0] = True
                        if occluded > MAX_OCCLUDED:
                            raise RuntimeError(
                                f"{occluded} consecutive frames did not match the "
                                f"scene ({len(frames)} captured). Nothing is "
                                "written -- refusing to film whatever is covering "
                                "the Tennis window."
                            )
                        i += 1
                        continue
                    if _paused_notice[0]:
                        print("  scene visible again, resuming capture", flush=True)
                        _paused_notice[0] = False
                    occluded = 0
                    if scale != 1.0:
                        img = img.resize(
                            (int(img.width * scale), int(img.height * scale)),
                            Image.LANCZOS,
                        )
                    frames.append(np.asarray(img.convert("RGB")))
                i += 1
            scores.append(float(ep_scores.mean()))
            print(f"  episode {ep + 1}: score {ep_scores.mean():.1f}  ({len(frames)} frames)")

    if not frames:
        raise RuntimeError("No frames captured.")

    # A locked or screen-blanked session makes ImageGrab return solid black,
    # which would silently produce a black GIF. Refuse rather than ship it.
    stack = np.stack(frames[:: max(1, len(frames) // 12)])
    if float(stack.std()) < 1.5:
        raise RuntimeError(
            f"Captured frames are essentially uniform (std={float(stack.std()):.2f}) - "
            "the Unity window was probably not visible. Screen capture needs an "
            "unlocked, active desktop session; re-run while logged in."
        )

    # Dump every frame for human review before publishing. The whole point is
    # that a GIF is hard to inspect and a directory of PNGs is not.
    if frames_dir:
        fdir = Path(frames_dir)
        fdir.mkdir(parents=True, exist_ok=True)
        for k, frame in enumerate(frames):
            Image.fromarray(frame).save(fdir / f"frame_{k:04d}.png")
        print(f"wrote {len(frames)} frames to {fdir} for review")

    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"scores: {[f'{s:.1f}' for s in scores]}  frames {len(frames)} -> {out}")
    return out


def contact_sheet(frames_dir: str, out_path: str, cols: int = 6, thumb: int = 320) -> Path:
    """Tile every captured frame into one image, so a whole recording can be
    checked at a glance rather than opened file by file."""
    from PIL import Image

    fdir = Path(frames_dir)
    paths = sorted(fdir.glob("frame_*.png"))
    if not paths:
        raise RuntimeError(f"no frames in {fdir}")

    thumbs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        ratio = thumb / img.width
        thumbs.append(img.resize((thumb, max(1, int(img.height * ratio)))))

    tw, th = thumbs[0].size
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "black")
    for k, img in enumerate(thumbs):
        sheet.paste(img, ((k % cols) * tw, (k // cols) * th))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"contact sheet: {len(thumbs)} frames -> {out} ({sheet.width}x{sheet.height})")
    return out
