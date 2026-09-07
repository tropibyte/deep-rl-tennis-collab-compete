"""Record a GIF of a trained policy driving the arms.

The vector-observation Reacher build returns no visual observations, so there
are no frames to pull out of the environment. The only way to film it is to run
the Unity window with graphics enabled and screen-capture its client area.

That makes this a privacy problem as much as a graphics one. Screen capture
sees whatever is on the screen, and on the Navigation project it twice caught
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
                    or "Reacher" in title.value
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
    """
    if not _IS_WINDOWS:
        return True
    import ctypes

    return ctypes.windll.user32.GetForegroundWindow() == hwnd


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
    from .env import ReacherEnv

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
    with ReacherEnv(exe_path=env_path, no_graphics=False, seed=0, train_mode=True,
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
                "Click the Reacher window once to focus it, then re-run. "
                "Nothing was captured."
            )
        print(f"capturing region {box} (window confirmed foreground)")

        # Owning the foreground is not the same as having painted. A window that
        # exists but has not yet rendered lets a screen grab pick up whatever is
        # behind it, which is how the first 17 frames of an otherwise clean
        # recording came back showing another application. Wait, then discard
        # grabs until the content stops changing wildly.
        time.sleep(warmup_sec)
        prev_mean = None
        for _ in range(40):
            probe = np.asarray(ImageGrab.grab(bbox=box, all_screens=True).convert("L"),
                               dtype=np.float32)
            m = float(probe.mean())
            # A near-black grab is the splash screen or an unpainted surface.
            if m > 10.0 and prev_mean is not None and abs(m - prev_mean) < 2.0:
                break
            prev_mean = m
            time.sleep(0.25)
        print(f"  warm-up complete (scene brightness {prev_mean:.0f})")

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
                    # Re-check every single frame. Focus can be stolen mid-run
                    # by anything that pops up, and the frames captured after
                    # that point would be of someone else's window.
                    # Never capture while another window is in front. But a
                    # transient focus change should pause the recording, not
                    # destroy it -- the episode keeps running either way, so we
                    # simply skip frames until the window comes back.
                    if not is_foreground(hwnd):
                        if not _paused_notice[0]:
                            print("\n  focus lost - pausing capture, click the "
                                  "Reacher window to resume", flush=True)
                            _paused_notice[0] = True
                        if not _wait_for_foreground(hwnd, timeout=30.0):
                            raise RuntimeError(
                                f"Focus did not return within 30s ({len(frames)} "
                                "frames captured). Nothing is written. Re-run and "
                                "leave the Reacher window in front."
                            )
                        _paused_notice[0] = False
                        print("  focus regained, resuming capture", flush=True)
                        i += 1
                        continue
                    img = ImageGrab.grab(bbox=box, all_screens=True)
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
