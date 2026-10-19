"""Rendering of stored simulation states (video, key frames, interactive replay)."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


class Replayer:
    """Applies a stored state (qpos + vehicle motion) to a model for display."""

    def __init__(self, res):
        self.res = res
        self.m = res.scene.model
        self.d = mujoco.MjData(self.m)
        self.sc = res.scene
        mo = res.motion
        self.mo = mo
        self.rgba0 = {ab["gid"]: ab["rgba"] for ab in self.sc.airbags}

    def apply(self, idx: int) -> float:
        res, m, d, mo = self.res, self.m, self.d, self.mo
        t = float(res.states_t[idx])
        d.qpos[:] = res.states_qpos[idx]
        k = int(np.clip(np.searchsorted(mo.t, t), 0, len(mo.t) - 1))
        R = mo.R[k]
        O_w = mo.pos[k] - R @ np.array([0.0, 0.0, mo.cg_height])
        RT = R.T
        mid = self.sc.mocap
        # road under the car, snapped to the checker period so the texture moves continuously
        period = 4.0
        c_w = np.round(O_w / period) * period
        c_w[2] = 0.0
        d.mocap_pos[mid["road"]] = RT @ (c_w - O_w)
        d.mocap_quat[mid["road"]] = _mat2quat(RT)
        # partner
        pp = mo.partner_pos[k]
        if np.all(np.abs(pp) < 500):
            yaw = mo.partner_yaw[k]
            Rp = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            d.mocap_pos[mid["partner"]] = RT @ (np.array([pp[0], pp[1], 0.0]) - O_w)
            d.mocap_quat[mid["partner"]] = _mat2quat(RT @ Rp)
        else:
            d.mocap_pos[mid["partner"]] = [1e3, 0, -50]
        lay = self.sc.layout
        xb = lay.x_front_bumper - 0.65
        xa = lay.x_rear_bumper + 0.55
        d.mocap_pos[mid["front_end"]] = [0.5 * (xb + lay.x_front_bumper) - mo.crush_front[k], 0, 0]
        rear_crush = mo.crush_subject[k] if self.res.cfg.collision.impact_direction == "rear" else 0.0
        d.mocap_pos[mid["rear_end"]] = [0.5 * (lay.x_rear_bumper + xa) + rear_crush, 0, 0]
        for j, ab in enumerate(self.sc.airbags):
            frac = res.states_extra["airbag"][idx][j] if len(res.states_extra["airbag"]) else 0.0
            rgba = self.rgba0[ab["gid"]].copy()
            rgba[3] = rgba[3] if frac > 0.02 else 0.0
            m.geom_rgba[ab["gid"]] = rgba
        mujoco.mj_forward(m, d)
        return t


def _frame_indices(res, fps: float, slow: float):
    """Indices of stored states for a video with the impact phase slowed down."""
    ts = res.states_t
    out = []
    t = ts[0]
    t_end = ts[-1]
    while t <= t_end:
        out.append(int(np.argmin(np.abs(ts - t))))
        # real time before the impact, slow motion from 50 ms before impact
        factor = slow if t > -0.05 else max(1.0, slow / 4.0)
        t += 1.0 / (fps * factor)
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 8 + 7 * len(text), 22], fill=(0, 0, 0))
    dr.text((5, 5), text, fill=(255, 255, 255))
    return np.asarray(im)


def render_frames(res, camera: str, indices, width: int, height: int, show_skeleton: bool = False):
    rp = Replayer(res)
    r = mujoco.Renderer(rp.m, height, width)
    opt = mujoco.MjvOption()
    if show_skeleton:
        opt.geomgroup[5] = 1
        opt.skingroup[0] = 0
    frames = []
    for idx in indices:
        t = rp.apply(idx)
        r.update_scene(rp.d, camera=camera, scene_option=opt)
        img = r.render()
        frames.append(_label(img, f"{camera}  t = {t * 1e3:7.1f} ms"))
    r.close()
    return frames


def write_video(frames, path: Path, fps: int) -> Path:
    path = Path(path)
    if shutil.which("ffmpeg") and path.suffix == ".mp4":
        h, w = frames[0].shape[:2]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
               str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
        p.wait()
        if p.returncode == 0:
            return path
    from PIL import Image
    gif = path.with_suffix(".gif")
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
    return gif


def render_video(res, out_dir: Path, cameras=None, width=None, height=None, fps=None, slow=None) -> list:
    rc = res.cfg.simulation.render
    cameras = cameras or rc.cameras
    width = width or rc.width
    height = height or rc.height
    fps = fps or rc.fps
    slow = slow or rc.slow_motion
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = _frame_indices(res, fps, slow)
    paths = []
    for cam in cameras:
        frames = render_frames(res, cam, idx, width, height, rc.show_skeleton)
        paths.append(write_video(frames, out_dir / f"video_{cam}.mp4", fps))
    return paths


def render_keyframes(res, out_path: Path, camera: str = "side", times_ms=(-100, 0, 40, 70, 100, 150, 200, 300),
                     width: int = 480, height: int = 360) -> Path:
    from PIL import Image
    ts = res.states_t
    idx = [int(np.argmin(np.abs(ts - tm / 1e3))) for tm in times_ms if ts[0] <= tm / 1e3 <= ts[-1] + 1e-9]
    frames = render_frames(res, camera, idx, width, height)
    cols = 4
    rows = int(math.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * width, rows * height))
    for i, f in enumerate(frames):
        sheet.paste(Image.fromarray(f), ((i % cols) * width, (i // cols) * height))
    out_path = Path(out_path)
    sheet.save(out_path)
    return out_path


def interactive_replay(res, speed: float = 0.1) -> None:  # pragma: no cover - needs a display
    import time
    import mujoco.viewer
    rp = Replayer(res)
    with mujoco.viewer.launch_passive(rp.m, rp.d) as v:
        while v.is_running():
            for idx in range(len(res.states_t)):
                if not v.is_running():
                    break
                t0 = time.time()
                rp.apply(idx)
                v.sync()
                dt_state = (res.states_t[1] - res.states_t[0]) if len(res.states_t) > 1 else 0.002
                time.sleep(max(0.0, dt_state / speed - (time.time() - t0)))
