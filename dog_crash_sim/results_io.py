"""Saving and loading simulation results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config import G, load_config


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


MOTION_FIELDS = ["t", "pos", "euler", "vel", "acc_cg_world", "omega", "alpha", "acc_origin", "gravity", "acc_body",
                 "crush_subject", "crush_partner", "interface_force", "intrusion_left", "intrusion_right",
                 "intrusion_rear", "crush_front", "partner_pos", "partner_yaw"]


def save_results(res, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.cfg.save(out / "config.yaml")
    summary = {"name": res.cfg.name, "metrics": res.metrics, "events": res.events, "info": res.info}
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    ch = {k: v for k, v in res.channels.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out / "channels.npz", t=res.t, tendon_names=np.array(res.channels.get("tendon_names", [])), **ch)
    np.savez_compressed(out / "states.npz", t=res.states_t, qpos=res.states_qpos,
                        **{f"extra_{k}": v for k, v in res.states_extra.items()})
    mo = res.motion
    np.savez_compressed(out / "motion.npz", cg_height=mo.cg_height, info=json.dumps(_jsonable(mo.info)),
                        **{f: getattr(mo, f) for f in MOTION_FIELDS})
    _write_csv(res, out / "timeseries.csv")
    return out


def _write_csv(res, path, rate_hz=1000.0):
    t = res.t
    if len(t) < 2:
        return
    step = max(1, int(round(1.0 / (rate_hz * (t[1] - t[0])))))
    C = res.channels
    cols = {"time_ms": t * 1e3}
    mo = res.motion
    idx = np.searchsorted(mo.t, t)
    idx = np.clip(idx, 0, len(mo.t) - 1)
    cols["vehicle_ax_g"] = mo.acc_body[idx, 0] / G
    cols["vehicle_ay_g"] = mo.acc_body[idx, 1] / G
    for nm in ("head_acc", "chest_acc", "pelvis_acc"):
        cols[f"{nm}_res_g"] = np.linalg.norm(C[nm], axis=1) / G
    for j, nm in enumerate(C.get("tendon_names", [])):
        cols[f"{nm}_N"] = C["tendon_tension"][:, j]
    cols["ring_force_N"] = C.get("ring_force", np.zeros(len(t)))
    for lc in ("upper_neck", "lower_neck"):
        cols[f"{lc}_tension_N"] = -C[f"{lc}_force"][:, 2]
        cols[f"{lc}_My_Nm"] = C[f"{lc}_torque"][:, 1]
    cols["thoracolumbar_Fx_N"] = C["thoracolumbar_force"][:, 0]
    for k, ax in enumerate("xyz"):
        cols[f"head_{ax}_m"] = C["head_pos"][:, k]
        cols[f"torso_{ax}_m"] = C["torso_pos"][:, k]
    cols["torso_rotation_deg"] = C["torso_rot_angle"]
    cols["harness_shift_mm"] = C["harness_q"][:, 0] * 1e3
    cols["seat_back_angle_deg"] = np.degrees(C["seat_back_angle"])
    cols["door_left_intrusion_m"] = C["door_left"]
    cols["door_right_intrusion_m"] = C["door_right"]
    cols["throat_force_N"] = C["throat_force"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        keys = list(cols)
        w.writerow(keys)
        for i in range(0, len(t), step):
            w.writerow([f"{cols[k][i]:.6g}" for k in keys])


def load_results(out_dir):
    """Reload a saved run (the scene is rebuilt from the saved configuration)."""
    from .scene.builder import build_scene
    from .vehicle.crash_model import VehicleMotion, euler_to_R
    out = Path(out_dir)
    cfg = load_config(out / "config.yaml")
    summary = json.loads((out / "summary.json").read_text())
    ch = dict(np.load(out / "channels.npz", allow_pickle=False))
    t = ch.pop("t")
    ch["tendon_names"] = [str(x) for x in ch.pop("tendon_names")]
    st = np.load(out / "states.npz")
    extras = {k[6:]: st[k] for k in st.files if k.startswith("extra_")}
    mz = np.load(out / "motion.npz")
    kw = {f: mz[f] for f in MOTION_FIELDS}
    R = np.stack([euler_to_R(e) for e in kw["euler"]])
    motion = VehicleMotion(R=R, cg_height=float(mz["cg_height"]), info=json.loads(str(mz["info"])), **kw)
    scene = build_scene(cfg)
    return SimpleNamespace(cfg=cfg, t=t, channels=ch, states_t=st["t"], states_qpos=st["qpos"], states_extra=extras,
                           events=summary["events"], info=summary["info"], metrics=summary["metrics"], motion=motion,
                           scene=scene)
