"""Occupant simulation (stage 2): the dog, its harness and the cargo in the crashing vehicle.

The MuJoCo world is the vehicle frame.  Because that frame accelerates and
rotates with the vehicle, the simulation adds the inertial (fictitious)
loads at every step:

* translation: the gravity vector is replaced by ``g_v - a_O`` (gravity in
  the vehicle frame minus the acceleration of the frame origin), which acts
  on every body exactly like the crash pulse does in reality;
* rotation: Euler, centrifugal and Coriolis forces and the matching torques
  are applied to every free/jointed body through ``xfrc_applied``.

Accelerations reported for injury assessment are converted back to the
inertial frame, so they are what a real accelerometer would measure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mujoco
import numpy as np

from .config import G, ScenarioConfig
from .biophysics.neuromuscular import NeuromuscularController
from .restraints.models import AirbagState, EnergyAbsorber, Retractor, SeatBack, Webbing, first_crossing
from .scene.builder import DOG_PARTS, VEH_PARTS, Scene, build_scene
from .vehicle.crash_model import VehicleMotion, run_vehicle_model

GROUPS = {
    "neck": ("C_", "atlas"),
    "spine": ("L_",),
    "forelimb": ("scapula", "shoulder", "elbow", "wrist", "finger"),
    "hindlimb": ("hip", "knee", "ankle", "toe"),
}


@dataclass
class Results:
    cfg: ScenarioConfig
    t: np.ndarray
    channels: dict
    states_t: np.ndarray
    states_qpos: np.ndarray
    states_extra: dict
    events: dict
    info: dict
    motion: VehicleMotion
    scene: Scene
    metrics: dict = field(default_factory=dict)


def vcross(a, b):
    """Fast cross product for (3,) and (n, 3) arrays (np.cross is slow for small inputs)."""
    a = np.asarray(a)
    b = np.asarray(b)
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


class OccupantSimulation:
    def __init__(self, cfg: ScenarioConfig, progress: Optional[Callable[[str, float], None]] = None):
        self.cfg = cfg
        self.progress = progress or (lambda msg, frac: None)

    # ------------------------------------------------------------------
    def run(self) -> Results:
        cfg = self.cfg
        t_wall = time.time()
        self.progress("vehicle crash model", 0.0)
        motion_raw = run_vehicle_model(cfg)
        self.progress("building scene", 0.05)
        scene = build_scene(cfg)
        self.scene = scene
        m = scene.model
        d = mujoco.MjData(m)
        self.m, self.d = m, d
        dt = cfg.simulation.timestep_s
        T_pre = cfg.environment.pre_impact_duration_s
        T_post = cfg.collision.duration_after_impact_s
        n = int(round((T_pre + T_post) / dt))
        t_grid = -T_pre + dt * np.arange(n + 1)
        motion = motion_raw.resample(t_grid)
        self.motion = motion
        self._prepare(scene, motion, t_grid)
        self.progress("settling dog into posture", 0.08)
        self._settle()
        self.progress("simulating event", 0.15)
        rec = self._event_loop(t_grid)
        info = dict(motion.info)
        info["wall_time_s"] = time.time() - t_wall
        info["dog_mass_kg"] = scene.scaling.body_mass
        info["dog_withers_height_m"] = scene.scaling.withers_height
        info["length_scale"] = scene.scaling.length_scale
        info["attachment"] = getattr(self, "attachment", None)
        info["notes"] = list(scene.notes)
        res = Results(cfg=cfg, t=rec.pop("t"), channels=rec, states_t=np.array(self.state_t),
                      states_qpos=np.array(self.state_q), states_extra={k: np.array(v) for k, v in self.state_x.items()},
                      events=self.events, info=info, motion=motion, scene=scene)
        from .analysis.injury import assess
        res.metrics = assess(res)
        self.progress("done", 1.0)
        return res

    # ------------------------------------------------------------------
    def _prepare(self, scene: Scene, motion: VehicleMotion, t_grid):
        cfg = self.cfg
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[scene.dog_qpos_slice] = scene.dog_init_qpos
        # passive joint springs (and the very stiff caudal springs) relax about the initial posture
        hq = m.jnt_qposadr[scene.dog_joints]
        m.qpos_spring[hq] = d.qpos[hq]
        mujoco.mj_forward(m, d)
        self.dyn = scene.dynamic_bodies
        self.dyn_mass = m.body_mass[self.dyn].copy()
        self.dyn_inertia = m.body_inertia[self.dyn].copy()
        self.dyn_root = m.body_rootid[self.dyn].copy()
        rp = cfg.rear_passenger
        self.nm = NeuromuscularController(m, d, rp.neuromuscular, scene.dog_actuators, scene.dog_joints,
                                         scene.scaling.torque_scale, cfg.simulation.timestep_s)
        tid = m.body("torso").id
        com_h = 0.45 * scene.scaling.withers_height
        self.nm.setup_balance(tid, scene.scaling.body_mass, com_h,
                              scene.scaling.body_mass * (0.3 * scene.scaling.withers_height) ** 2)
        self.nm.capture_targets(d)
        # events from the vehicle motion
        a_h = np.hypot(motion.acc_body[:, 0], motion.acc_body[:, 1]) / G
        self.a_h = a_h
        self.events = {}
        onset = first_crossing(t_grid, a_h, 0.35)
        self.events["reflex_onset_s"] = onset
        imp = cfg.collision.impact_direction
        pt = cfg.restraints.pretensioner
        fire = None
        if pt.enabled and imp in pt.fire_on:
            c = first_crossing(t_grid, a_h, pt.fire_threshold_g, t_min=0.0)
            fire = None if c is None else c + pt.fire_delay_ms * 1e-3
        self.events["pretensioner_fire_s"] = fire
        # restraint models
        teth = cfg.harness.tether
        self.webs = {}
        self.retractor = None
        self.absorbers = {}
        for ti in scene.tendons:
            if ti.kind == "belt":
                k = 10e3 / (cfg.restraints.belt.webbing_elongation_pct_at_10kN / 100.0)
                self.webs[ti.name] = Webbing(ti.name, 10.0, k, ti.strength, 0.08, 3.0, 2.0 + 0.3 * scene.scaling.body_mass)
                ll = cfg.restraints.load_limiter
                self.retractor = Retractor(
                    lock_g=cfg.restraints.belt.retractor_lock_g, film_spool=cfg.restraints.belt.film_spool_mm * 1e-3,
                    pretensioner_stroke=pt.stroke_mm * 1e-3 if (pt.enabled and pt.type == "retractor") else 0.0,
                    pretensioner_force=pt.max_force_kN * 1e3, pretensioner_speed=pt.speed_m_per_s,
                    pretensioner_fire_time=fire if pt.type == "retractor" else None,
                    load_limit=ll.force_kN * 1e3 if ll.enabled else None, load_limit_stroke=ll.max_payout_mm * 1e-3)
            elif ti.kind == "belt_lap":
                continue
            elif ti.kind == "tether":
                k = teth.rated_strength_kN * 1e3 / (teth.elongation_pct_at_rated / 100.0)
                self.webs[ti.name] = Webbing(ti.name, 10.0, k, min(ti.strength, ti.connector_strength), teth.damping_ratio,
                                             3.0, 2.0 + 0.3 * scene.scaling.body_mass)
                if teth.energy_absorber_force_kN:
                    self.absorbers[ti.name] = EnergyAbsorber(teth.energy_absorber_force_kN * 1e3, teth.energy_absorber_stroke_m)
            elif ti.kind == "strap":
                h = cfg.harness
                k = h.strap_rated_strength_kN * 1e3 / (h.strap_elongation_pct_at_rated / 100.0)
                self.webs[ti.name] = Webbing(ti.name, 10.0, k, min(ti.strength, ti.connector_strength), 0.15, 3.0,
                                             1.0 + 0.1 * scene.scaling.body_mass)
            elif ti.kind == "cargo_strap":
                self.webs[ti.name] = Webbing(ti.name, 10.0, ti.meta["stiffness"], ti.strength, 0.1, 3.0, 20.0)
        self.tendon_by_name = {ti.name: ti for ti in scene.tendons}
        # seat back
        rs = cfg.restraints.seat
        self.seat = SeatBack(stiffness=rs.seat_back_stiffness_kNm_per_rad * 1e3, forward_yield=rs.seat_back_forward_yield_Nm,
                             rearward_yield=rs.seat_back_rearward_yield_Nm, latch_strength=rs.seat_back_latch_strength_Nm,
                             damping=2.0 * 0.3 * math.sqrt(rs.seat_back_stiffness_kNm_per_rad * 1e3 * 3.0))
        self.seat_q = m.jnt_qposadr[scene.seatback_joint]
        self.seat_v = m.jnt_dofadr[scene.seatback_joint]
        # airbags
        self.bags = []
        for ab in scene.airbags:
            c = ab["cfg"]
            ft = None
            if imp in c.fire_on:
                cr = first_crossing(t_grid, a_h, c.fire_threshold_g, t_min=0.0)
                ft = None if cr is None else cr + c.fire_delay_ms * 1e-3
            st = AirbagState(ab["name"], ft, c.inflation_time_ms * 1e-3, c.vent_time_constant_ms * 1e-3,
                             c.peak_pressure_kPa * 1e3, ab["stroke"])
            self.bags.append((ab, st))
            self.events[f"{ab['name']}_fire_s"] = ft
        # prescribed joints from the intrusion signals
        # slide joints of single-body trees: generalised mass = body mass + armature
        for p in list(scene.prescribed) + list(scene.airbags):
            p["M"] = float(m.body_mass[m.jnt_bodyid[p["jid"]]] + m.dof_armature[p["dof"]])
        self.presc = []
        for p in scene.prescribed:
            sig = getattr(motion, p["signal"]) * p["sign"]
            vel = np.gradient(sig, t_grid)
            acc = np.gradient(vel, t_grid)
            self.presc.append((p, sig, vel, acc))
        # harness bushing indices
        self.h_q = np.array([m.jnt_qposadr[j] for j in scene.harness_joints], int)
        self.h_v = np.array([m.jnt_dofadr[j] for j in scene.harness_joints], int)
        b = scene.bushing
        self.h_k = np.array([0.0, b["k_yz"], b["k_yz"], b["k_roll"], b["k_pitch"], b["k_yaw"]]) if len(self.h_q) else None
        # cargo restraint
        self.cr = scene.cargo_restraint
        self.partition_broken = False
        self.weld_broken = {}
        # sensors and sites
        self.sens = scene.sensors
        self.site = {nm: m.site(nm).id for nm in ("head_acc_site", "chest_acc_site", "pelvis_acc_site", "neck_ventral",
                                                   "prosternum")}
        self.site_body = {k: m.site_bodyid[v] for k, v in self.site.items()}
        self.site_root = {k: m.body_rootid[b] for k, b in self.site_body.items()}
        self.rotating = (np.abs(motion.omega).max(axis=1) > 1e-9) | (np.abs(motion.alpha).max(axis=1) > 1e-9)
        self.skull = m.body("skull").id
        self.torso = m.body("torso").id
        self.pelvis = m.body("pelvis").id
        self.dog_hinge_q = m.jnt_qposadr[scene.dog_joints]
        self.dog_lo = m.jnt_range[scene.dog_joints, 0]
        self.dog_hi = m.jnt_range[scene.dog_joints, 1]
        names = [m.joint(j).name for j in scene.dog_joints]
        self.group_idx = {g: np.array([i for i, nm in enumerate(names) if nm.startswith(pref)], int)
                          for g, pref in GROUPS.items()}
        self.seatback_geom = m.geom("rear_seat_back").id
        self.cargo_geoms = np.array([c["gid"] for c in scene.cargo], int)
        self.is_cargo_geom = np.zeros(m.ngeom, bool)
        self.is_cargo_geom[self.cargo_geoms] = True

    # ------------------------------------------------------------------
    def _bushing_forces(self):
        if self.h_k is None:
            return np.zeros(6)
        d = self.d
        q = d.qpos[self.h_q]
        b = self.scene.bushing
        f = -self.h_k * q
        kx = b["kx_rear"] if q[0] < 0 else b["kx_fwd"]
        f[0] = -kx * q[0]
        d.qfrc_applied[self.h_v] += f
        return f

    def _tendon_forces(self, t, dt, accel_g, settling=False):
        d = self.d
        out = {}
        for ti in self.scene.tendons:
            if ti.kind == "belt_lap":
                d.ctrl[ti.actuator] = 0.0
                continue
            web = self.webs[ti.name]
            L = d.ten_length[ti.tendon]
            Ld = d.ten_velocity[ti.tendon]
            if ti.kind == "belt" and self.retractor is not None:
                if settling:
                    web.free_length = L
                    web.eps_max = 0.0
                    T = self.retractor.rewind_tension
                else:
                    T = self.retractor.update(web, L, Ld, t, accel_g, dt)
                    if web.broken:
                        T = 0.0
            elif ti.kind == "tether" and not settling:
                if ti.meta.get("buckle_pretensioner") and self.events.get("pretensioner_fire_s") is not None:
                    pt = self.cfg.restraints.pretensioner
                    tf = self.events["pretensioner_fire_s"]
                    if t >= tf and getattr(web, "_pt_done", 0.0) < pt.stroke_mm * 1e-3 and web.tension < pt.max_force_kN * 1e3:
                        dl = min(pt.speed_m_per_s * dt, pt.stroke_mm * 1e-3 - getattr(web, "_pt_done", 0.0))
                        web.free_length -= dl
                        web._pt_done = getattr(web, "_pt_done", 0.0) + dl
                ea = self.absorbers.get(ti.name)
                T = ea.limit(web, L, Ld, t) if ea else web.force(L, Ld, t)
            elif ti.kind == "tether":
                T = 0.0
            else:
                T = web.force(L, Ld, t)
            d.ctrl[ti.actuator] = -T
            out[ti.name] = T
        return out

    def _seat_back(self, t):
        d = self.d
        M = self.seat.update(d.qpos[self.seat_q], d.qvel[self.seat_v], t)
        d.qfrc_applied[self.seat_v] += M

    def _cargo_restraint(self, t):
        d, m = self.d, self.m
        cr = self.cr
        force = 0.0
        if "jid" in cr and not self.partition_broken:
            x = d.qpos[cr["qadr"]]
            F = -cr["stiffness"] * x
            if abs(F) > cr["strength"]:
                self.partition_broken = True
                self.events["cargo_partition_failure_s"] = t
                m.geom_contype[cr["gid"]] = 0
                m.geom_conaffinity[cr["gid"]] = 0
                F = 0.0
            d.qfrc_applied[cr["dof"]] += F
            force = F
        if "weld_ids" in cr:
            strength = self.cfg.trunk_cargo.floor_box_strength_kN * 1e3
            for eid in cr["weld_ids"]:
                if self.weld_broken.get(eid):
                    continue
                rows = np.nonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) &
                                  (d.efc_id[:d.nefc] == eid))[0]
                if len(rows) >= 3:
                    F = np.linalg.norm(d.efc_force[rows[:3]])
                    force = max(force, F)
                    if F > strength:
                        d.eq_active[eid] = 0
                        self.weld_broken[eid] = True
                        self.events[f"cargo_lock_{eid}_failure_s"] = t
        return force

    def _frame_loads(self, k):
        """Gravity and fictitious loads for the vehicle frame at grid index k."""
        m, d = self.m, self.d
        mo = self.motion
        m.opt.gravity[:] = mo.gravity[k] - mo.acc_origin[k]
        if not self.rotating[k]:
            return
        Om = mo.omega[k]
        Al = mo.alpha[k]
        B = self.dyn
        r = d.xipos[B]
        w_rel = d.cvel[B, :3]
        c = d.subtree_com[self.dyn_root]
        v = d.cvel[B, 3:] + vcross(w_rel, r - c)
        mass = self.dyn_mass[:, None]
        F = -mass * (vcross(Al, r) + vcross(Om, vcross(Om, r)) + 2.0 * vcross(Om, v))
        Rm = d.ximat[B].reshape(-1, 3, 3)
        Ib = self.dyn_inertia

        def Iw(vec):
            loc = np.einsum("nji,nj->ni", Rm, vec)
            return np.einsum("nij,nj->ni", Rm, loc * Ib)

        w_abs = w_rel + Om
        tau = -Iw(Al + vcross(Om, w_rel)) - vcross(w_abs, Iw(w_abs)) + vcross(w_rel, Iw(w_rel))
        d.xfrc_applied[B, :3] += F
        d.xfrc_applied[B, 3:] += tau

    def _apply_prescribed(self, k, set_state):
        d, m = self.d, self.m
        for p, sig, vel, acc in self.presc:
            if set_state:
                d.qpos[p["qadr"]] = sig[k]
                d.qvel[p["dof"]] = vel[k]
            else:
                d.qfrc_applied[p["dof"]] += p["M"] * acc[k] + d.qfrc_bias[p["dof"]]
        for ab, st in self.bags:
            t = self.t_grid[k]
            x, xd, xdd = st.kinematics(t)
            if set_state:
                d.qpos[ab["qadr"]] = x
                d.qvel[ab["dof"]] = xd
            else:
                d.qfrc_applied[ab["dof"]] += ab["M"] * xdd + d.qfrc_bias[ab["dof"]]
                g = ab["gid"]
                if st.deployed_fraction > 0.02 and m.geom_contype[g] == 0:
                    m.geom_contype[g] = 2
                    m.geom_conaffinity[g] = 1
                    m.geom_rgba[g] = ab["rgba"]
                if st.pressure > 0:
                    tc = float(np.clip(0.02 * math.sqrt(45e3 / max(st.pressure, 1e3)), 0.004, 0.05))
                    m.geom_solref[g, 0] = tc

    # ------------------------------------------------------------------
    def _settle(self):
        cfg = self.cfg
        m, d = self.m, self.d
        dt_s = 2.5e-4
        m.opt.timestep = dt_s
        m.opt.gravity[:] = [0, 0, -cfg.environment.gravity]
        n = int(round(cfg.simulation.settle_time_s / dt_s))
        nm = self.nm
        nm.set_tone(max(cfg.rear_passenger.neuromuscular.muscle_tone, 0.7))
        # straps fitted to the dog at the start
        self._fit_straps()
        self.t_grid = np.array([0.0])
        for i in range(n):
            d.qfrc_applied[:] = 0.0
            d.xfrc_applied[:] = 0.0
            frac = i / max(n - 1, 1)
            if frac > 0.6:
                nm.set_tone(max(cfg.rear_passenger.neuromuscular.muscle_tone, 0.7) * (1 - (frac - 0.6) / 0.4) +
                            cfg.rear_passenger.neuromuscular.muscle_tone * (frac - 0.6) / 0.4)
            nm.step(0.0, d)
            nm.passive_tissue(d)
            nm.balance(d, 1.0)
            self._bushing_forces()
            self._tendon_forces(0.0, dt_s, 0.0, settling=True)
            self._seat_back(0.0)
            self._cargo_restraint(0.0)
            mujoco.mj_step(m, d)
            for p, sig, vel, acc in self.presc:
                d.qpos[p["qadr"]] = sig[0]
                d.qvel[p["dof"]] = 0.0
            for ab, st in self.bags:
                d.qpos[ab["qadr"]] = 0.0
                d.qvel[ab["dof"]] = 0.0
        nm.set_tone(cfg.rear_passenger.neuromuscular.muscle_tone)
        nm.capture_targets(d)
        nm.hist.clear()
        # tether length
        teth = cfg.harness.tether
        for ti in self.scene.tendons:
            web = self.webs.get(ti.name)
            if web is None:
                continue
            L = float(d.ten_length[ti.tendon])
            if ti.kind == "tether":
                if teth.length_m is not None:
                    web.free_length = max(teth.length_m, L - 0.002)
                    if teth.length_m < L - 0.002:
                        self.scene.notes.append(f"Tether length {teth.length_m:.3f} m is shorter than the taut length "
                                                f"{L:.3f} m at rest; the tether starts taut.")
                else:
                    web.free_length = L + teth.slack_m
            elif ti.kind == "belt":
                web.free_length = L + cfg.restraints.belt.slack_mm * 1e-3
            web.reset_set()
        self.attachment = next((ti.kind for ti in self.scene.tendons if ti.kind in ("tether", "belt")), "none")
        self.seat.peak_moment = 0.0
        m.opt.timestep = cfg.simulation.timestep_s
        # reference positions for excursion
        self.ref = {"head": d.xipos[self.skull].copy(), "torso": d.xipos[self.torso].copy(),
                    "pelvis": d.xipos[self.pelvis].copy(), "torso_R": d.ximat[self.torso].reshape(3, 3).copy(),
                    "cargo": [d.xipos[c["bid"]].copy() for c in self.scene.cargo]}
        self.settled_qpos = d.qpos.copy()

    def _fit_straps(self):
        d = self.d
        mujoco.mj_forward(self.m, d)
        for ti in self.scene.tendons:
            web = self.webs.get(ti.name)
            if web is None:
                continue
            L = float(d.ten_length[ti.tendon])
            if ti.kind == "strap":
                web.free_length = L + ti.meta.get("slack", 0.01)
            elif ti.kind == "cargo_strap":
                web.free_length = L * 0.995
            else:
                web.free_length = L + 0.5  # slack while settling

    # ------------------------------------------------------------------
    def _event_loop(self, t_grid):
        cfg = self.cfg
        m, d = self.m, self.d
        sc = self.scene
        mo = self.motion
        self.t_grid = t_grid
        dt = cfg.simulation.timestep_s
        n = len(t_grid) - 1
        every = max(1, cfg.simulation.record_every)
        nrec = n // every + 1
        tendon_names = [ti.name for ti in sc.tendons if ti.kind != "belt_lap"]
        ncargo = len(sc.cargo)
        C = {
            "t": np.zeros(nrec),
            "head_acc": np.zeros((nrec, 3)), "chest_acc": np.zeros((nrec, 3)), "pelvis_acc": np.zeros((nrec, 3)),
            "lower_neck_force": np.zeros((nrec, 3)), "lower_neck_torque": np.zeros((nrec, 3)),
            "upper_neck_force": np.zeros((nrec, 3)), "upper_neck_torque": np.zeros((nrec, 3)),
            "thoracolumbar_force": np.zeros((nrec, 3)), "thoracolumbar_torque": np.zeros((nrec, 3)),
            "lumbosacral_force": np.zeros((nrec, 3)), "lumbosacral_torque": np.zeros((nrec, 3)),
            "head_pos": np.zeros((nrec, 3)), "torso_pos": np.zeros((nrec, 3)), "pelvis_pos": np.zeros((nrec, 3)),
            "torso_rot": np.zeros((nrec, 3)),
            "tendon_tension": np.zeros((nrec, len(tendon_names))),
            "throat_force": np.zeros(nrec), "prosternum_strap_force": np.zeros(nrec),
            "harness_q": np.zeros((nrec, 6)), "harness_force": np.zeros((nrec, 6)),
            "seat_back_angle": np.zeros(nrec), "seat_back_moment": np.zeros(nrec),
            "door_left": np.zeros(nrec), "door_right": np.zeros(nrec), "rear_wall": np.zeros(nrec),
            "airbag_fraction": np.zeros((nrec, max(len(self.bags), 1))),
            "cargo_pos": np.zeros((nrec, max(ncargo, 1), 3)), "cargo_seatback_force": np.zeros(nrec),
            "cargo_restraint_force": np.zeros(nrec),
            "contact_force": np.zeros((nrec, len(DOG_PARTS), len(VEH_PARTS))),
            "dog_cargo_force": np.zeros(nrec),
            "joint_overrange_deg": np.zeros((nrec, len(GROUPS))), "ligament_load": np.zeros((nrec, len(GROUPS))),
            "activation": np.zeros(nrec), "vehicle_acc": np.zeros((nrec, 3)), "vehicle_dv": np.zeros((nrec, 3)),
            "pretension_retracted": np.zeros(nrec), "load_limiter_payout": np.zeros(nrec),
        }
        C["tendon_names"] = tendon_names
        self.state_t, self.state_q = [], []
        self.state_x = {"airbag": [], "seat_back_latch": []}
        state_every = max(1, int(round(1.0 / (cfg.simulation.state_rate_hz * dt))))
        onset = self.events.get("reflex_onset_s")
        nm = self.nm
        vb = np.einsum("nji,nj->ni", mo.R, mo.vel)
        i0 = int(np.argmin(np.abs(t_grid)))
        v_imp = vb[i0]
        r_idx = 0
        gd = sc.geom_dog_part
        gv = sc.geom_veh_part
        ten_idx = {ti.name: ti for ti in sc.tendons}
        throat = ten_idx.get("neck_loop")
        front = ten_idx.get("front_strap")
        site_ids = {nm_: m.site(nm_).id for nm_ in ("neck_lat_L", "neck_ventral", "neck_lat_R", "shoulder_front_L",
                                                     "prosternum", "shoulder_front_R")}
        report_every = max(1, n // 20)
        self._apply_prescribed(0, True)
        mujoco.mj_forward(m, d)
        for k in range(n):
            t = t_grid[k]
            if onset is not None and t >= onset:
                nm.trigger(onset)
            d.qfrc_applied[:] = 0.0
            d.xfrc_applied[:] = 0.0
            self._frame_loads(k)
            self._apply_prescribed(k, False)
            nm.step(t, d)
            lig = nm.passive_tissue(d)
            nm.balance(d, min(1.0, nm.activation / 0.35))
            hforce = self._bushing_forces()
            T = self._tendon_forces(t, dt, self.a_h[k])
            self._seat_back(t)
            crf = self._cargo_restraint(t)
            mujoco.mj_step(m, d)
            self._apply_prescribed(k + 1, True)
            # ---- record (quantities of the forward pass at time t) ----
            if k % every == 0 and r_idx < nrec:
                i = r_idx
                C["t"][i] = t
                for nm_s in ("head_acc", "chest_acc", "pelvis_acc"):
                    C[nm_s][i] = self._true_accel(nm_s, k)
                for lc in ("lower_neck", "upper_neck", "thoracolumbar", "lumbosacral"):
                    a, _ = self.sens[lc + "_force"]
                    C[lc + "_force"][i] = d.sensordata[a:a + 3]
                    a, _ = self.sens[lc + "_torque"]
                    C[lc + "_torque"][i] = d.sensordata[a:a + 3]
                C["head_pos"][i] = d.xipos[self.skull]
                C["torso_pos"][i] = d.xipos[self.torso]
                C["pelvis_pos"][i] = d.xipos[self.pelvis]
                Rt = d.ximat[self.torso].reshape(3, 3)
                Rrel = self.ref["torso_R"].T @ Rt
                C["torso_rot"][i] = [math.degrees(math.atan2(Rrel[2, 1], Rrel[2, 2])),
                                     math.degrees(-math.asin(max(-1.0, min(1.0, Rrel[2, 0])))),
                                     math.degrees(math.atan2(Rrel[1, 0], Rrel[0, 0]))]
                C["tendon_tension"][i] = [T.get(nm_t, 0.0) for nm_t in tendon_names]
                if throat is not None:
                    fv = self._via_force(T.get("neck_loop", 0.0), site_ids["neck_lat_L"], site_ids["neck_ventral"],
                                         site_ids["neck_lat_R"])
                    if throat.meta.get("on_throat", True):
                        C["throat_force"][i] = fv
                    else:
                        C["prosternum_strap_force"][i] += fv
                if front is not None:
                    C["prosternum_strap_force"][i] += self._via_force(T.get("front_strap", 0.0), site_ids["shoulder_front_L"],
                                                                     site_ids["prosternum"], site_ids["shoulder_front_R"])
                if len(self.h_q):
                    C["harness_q"][i] = d.qpos[self.h_q]
                    C["harness_force"][i] = -(hforce + d.qfrc_constraint[self.h_v] + d.qfrc_passive[self.h_v])
                C["seat_back_angle"][i] = d.qpos[self.seat_q]
                C["seat_back_moment"][i] = self.seat.moment
                for p, sig, _, _ in self.presc:
                    key = {"intrusion_left": "door_left", "intrusion_right": "door_right",
                           "intrusion_rear": "rear_wall"}[p["signal"]]
                    C[key][i] = abs(sig[k])
                for j, (ab, st) in enumerate(self.bags):
                    C["airbag_fraction"][i, j] = st.deployed_fraction
                for j, c in enumerate(sc.cargo):
                    C["cargo_pos"][i, j] = d.xipos[c["bid"]]
                C["cargo_restraint_force"][i] = crf
                self._record_contacts(C, i, gd, gv)
                q = d.qpos[self.dog_hinge_q]
                over = np.degrees(np.maximum(q - self.dog_hi, self.dog_lo - q))
                for gi, g in enumerate(GROUPS):
                    idx = self.group_idx[g]
                    if len(idx):
                        C["joint_overrange_deg"][i, gi] = max(over[idx].max(), 0.0)
                        C["ligament_load"][i, gi] = np.abs(lig[idx]).max()
                C["activation"][i] = nm.activation
                C["vehicle_acc"][i] = mo.acc_body[k]
                C["vehicle_dv"][i] = vb[k] - v_imp
                if self.retractor is not None:
                    C["pretension_retracted"][i] = self.retractor.retracted
                    C["load_limiter_payout"][i] = self.retractor.paid_out
                r_idx += 1
            if k % state_every == 0:
                self.state_t.append(t)
                self.state_q.append(d.qpos.copy())
                self.state_x["airbag"].append([st.deployed_fraction for _, st in self.bags])
                self.state_x["seat_back_latch"].append(float(self.seat.latch_broken))
            if not np.isfinite(d.qacc).all() or np.abs(d.qvel[sc.dog_qvel_slice]).max() > 500.0:
                self.events["numerical_instability_s"] = float(t)
                break
            if k % report_every == 0:
                self.progress("simulating event", 0.15 + 0.8 * k / n)
        for nm_t, web in self.webs.items():
            if web.broken:
                self.events[f"{nm_t}_failure_s"] = web.broken_at
        if self.seat.latch_broken:
            self.events["seat_back_latch_failure_s"] = self.seat.latch_broken_at
        if self.retractor is not None:
            self.events["retractor_lock_s"] = self.retractor.lock_time
        for key in list(C):
            if isinstance(C[key], np.ndarray) and C[key].shape[:1] == (nrec,):
                C[key] = C[key][:r_idx]
        return C

    def _true_accel(self, name, k):
        """Inertial proper acceleration (m/s^2, site frame) of an accelerometer site."""
        d = self.d
        mo = self.motion
        a, _ = self.sens[name]
        meas = d.sensordata[a:a + 3].copy()
        Om, Al = mo.omega[k], mo.alpha[k]
        if self.rotating[k]:
            sid = self.site[name + "_site"]
            b = self.site_body[name + "_site"]
            p = d.site_xpos[sid]
            c = d.subtree_com[self.site_root[name + "_site"]]
            w = d.cvel[b, :3]
            pc = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
            wx = cross3(w, pc)
            v = (d.cvel[b, 3] + wx[0], d.cvel[b, 4] + wx[1], d.cvel[b, 5] + wx[2])
            a1 = cross3(Al, p)
            a2 = cross3(Om, cross3(Om, p))
            a3 = cross3(Om, v)
            extra = np.array([a1[0] + a2[0] + 2 * a3[0], a1[1] + a2[1] + 2 * a3[1], a1[2] + a2[2] + 2 * a3[2]])
            Rs = d.site_xmat[sid].reshape(3, 3)
            meas += Rs.T @ extra
        return meas

    def _via_force(self, T, a, b, c):
        if T <= 0:
            return 0.0
        d = self.d
        pa, pb, pc = d.site_xpos[a], d.site_xpos[b], d.site_xpos[c]
        u1 = pa - pb
        u2 = pc - pb
        u1 /= max(np.linalg.norm(u1), 1e-9)
        u2 /= max(np.linalg.norm(u2), 1e-9)
        return float(T * np.linalg.norm(u1 + u2))

    def _record_contacts(self, C, i, gd, gv):
        d = self.d
        nc = d.ncon
        if nc == 0:
            return
        con = d.contact
        g = con.geom[:nc]
        adr = con.efc_address[:nc]
        ok = adr >= 0
        if not ok.any():
            return
        g = g[ok]
        fn = d.efc_force[adr[ok]]
        d1, d2 = gd[g[:, 0]], gd[g[:, 1]]
        v1, v2 = gv[g[:, 0]], gv[g[:, 1]]
        sel = (d1 >= 0) & (v2 >= 0)
        np.add.at(C["contact_force"][i], (d1[sel], v2[sel]), fn[sel])
        sel = (d2 >= 0) & (v1 >= 0)
        np.add.at(C["contact_force"][i], (d2[sel], v1[sel]), fn[sel])
        if len(self.cargo_geoms):
            cg = self.is_cargo_geom[g]
            sb = g == self.seatback_geom
            pair = (cg[:, 0] & sb[:, 1]) | (cg[:, 1] & sb[:, 0])
            C["cargo_seatback_force"][i] = fn[pair].sum()
            dogc = ((gd[g[:, 0]] >= 0) & cg[:, 1]) | ((gd[g[:, 1]] >= 0) & cg[:, 0])
            C["dog_cargo_force"][i] = fn[dogc].sum()


def run_scenario(cfg: ScenarioConfig, progress=None) -> Results:
    return OccupantSimulation(cfg, progress).run()
