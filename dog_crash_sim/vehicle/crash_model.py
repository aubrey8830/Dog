"""Vehicle-level crash dynamics (stage 1 of the simulation).

The subject vehicle (the one carrying the dog) is a rigid body moving in the
road plane (x, y, yaw) with pitch and roll suspension modes.  Before the
impact it is driven by tyre forces: braking limited by the road friction
(with or without ABS), cornering forces from a saturating tyre model and an
optional evasive steering input.  At ``t = 0`` it meets the collision
partner.  The interface between the two is a pair of elasto-plastic springs
in series - the subject's crush zones and the partner's face - whose force
depends on the crush, on how much of each structure is engaged (overlap,
pole width, ride-height mismatch) and on the loading history (plastic
crush, elastic spring-back).  The interface force acts at the engaged
structure's centroid, so offset and oblique impacts produce yaw.

The result is a :class:`VehicleMotion`: the rigid-body motion of the
passenger compartment plus crush/intrusion histories.  The occupant stage
uses it as the non-inertial reference frame in which the dog, its harness
and the cargo are simulated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import G, KPH, ScenarioConfig, partner_defaults, road_friction
from ..scene.layout import compute_layout

FACE_NORMAL = {"frontal": (1.0, 0.0), "rear": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
PARTNER_FACE_NORMAL = {"front": (1.0, 0.0), "side": (0.0, -1.0), "rear": (-1.0, 0.0)}


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def cross2(r, f) -> float:
    return r[0] * f[1] - r[1] * f[0]


def interval_overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------

class ElastoPlastic:
    """Monotone elasto-plastic force-crush characteristic with elastic unloading.

    The loading curve rises with the elastic stiffness to each zone's crush
    force, stays on that plateau for the zone depth, then continues to the
    next zone; once every zone is used the (much stiffer) passenger cell or
    barrier backing takes over.  Unloading follows ``unload_k`` from the
    largest crush reached, which sets the restitution.
    """

    def __init__(self, zones, k_elastic, k_backing, unload_factor=6.0, scale=1.0, backing_scale=None):
        k = max(k_elastic * scale, 1.0)
        kb = k_backing * (backing_scale if backing_scale is not None else scale)
        xs, fs = [0.0], [0.0]
        cum, fprev = 0.0, 0.0
        for depth, force in zones:
            target = max(force * scale, fprev)
            if target > fprev:
                ramp = min((target - fprev) / k, depth)
                xs.append(cum + ramp)
                fs.append(fprev + k * ramp)
                fprev = fs[-1]
            cum += depth
            if xs[-1] < cum:
                xs.append(cum)
                fs.append(fprev)
        xs.append(cum + 1.0)
        fs.append(fprev + kb)
        self.xs = np.array(xs)
        # small hardening keeps the curve strictly increasing (unique series split)
        self.fs = np.array(fs) + 2.0e4 * self.xs
        self.slope_end = kb + 2.0e4
        self.ku = max(unload_factor * k, 2.0 * k)
        self.dmax = 0.0
        self.total_depth = cum

    def loading(self, d: float) -> float:
        if d <= 0.0:
            return 0.0
        if d >= self.xs[-1]:
            return float(self.fs[-1] + self.slope_end * (d - self.xs[-1]))
        return float(np.interp(d, self.xs, self.fs))

    def force(self, d: float) -> float:
        if d >= self.dmax:
            return self.loading(d)
        return max(0.0, self.loading(self.dmax) - self.ku * (self.dmax - d))

    def commit(self, d: float) -> None:
        if d > self.dmax:
            self.dmax = d


class SeriesInterface:
    """Two crushable faces in series; ``partner=None`` means a rigid partner."""

    def __init__(self, subject: ElastoPlastic, partner: Optional[ElastoPlastic]):
        self.s = subject
        self.p = partner
        self.ds = 0.0
        self.dp = 0.0

    def solve(self, d: float) -> float:
        if d <= 0.0 and self.s.force(0.0) == 0.0:
            self.ds, self.dp = 0.0, 0.0
            return 0.0
        if self.p is None:
            self.ds, self.dp = d, 0.0
            f = self.s.force(d)
            self.s.commit(d)
            return f
        lo, hi = 0.0, max(d, 0.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self.s.force(mid) - self.p.force(d - mid) > 0.0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-7:
                break
        ds = 0.5 * (lo + hi)
        dp = d - ds
        f = 0.5 * (self.s.force(ds) + self.p.force(dp))
        self.s.commit(ds)
        self.p.commit(dp)
        self.ds, self.dp = ds, dp
        return f


def structural_engagement(face, lo, hi, span, st, lay, kappa=0.5, spread=0.25):
    """Fraction of the face's structure loaded by a partner covering [lo, hi] and its centroid.

    Discrete members (rails, power train, pillars) are loaded directly when
    covered and partially, through the bumper/door beams, when they sit
    close to the contact; distributed beams (bumper beam, sill/door beams)
    spread the load over a length ``beam_spread`` on either side.
    """
    width = hi - lo
    length = span[1] - span[0]
    if face in ("frontal", "rear"):
        half = 0.5 * length
        yr = st.rail_lateral_position * half
        r = st.front_rail_share
        beam_share = max(0.0, 1.0 - 2.0 * r - 0.15)
        members = [("beam", beam_share, span[0], span[1], 0.40), ("disc", r, yr - 0.06, yr + 0.06, 0.0),
                   ("disc", r, -yr - 0.06, -yr + 0.06, 0.0), ("disc", 0.15, -0.30, 0.30, 0.0)]
    else:
        bs, cs = st.b_pillar_share, 0.2
        members = [("beam", max(0.0, 1.0 - bs - cs), span[0], span[1], 0.60),
                   ("disc", bs, lay.b_pillar_x - 0.075, lay.b_pillar_x + 0.075, 0.0),
                   ("disc", cs, lay.c_pillar_x - 0.075, lay.c_pillar_x + 0.075, 0.0)]
    e = 0.0
    mom = 0.0
    for kind, share, a0, a1, lb in members:
        if kind == "beam":
            g = min(1.0, (width + 2.0 * lb) / length)
            c = 0.5 * (lo + hi)
        else:
            frac = interval_overlap(a0, a1, lo, hi) / (a1 - a0)
            gap = max(0.0, a0 - hi, lo - a1)
            g = max(frac, kappa * math.exp(-(gap / spread) ** 2) if frac < 1.0 else 0.0)
            c = min(max(0.5 * (a0 + a1), lo), hi)
        e += share * g
        mom += share * g * c
    u_cent = mom / e if e > 1e-9 else 0.5 * (lo + hi)
    return e, min(max(u_cent, lo), hi)


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class VehicleMotion:
    t: np.ndarray
    pos: np.ndarray          # (N,3) world CG position
    euler: np.ndarray        # (N,3) roll, pitch, yaw
    vel: np.ndarray          # (N,3) world CG velocity
    acc_cg_world: np.ndarray  # (N,3)
    omega: np.ndarray        # (N,3) angular velocity, vehicle frame
    alpha: np.ndarray        # (N,3) angular acceleration, vehicle frame
    acc_origin: np.ndarray   # (N,3) acceleration of the vehicle-frame origin, vehicle frame
    gravity: np.ndarray      # (N,3) gravity in the vehicle frame
    R: np.ndarray            # (N,3,3) world-from-vehicle
    acc_body: np.ndarray     # (N,3) CG acceleration in the vehicle frame (what an accelerometer reads, minus gravity)
    crush_subject: np.ndarray
    crush_partner: np.ndarray
    interface_force: np.ndarray
    intrusion_left: np.ndarray
    intrusion_right: np.ndarray
    intrusion_rear: np.ndarray
    crush_front: np.ndarray
    partner_pos: np.ndarray  # (N,3)
    partner_yaw: np.ndarray
    cg_height: float
    info: dict = field(default_factory=dict)

    def resample(self, t_new: np.ndarray) -> "VehicleMotion":
        def ip(a):
            a = np.asarray(a)
            if a.ndim == 1:
                return np.interp(t_new, self.t, a)
            flat = a.reshape(len(self.t), -1)
            out = np.stack([np.interp(t_new, self.t, flat[:, k]) for k in range(flat.shape[1])], axis=1)
            return out.reshape((len(t_new),) + a.shape[1:])

        euler = ip(self.euler)
        R = np.stack([euler_to_R(e) for e in euler])
        return VehicleMotion(
            t=np.asarray(t_new), pos=ip(self.pos), euler=euler, vel=ip(self.vel), acc_cg_world=ip(self.acc_cg_world),
            omega=ip(self.omega), alpha=ip(self.alpha), acc_origin=ip(self.acc_origin), gravity=ip(self.gravity), R=R,
            acc_body=ip(self.acc_body), crush_subject=ip(self.crush_subject), crush_partner=ip(self.crush_partner),
            interface_force=ip(self.interface_force), intrusion_left=ip(self.intrusion_left),
            intrusion_right=ip(self.intrusion_right), intrusion_rear=ip(self.intrusion_rear),
            crush_front=ip(self.crush_front), partner_pos=ip(self.partner_pos), partner_yaw=ip(self.partner_yaw),
            cg_height=self.cg_height, info=dict(self.info))


def euler_to_R(e) -> np.ndarray:
    """World-from-vehicle rotation for (roll, pitch, yaw), R = Rz(yaw) Ry(pitch) Rx(roll)."""
    phi, th, psi = e
    cf, sf = math.cos(phi), math.sin(phi)
    ct, st = math.cos(th), math.sin(th)
    cp, sp = math.cos(psi), math.sin(psi)
    return np.array([
        [cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf],
        [sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf],
        [-st, ct * sf, ct * cf],
    ])


# --------------------------------------------------------------------------
# Pulses for sled mode
# --------------------------------------------------------------------------

def sled_pulse(cfg: ScenarioConfig):
    """Return (accel_function(t) in m/s^2 of deceleration, duration_s)."""
    sp = cfg.collision.sled
    dv = sp.delta_v_kph * KPH
    if sp.shape == "half_sine":
        T = sp.duration_ms * 1e-3
        A = dv * math.pi / (2.0 * T)
        return (lambda t: A * math.sin(math.pi * t / T) if 0.0 <= t <= T else 0.0), T
    if sp.shape in ("trapezoid", "fmvss213"):
        if sp.shape == "fmvss213":
            # approximation of the FMVSS 213 sled acceleration corridor (~48 km/h, ~24 g plateau)
            r, T, fl = 0.022, 0.080, 0.028
        else:
            r, T, fl = sp.rise_ms * 1e-3, sp.duration_ms * 1e-3, sp.fall_ms * 1e-3
        plateau = max(T - r - fl, 0.0)
        A = dv / (0.5 * r + plateau + 0.5 * fl)

        def f(t, A=A, r=r, T=T, fl=fl):
            if t < 0.0 or t > T:
                return 0.0
            if t < r:
                return A * t / r
            if t > T - fl:
                return A * (T - t) / fl
            return A
        return f, T
    if sp.shape == "csv":
        data = np.loadtxt(sp.csv_path, delimiter=",", skiprows=1)
        ts, ag = data[:, 0], data[:, 1] * G
        return (lambda t: float(np.interp(t, ts, ag, left=0.0, right=0.0))), float(ts[-1])
    raise ValueError(sp.shape)


# --------------------------------------------------------------------------
# Tyres
# --------------------------------------------------------------------------

def axle_forces(vx, vy, Fz, mu, steer, brake_force, abs_, C_alpha, v_min=1.0):
    """Body-frame forces of one axle (front: steer != 0).  Returns (Fx, Fy)."""
    c, s = math.cos(steer), math.sin(steer)
    vxt = vx * c + vy * s
    vyt = -vx * s + vy * c
    cap = mu * max(Fz, 0.0)
    speed = math.hypot(vxt, vyt)
    if cap <= 0.0:
        return 0.0, 0.0
    if brake_force > cap and not abs_:
        # locked wheels slide along the contact velocity
        k = cap / math.sqrt(speed * speed + 0.05 ** 2)
        fxt, fyt = -k * vxt, -k * vyt
    else:
        fb = min(brake_force, cap)
        fxt = -fb * math.tanh(vxt / 0.3)
        lat_cap = math.sqrt(max(cap * cap - fxt * fxt, 0.0))
        alpha = math.atan2(vyt, max(abs(vxt), v_min))
        fyt = -lat_cap * math.tanh(C_alpha * alpha / max(lat_cap, 1.0)) if lat_cap > 0 else 0.0
    return fxt * c - fyt * s, fxt * s + fyt * c


# --------------------------------------------------------------------------
# Main model
# --------------------------------------------------------------------------

class CrashModel:
    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg
        v = cfg.vehicle
        self.m = v.mass_kg
        L, W = v.length_m, v.width_m
        self.Izz = v.yaw_inertia_kgm2 or 0.0625 * self.m * (L * L + W * W)
        self.Iyy = v.pitch_inertia_kgm2 or 0.06 * self.m * (L * L + v.height_m ** 2)
        self.Ixx = v.roll_inertia_kgm2 or 0.067 * self.m * (W * W + v.height_m ** 2)
        self.a = v.cg_to_front_axle_m
        self.b = v.wheelbase_m - self.a
        self.h = v.cg_height_m
        self.mu = road_friction(cfg.environment)
        self.g = cfg.environment.gravity
        self.lay = compute_layout(cfg)
        sus = v.suspension
        self.kp = self.Iyy * (2 * math.pi * sus.pitch_frequency_hz) ** 2
        self.cp = 2 * sus.damping_ratio * math.sqrt(self.kp * self.Iyy)
        self.kr = self.Ixx * (2 * math.pi * sus.roll_frequency_hz) ** 2
        self.cr = 2 * sus.damping_ratio * math.sqrt(self.kr * self.Ixx)
        self.bump = math.radians(sus.bump_stop_deg)

    # ---- driver inputs -------------------------------------------------
    def brake_demand(self, t: float) -> float:
        br = self.cfg.environment.braking
        if not br.enabled:
            return 0.0
        t_on = -br.onset_before_impact_s
        if t < t_on:
            return 0.0
        if t > 0.0 and not br.continue_after_impact:
            return 0.0
        ramp = min(1.0, (t - t_on) / max(br.ramp_time_s, 1e-3))
        return ramp * br.deceleration_g * self.g * self.m

    def steer(self, t: float) -> float:
        mo = self.cfg.environment.motion
        if mo.steer_angle_deg == 0.0:
            return 0.0
        t_on = -mo.steer_onset_before_impact_s
        if t < t_on:
            return 0.0
        return math.radians(mo.steer_angle_deg) * min(1.0, (t - t_on) / max(mo.steer_ramp_s, 1e-3))

    # ---- subject tyre forces (body frame) ------------------------------
    def tyre_forces(self, t, u, v_lat, r, ax_prev):
        veh = self.cfg.vehicle
        Lwb = veh.wheelbase_m
        ax_c = max(-1.5 * self.g, min(1.5 * self.g, ax_prev))
        Fzf = self.m * self.g * self.b / Lwb - self.m * ax_c * self.h / Lwb
        Fzr = self.m * self.g * self.a / Lwb + self.m * ax_c * self.h / Lwb
        Fzf = min(max(Fzf, 0.0), self.m * self.g)
        Fzr = min(max(Fzr, 0.0), self.m * self.g)
        Fb = self.brake_demand(t)
        bf = veh.brake_balance_front
        abs_ = self.cfg.environment.braking.abs
        C = veh.tire_cornering_stiffness_N_per_rad
        fxf, fyf = axle_forces(u, v_lat + self.a * r, Fzf, self.mu, self.steer(t), Fb * bf, abs_, C)
        fxr, fyr = axle_forces(u, v_lat - self.b * r, Fzr, self.mu, 0.0, Fb * (1 - bf), abs_, C)
        Fx = fxf + fxr
        Fy = fyf + fyr
        Mz = self.a * fyf - self.b * fyr
        return Fx, Fy, Mz

    # ---- one pre-impact integration, returns states --------------------
    def _integrate(self, v0, T_pre, dt, record=True, crash=None, t_end=0.0):
        mo = self.cfg.environment.motion
        beta = math.radians(mo.sideslip_deg)
        X = np.zeros(2)
        psi = 0.0
        V = v0 * np.array([math.cos(beta), math.sin(beta)])
        r = math.radians(mo.initial_yaw_rate_deg_s)
        th = thd = ph = phd = 0.0
        ax_prev = 0.0
        t = -T_pre
        rows = []
        state = dict(X=X, psi=psi, V=V, r=r, th=th, thd=thd, ph=ph, phd=phd)
        while t < t_end - 1e-12:
            state, row, ax_prev = self._step(t, dt, state, ax_prev, crash)
            if record:
                rows.append(row)
            t += dt
        return state, rows

    def _step(self, t, dt, st, ax_prev, crash):
        X, psi, V, r = st["X"], st["psi"], st["V"], st["r"]
        Rz = rot2(psi)
        vb = Rz.T @ V
        Fx, Fy, Mz = self.tyre_forces(t, vb[0], vb[1], r, ax_prev)
        F_tyre_w = Rz @ np.array([Fx, Fy])
        F_crash_w = np.zeros(2)
        M_crash = 0.0
        z_force = self.cfg.vehicle.structure.structure_height_m
        extra = {}
        if crash is not None:
            F_crash_w, M_crash, extra = crash(t, st)
            z_force = extra.get("z_force", z_force)
        Ftot = F_tyre_w + F_crash_w
        acc = Ftot / self.m
        racc = (Mz + M_crash) / self.Izz
        # pitch / roll with bump stops
        Fb_crash = Rz.T @ F_crash_w
        My = (z_force - self.h) * Fb_crash[0] - self.h * Fx
        Mx = (self.h - z_force) * Fb_crash[1] + self.h * Fy

        def spring(k, ang):
            over = max(0.0, abs(ang) - self.bump)
            return k * ang + 8.0 * k * over * math.copysign(1.0, ang)

        thdd = (My - spring(self.kp, st["th"]) - self.cp * st["thd"]) / self.Iyy
        phdd = (Mx - spring(self.kr, st["ph"]) - self.cr * st["phd"]) / self.Ixx
        # semi-implicit Euler
        V = V + acc * dt
        r = r + racc * dt
        X = X + V * dt
        psi = psi + r * dt
        thd = st["thd"] + thdd * dt
        th = st["th"] + thd * dt
        phd = st["phd"] + phdd * dt
        ph = st["ph"] + phd * dt
        new = dict(st)
        new.update(X=X, psi=psi, V=V, r=r, th=th, thd=thd, ph=ph, phd=phd)
        ax_body = (rot2(psi).T @ acc)[0]
        row = dict(t=t + dt, X=X.copy(), psi=psi, V=V.copy(), r=r, th=th, thd=thd, thdd=thdd, ph=ph, phd=phd,
                   phdd=phdd, acc=acc.copy(), racc=racc, **extra)
        return new, row, ax_body

    # ---- solve the initial speed so the impact speed matches -----------
    def solve_initial_speed(self, T_pre):
        target = self.cfg.collision.impact_speed_kph * KPH
        if target <= 0.0:
            return 0.0
        dt = 1e-3

        def speed_at_impact(v0):
            st, _ = self._integrate(v0, T_pre, dt, record=False)
            return float(np.linalg.norm(st["V"]))

        br = self.cfg.environment.braking
        decel = min(br.deceleration_g, self.mu) * self.g if br.enabled else 0.0
        v_a = target + decel * min(br.onset_before_impact_s, T_pre)
        f_a = speed_at_impact(v_a) - target
        v_b = v_a + max(1.0, abs(f_a))
        f_b = speed_at_impact(v_b) - target
        for _ in range(25):
            if abs(f_b) < 2e-3 or f_b == f_a:
                break
            v_c = v_b - f_b * (v_b - v_a) / (f_b - f_a)
            v_c = max(v_c, 0.0)
            v_a, f_a = v_b, f_b
            v_b, f_b = v_c, speed_at_impact(v_c) - target
        return v_b

    # ---- crash interface ----------------------------------------------
    def _face_geometry(self):
        """Engaged region, contact point and engagement fraction on the struck face."""
        cfg = self.cfg
        col = cfg.collision
        veh = cfg.vehicle
        st = veh.structure
        lay = self.lay
        face = col.impact_direction
        pdef = partner_defaults(col.partner)
        wp = pdef["width_m"]
        W = veh.width_m
        half = 0.5 * W
        if face in ("frontal", "rear"):
            span = (-half, half)
            ov = col.overlap * W
            if col.partner.type == "pole":
                uc = col.impact_location_m if col.impact_location_m is not None else 0.0
                u_pc = uc
                region = span
            else:
                if col.overlap_side == "left":
                    region = (half - ov, half)
                    u_pc = half - ov + 0.5 * wp
                elif col.overlap_side == "right":
                    region = (-half, -half + ov)
                    u_pc = -half + ov - 0.5 * wp
                else:
                    region = (-0.5 * ov, 0.5 * ov)
                    u_pc = 0.0
                if wp >= 2.0 * W:  # wide barriers: their edge is at the overlap boundary
                    u_pc = (half - ov + 0.5 * wp) if col.overlap_side != "right" else (-half + ov - 0.5 * wp)
                    if col.overlap_side == "center" or col.overlap >= 0.999:
                        u_pc = 0.0
        else:
            span = (lay.x_rear_bumper + 0.45, lay.x_front_bumper - 0.55)
            uc = col.impact_location_m if col.impact_location_m is not None else float(lay.dog_ref[0])
            u_pc = uc
            region = span
        lo = max(region[0], u_pc - 0.5 * wp, span[0])
        hi = min(region[1], u_pc + 0.5 * wp, span[1])
        if hi <= lo:
            raise ValueError("the collision partner does not overlap the struck face; check overlap/impact_location")
        e, u_cent = structural_engagement(face, lo, hi, span, st, lay)
        # ride-height compatibility
        dh = abs(pdef["structure_height_m"] - st.structure_height_m)
        fh = min(1.0, max(0.35, 1.0 - max(0.0, dh - 0.10) / 0.40))
        e = max(0.08, min(1.0, e)) * fh
        if face == "frontal":
            p_body = np.array([lay.x_front_bumper, u_cent])
            t_body = np.array([0.0, 1.0])
        elif face == "rear":
            p_body = np.array([lay.x_rear_bumper, u_cent])
            t_body = np.array([0.0, 1.0])
        elif face == "left":
            p_body = np.array([u_cent, half])
            t_body = np.array([1.0, 0.0])
        else:
            p_body = np.array([u_cent, -half])
            t_body = np.array([1.0, 0.0])
        return dict(face=face, lo=lo, hi=hi, u_pc=u_pc, u_cent=u_cent, engagement=e, height_factor=fh,
                    p_body=p_body, t_body=t_body, n_body=np.array(FACE_NORMAL[face]), partner_width=wp,
                    engaged_width=hi - lo)

    def _subject_structure(self, geo):
        veh = self.cfg.vehicle
        st = veh.structure
        face = geo["face"]
        zones = {"frontal": veh.crush_zones_front, "rear": veh.crush_zones_rear,
                 "left": veh.crush_zones_side, "right": veh.crush_zones_side}[face]
        k = {"frontal": st.front_stiffness_kN_per_m, "rear": st.rear_stiffness_kN_per_m,
             "left": st.side_stiffness_kN_per_m, "right": st.side_stiffness_kN_per_m}[face] * 1e3
        e = geo["engagement"]
        return ElastoPlastic([(z.depth_m, z.force_kN * 1e3) for z in zones], k, st.compartment_stiffness_kN_per_m * 1e3,
                             st.unloading_stiffness_factor, scale=e, backing_scale=max(e, 0.35))

    def _partner_structure(self, geo):
        p = self.cfg.collision.partner
        d = partner_defaults(p)
        uf = self.cfg.vehicle.structure.unloading_stiffness_factor
        ew = geo["engaged_width"]
        if p.type in ("rigid_barrier", "pole"):
            return None
        k = d["stiffness_kN_per_m"] * 1e3
        if p.type == "deformable_barrier":
            fc = (d["crush_force_kN"] * 1e3) if d["crush_force_kN"] else 0.342e6 * 0.65 * max(ew, 0.2)
            return ElastoPlastic([(d["crush_depth_m"], fc)], k, 5.0e7, uf)
        if p.type in ("moving_barrier", "truck"):
            frac = min(1.0, ew / d["width_m"])
            return ElastoPlastic([(d["crush_depth_m"], d["crush_force_kN"] * 1e3)], k, 3.0e7, uf,
                                 scale=max(frac, 0.2), backing_scale=max(frac, 0.5))
        # another vehicle: generic structure scaled by mass and engagement
        ratio = d["mass_kg"] / 1600.0
        base = self.cfg.vehicle.__class__()
        face = p.engaged_face
        zones = {"front": base.crush_zones_front, "rear": base.crush_zones_rear, "side": base.crush_zones_side}[face]
        kface = {"front": d["stiffness_kN_per_m"], "rear": 1000.0, "side": 700.0}[face] * 1e3 * ratio
        frac = min(1.0, ew / d["width_m"])
        if face in ("front", "rear"):
            e = 0.3 * frac + (0.7 if frac > 0.62 else (0.35 if frac > 0.25 else 0.1))
        else:
            e = 0.45 * frac + 0.35
        e = min(1.0, e) * geo["height_factor"]
        depth_scale = (d["crush_depth_m"] / 0.70) if face == "front" else 1.0
        return ElastoPlastic([(z.depth_m * depth_scale, z.force_kN * 1e3 * ratio) for z in zones], kface,
                             1.0e7 * ratio, uf, scale=e, backing_scale=max(e, 0.5))

    # ---- full run -------------------------------------------------------
    def run(self) -> VehicleMotion:
        cfg = self.cfg
        col = cfg.collision
        T_pre = cfg.environment.pre_impact_duration_s
        T_post = col.duration_after_impact_s
        v0 = self.solve_initial_speed(T_pre)
        st0, rows_pre = self._integrate(v0, T_pre, 1e-3, record=True)
        dt = cfg.simulation.vehicle_timestep_s
        info = {"initial_speed_kph": v0 / KPH, "road_friction": self.mu}
        psi0 = st0["psi"]
        X0 = st0["X"].copy()
        V0 = st0["V"].copy()
        info["impact_speed_kph"] = float(np.linalg.norm(V0)) / KPH
        info["sideslip_at_impact_deg"] = math.degrees(math.atan2(*(rot2(psi0).T @ V0)[::-1])) if np.linalg.norm(V0) > 0.1 else 0.0

        if col.mode == "sled":
            crash, pinfo = self._make_sled(st0)
        else:
            crash, pinfo = self._make_crash(st0)
        info.update(pinfo)
        state = st0
        ax_prev = 0.0
        rows = []
        t = 0.0
        n = int(round(T_post / dt))
        for _ in range(n):
            state, row, ax_prev = self._step(t, dt, state, ax_prev, crash)
            rows.append(row)
            t += dt
        return self._assemble(rows_pre, rows, X0, psi0, V0, info)

    def _make_sled(self, st0):
        acc_fn, T = sled_pulse(self.cfg)
        th = math.radians(self.cfg.collision.impact_angle_deg)
        n_body = np.array(FACE_NORMAL[self.cfg.collision.impact_direction])
        dirb = -(rot2(th) @ n_body)
        m = self.m

        def crash(t, st):
            a = acc_fn(t)
            Fw = rot2(st["psi"]) @ (dirb * a * m)
            return Fw, 0.0, {"force": a * m, "crush_s": 0.0, "crush_p": 0.0, "Pp": np.zeros(2), "psip": 0.0}

        dv = self.cfg.collision.sled.delta_v_kph
        return crash, {"mode": "sled", "pulse_duration_s": T, "pulse_delta_v_kph": dv, "partner_type": "sled"}

    def _make_crash(self, st0):
        col = self.cfg.collision
        geo = self._face_geometry()
        subj = self._subject_structure(geo)
        part = self._partner_structure(geo)
        iface = SeriesInterface(subj, part)
        pdef = partner_defaults(col.partner)
        psi_s = st0["psi"]
        Rs = rot2(psi_s)
        theta = math.radians(col.impact_angle_deg)
        n_w0 = rot2(psi_s + theta) @ geo["n_body"]
        t_w0 = rot2(psi_s + theta) @ geo["t_body"]
        P_s_body = geo["p_body"]
        P_s_w = st0["X"] + Rs @ P_s_body
        fixed = col.partner.type in ("rigid_barrier", "deformable_barrier", "pole")
        Lp, Wp = pdef["length_m"], pdef["width_m"]
        if fixed:
            psi_p = math.atan2(-n_w0[1], -n_w0[0])
            face_c = P_s_w + t_w0 * (geo["u_pc"] - geo["u_cent"])
            Xp = face_c - n_w0 * (0.5 * Lp if col.partner.type != "pole" else 0.5 * Wp)
            Vp = np.zeros(2)
            mp = math.inf
            Ip = math.inf
            Pp_body = rot2(psi_p).T @ (P_s_w - Xp)
        else:
            pf = col.partner.engaged_face
            n_pf = np.array(PARTNER_FACE_NORMAL[pf])
            psi_p = math.atan2(-n_w0[1], -n_w0[0]) - math.atan2(n_pf[1], n_pf[0])
            Rp = rot2(psi_p)
            f_body = {"front": np.array([0.5 * Lp, 0.0]), "rear": np.array([-0.5 * Lp, 0.0]),
                      "side": np.array([0.0, -0.5 * Wp])}[pf]
            face_c = P_s_w + t_w0 * (geo["u_pc"] - geo["u_cent"])
            Xp = face_c - Rp @ f_body
            Vp = col.partner.speed_kph * KPH * np.array([math.cos(psi_p), math.sin(psi_p)])
            mp = pdef["mass_kg"]
            Ip = 0.0625 * mp * (Lp * Lp + Wp * Wp)
            Pp_body = Rp.T @ (P_s_w - Xp)
        pst = dict(X=Xp.copy(), psi=psi_p, V=Vp.copy(), r=0.0)
        mu_i = col.interface_friction
        mu_road = self.mu
        h_s = self.cfg.vehicle.structure.structure_height_m
        z_force = 0.5 * (h_s + pdef["structure_height_m"]) if not fixed else h_s
        last = {"t": -1.0}
        glance_dir = None
        if col.partner.type != "pole" and col.impact_direction in ("frontal", "rear") and col.overlap < 0.95:
            # sliding towards the far side of the overlap reduces the engaged width
            sgn = {"left": -1.0, "right": 1.0}.get(col.overlap_side)
            glance_dir = sgn

        def crash(t, st):
            if last["t"] == t:
                return last["out"]
            Rs_ = rot2(st["psi"])
            Ps = st["X"] + Rs_ @ P_s_body
            if fixed:
                n = n_w0
                Pp = pst["X"] + rot2(pst["psi"]) @ Pp_body
            else:
                n = Rs_ @ geo["n_body"]
                Pp = pst["X"] + rot2(pst["psi"]) @ Pp_body
            d = float(np.dot(Ps - Pp, n))
            Fn = iface.solve(d)
            if glance_dir is not None:
                # the subject slides off the edge of a partial-overlap partner
                tn_ = np.array([-n[1], n[0]])
                s_out = float(np.dot(Ps - Pp, tn_)) * glance_dir
                Fn *= min(1.0, max(0.0, 1.0 - max(0.0, s_out) / max(geo["engaged_width"], 0.05)))
            Pc = Ps - n * iface.ds
            rs = Pc - st["X"]
            vs = st["V"] + st["r"] * np.array([-rs[1], rs[0]])
            if fixed:
                vp = np.zeros(2)
            else:
                rp_ = Pc - pst["X"]
                vp = pst["V"] + pst["r"] * np.array([-rp_[1], rp_[0]])
            vrel = vs - vp
            tn = np.array([-n[1], n[0]])
            vt = float(np.dot(vrel, tn))
            Ft = -mu_i * Fn * vt / math.sqrt(vt * vt + 0.05 ** 2)
            Fw = -Fn * n + Ft * tn
            Ms = cross2(rs, Fw)
            if not fixed:
                # partner dynamics (free-rolling tyres)
                rp_ = Pc - pst["X"]
                Mp = cross2(rp_, -Fw)
                Rp_ = rot2(pst["psi"])
                vbp = Rp_.T @ pst["V"]
                Fz = mp * self.g * 0.5
                C = 1.5e5 * mp / 1600.0
                fxf, fyf = axle_forces(vbp[0], vbp[1] + 0.4 * Lp * pst["r"], Fz, mu_road, 0.0, 0.0, True, C)
                fxr, fyr = axle_forces(vbp[0], vbp[1] - 0.4 * Lp * pst["r"], Fz, mu_road, 0.0, 0.0, True, C)
                Ftp = Rp_ @ np.array([fxf + fxr, fyf + fyr])
                Mtp = 0.4 * Lp * (fyf - fyr)
                pst["V"] = pst["V"] + (-Fw + Ftp) / mp * dt_
                pst["r"] = pst["r"] + (Mp + Mtp) / Ip * dt_
                pst["X"] = pst["X"] + pst["V"] * dt_
                pst["psi"] = pst["psi"] + pst["r"] * dt_
            out = (Fw, Ms, {"force": Fn, "crush_s": iface.ds, "crush_p": iface.dp, "Pp": pst["X"].copy(),
                            "psip": pst["psi"], "z_force": z_force})
            last["t"] = t
            last["out"] = out
            return out

        dt_ = self.cfg.simulation.vehicle_timestep_s
        pinfo = {"mode": "crash", "partner_type": col.partner.type, "engagement": geo["engagement"],
                 "engaged_width_m": geo["engaged_width"], "height_factor": geo["height_factor"],
                 "contact_point_body": geo["p_body"].tolist(), "partner_mass_kg": None if fixed else mp,
                 "partner_initial_pos": Xp.tolist(), "partner_yaw": psi_p, "partner_length_m": Lp,
                 "partner_width_m": Wp, "subject_crush_capacity_m": subj.total_depth,
                 "struck_face": geo["face"], "impact_location_u": geo["u_pc"]}
        return crash, pinfo

    # ---- assemble the output -------------------------------------------
    def _assemble(self, rows_pre, rows, X0, psi0, V0, info) -> VehicleMotion:
        cfg = self.cfg
        allrows = rows_pre + rows
        t = np.array([r_["t"] for r_ in allrows])
        # world re-expressed so the subject is at the origin, heading 0, at t = 0
        R0 = rot2(-psi0)
        XY = np.array([R0 @ (r_["X"] - X0) for r_ in allrows])
        psi = np.array([r_["psi"] - psi0 for r_ in allrows])
        Vw = np.array([R0 @ r_["V"] for r_ in allrows])
        Aw = np.array([R0 @ r_["acc"] for r_ in allrows])
        th = np.array([r_["th"] for r_ in allrows])
        ph = np.array([r_["ph"] for r_ in allrows])
        r_ = np.array([r__["r"] for r__ in allrows])
        thd = np.array([r__["thd"] for r__ in allrows])
        phd = np.array([r__["phd"] for r__ in allrows])
        N = len(t)
        h = self.h
        pos = np.column_stack([XY, np.full(N, h)])
        vel = np.column_stack([Vw, np.zeros(N)])
        acc = np.column_stack([Aw, np.zeros(N)])
        euler = np.column_stack([ph, th, psi])
        # body angular velocity from Euler rates (ZYX)
        sf, cf = np.sin(ph), np.cos(ph)
        st_, ct = np.sin(th), np.cos(th)
        omega = np.column_stack([phd - r_ * st_, thd * cf + r_ * ct * sf, -thd * sf + r_ * ct * cf])
        alpha = np.gradient(omega, t, axis=0)
        R = np.stack([euler_to_R(e) for e in euler])
        RT = np.transpose(R, (0, 2, 1))
        acc_body = np.einsum("nij,nj->ni", RT, acc)
        d = np.array([0.0, 0.0, -h])
        acc_origin = acc_body + np.cross(alpha, d) + np.cross(omega, np.cross(omega, d))
        g = cfg.environment.gravity
        gravity = np.einsum("nij,j->ni", RT, np.array([0.0, 0.0, -g]))

        def col(name, default=0.0):
            return np.array([r__.get(name, default) for r__ in allrows], dtype=float)

        crush_s = col("crush_s")
        crush_p = col("crush_p")
        force = col("force")
        face = cfg.collision.impact_direction
        rc = cfg.vehicle.rear_compartment
        lay = self.lay
        zero = np.zeros(N)
        intr_l, intr_r, intr_rear, crush_front = zero.copy(), zero.copy(), zero.copy(), zero.copy()
        if cfg.collision.mode == "crash":
            if face in ("left", "right"):
                lo = info.get("impact_location_u", lay.dog_ref[0]) - 0.5 * info.get("partner_width_m", 1.0)
                hi = lo + info.get("partner_width_m", 1.0)
                x = lay.dog_ref[0]
                dx = 0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi))
                lam = math.exp(-(dx / 0.45) ** 2)
                intr = lam * np.maximum(0.0, crush_s - rc.door_crush_space_m)
                if face == "left":
                    intr_l = intr
                else:
                    intr_r = intr
            elif face == "rear":
                intr_rear = np.maximum(0.0, crush_s - rc.rear_crush_space_m)
            else:
                crush_front = crush_s
        # partner
        if "Pp" in (rows[0] if rows else {}):
            Pp = np.array([R0 @ (r__.get("Pp", np.zeros(2)) - X0) if "Pp" in r__ else np.full(2, np.nan)
                           for r__ in allrows])
            psip = np.array([r__.get("psip", np.nan) - psi0 for r__ in allrows])
            pre = np.isnan(Pp[:, 0])
            if col_partner_moving(cfg) and rows:
                # partner trajectory before impact: straight line at constant speed
                Pp0 = Pp[~pre][0]
                psi_p0 = psip[~pre][0]
                vp = cfg.collision.partner.speed_kph * KPH * np.array([math.cos(psi_p0), math.sin(psi_p0)])
                Pp[pre] = Pp0[None, :] + t[pre, None] * vp[None, :]
            else:
                Pp[pre] = Pp[~pre][0] if (~pre).any() else 0.0
            psip[pre] = psip[~pre][0] if (~pre).any() else 0.0
        else:
            Pp = np.full((N, 2), 1e3)
            psip = np.zeros(N)
        if cfg.collision.mode == "sled":
            Pp = np.full((N, 2), 1e3)
        partner_pos = np.column_stack([Pp, np.zeros(N)])

        post = t >= 0.0
        vb = np.einsum("nij,nj->ni", RT, vel)
        v_imp = vb[np.argmin(np.abs(t))]
        info.update(self._severity(t, acc_body, vb, crush_s, crush_p, force, post, v_imp))
        return VehicleMotion(t=t, pos=pos, euler=euler, vel=vel, acc_cg_world=acc, omega=omega, alpha=alpha,
                             acc_origin=acc_origin, gravity=gravity, R=R, acc_body=acc_body, crush_subject=crush_s,
                             crush_partner=crush_p, interface_force=force, intrusion_left=intr_l,
                             intrusion_right=intr_r, intrusion_rear=intr_rear, crush_front=crush_front,
                             partner_pos=partner_pos, partner_yaw=psip, cg_height=h, info=info)

    def _severity(self, t, acc_body, vb, crush_s, crush_p, force, post, v_imp):
        out = {}
        tp = t[post]
        ab = acc_body[post]
        if len(tp) < 3:
            return out
        dv_vec = vb[post][-1] - v_imp
        # delta-V at the time the interface force first returns to zero (end of contact)
        f = force[post]
        contact = np.nonzero(f > 1.0)[0]
        if len(contact):
            end = contact[-1]
            out["contact_duration_ms"] = float(tp[end] * 1e3)
            dv_c = vb[post][end] - v_imp
        else:
            out["contact_duration_ms"] = 0.0
            dv_c = dv_vec
        out["delta_v_kph"] = float(np.linalg.norm(dv_c) / KPH)
        out["delta_v_x_kph"] = float(dv_c[0] / KPH)
        out["delta_v_y_kph"] = float(dv_c[1] / KPH)
        # 60 Hz-ish smoothing for peak (moving average over 5 ms)
        k = max(1, int(round(0.005 / max(tp[1] - tp[0], 1e-6))))
        ker = np.ones(k) / k
        ax = np.convolve(ab[:, 0], ker, mode="same")
        ay = np.convolve(ab[:, 1], ker, mode="same")
        res = np.hypot(ax, ay)
        out["peak_accel_g"] = float(res.max() / G)
        out["peak_accel_x_g"] = float(np.abs(ax).max() / G)
        out["peak_accel_y_g"] = float(np.abs(ay).max() / G)
        out["time_of_peak_ms"] = float(tp[np.argmax(res)] * 1e3)
        out["max_crush_subject_m"] = float(crush_s.max())
        out["max_crush_partner_m"] = float(crush_p.max())
        out["peak_interface_force_kN"] = float(force.max() / 1e3)
        n_body = np.array(FACE_NORMAL[self.cfg.collision.impact_direction] + (0.0,))
        out["olc_g"] = occupant_load_criterion(tp, -(vb[post] - v_imp) @ n_body)
        return out


def col_partner_moving(cfg) -> bool:
    return cfg.collision.mode == "crash" and cfg.collision.partner.type not in ("rigid_barrier", "deformable_barrier", "pole")


def occupant_load_criterion(t, u) -> float:
    """Occupant Load Criterion (g).

    ``u(t)`` is the velocity of a free occupant relative to the vehicle
    (positive towards the struck face).  The occupant flies freely for
    65 mm, then decelerates uniformly so that it reaches the vehicle speed
    after a further 235 mm; that uniform deceleration is the OLC.
    """
    if len(t) < 3:
        return 0.0
    s = np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(t))])
    i1 = int(np.searchsorted(s, 0.065))
    if i1 >= len(t) - 1:
        return 0.0
    for i2 in range(i1 + 1, len(t)):
        dt_ = t[i2] - t[i1]
        s2 = s[i2] - 0.5 * u[i2] * dt_
        if s2 >= 0.300:
            return float(max(u[i2], 0.0) / dt_ / G)
    return 0.0


def run_vehicle_model(cfg: ScenarioConfig) -> VehicleMotion:
    return CrashModel(cfg).run()
