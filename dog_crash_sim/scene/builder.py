"""Build the MuJoCo occupant scene in the vehicle frame.

The MuJoCo world frame *is* the vehicle frame (x forward, y left, z up,
origin on the ground below the CG).  Static interior parts are geoms of
the world body.  Parts that move relative to the vehicle are bodies:

* the dog (free joint) wearing the harness frame (6-DoF bushing);
* the rear seat back (hinge, elasto-plastic moment applied in Python);
* cargo items (free joints) and the cargo net / barrier (slide joint);
* intruding door trims, the trunk rear wall and the airbags, whose motion
  is prescribed from the vehicle crash model;
* mocap bodies for the road surface, the collision partner and the
  crushing front/rear end (visual only).

Collision bitmasks: dog geoms keep the model's contype 1; vehicle geoms
use contype 2 / conaffinity 1 (collide with the dog and with cargo); cargo
uses contype 4 / conaffinity 7.  Vehicle geoms carry a higher priority so
their friction and softness govern dog-vehicle contacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from ..config import ScenarioConfig, dog_mass_and_height, partner_defaults
from ..biophysics.dog_model import DogScaling, load_dog_spec, scale_dog_spec
from ..biophysics.posture import POSTURES, default_facing, quat_from_euler
from .harness import HarnessGeometry, build_geometry, bushing_params
from .layout import Layout, belt_anchor_points, compute_layout

GEOM = mujoco.mjtGeom
VEH_CONTYPE, VEH_CONAFF = 2, 1
CARGO_CONTYPE, CARGO_CONAFF = 4, 7

DOG_PARTS = ["head", "neck", "thorax", "forelimbs", "abdomen_lumbar", "pelvis_hindlimbs", "tail"]
VEH_PARTS = ["rear_cushion", "rear_seat_back", "head_restraint", "front_seat_backs", "console", "door_left",
             "door_right", "window_left", "window_right", "pillars_roof", "floor", "trunk", "cargo", "airbag",
             "cargo_net_barrier"]

TRIM = {"soft": ((0.012, 1.0), (0.55, 0.95, 0.03)), "medium": ((0.006, 1.0), (0.8, 0.95, 0.02)),
        "hard": ((0.003, 1.0), (0.9, 0.99, 0.01))}


def dog_part_of(body_name: str) -> str:
    if body_name in ("skull", "jaw"):
        return "head"
    if body_name.startswith("C_"):
        return "neck"
    if body_name in ("torso", "scapula_L", "scapula_R", "harness"):
        return "thorax"
    if any(body_name.startswith(p) for p in ("upper_arm", "lower_arm", "hand", "finger")):
        return "forelimbs"
    if body_name.startswith("L_"):
        return "abdomen_lumbar"
    if body_name.startswith("Ca_"):
        return "tail"
    return "pelvis_hindlimbs"


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)):
    pos, target, up = map(lambda v: np.asarray(v, float), (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    return np.concatenate([right, upv]).tolist()


def quat_y(angle):
    return [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]


def quat_z(angle):
    return [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]


def _clean_joint(j, limited=False, rng=None):
    """Reset properties inherited from the dog model's default joint class."""
    j.stiffness = [0.0, 0.0, 0.0]
    j.damping = [0.0, 0.0, 0.0]
    j.armature = 0.0
    j.springref = 0.0
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE if limited else mujoco.mjtLimited.mjLIMITED_FALSE
    if rng is not None:
        j.range = list(rng)
    return j


@dataclass
class TendonInfo:
    name: str
    kind: str                 # tether | belt | strap | cargo_strap
    tendon: int = -1
    actuator: int = -1
    width: float = 0.03       # m (pressure estimates)
    radius: float = 0.1       # m, local body radius under the strap (pressure estimates)
    strength: float = math.inf
    connector_strength: float = math.inf
    meta: dict = field(default_factory=dict)


@dataclass
class Scene:
    cfg: ScenarioConfig
    model: mujoco.MjModel
    layout: Layout
    scaling: DogScaling
    harness: Optional[HarnessGeometry]
    anchors: dict
    dog_bodies: np.ndarray
    dog_joints: np.ndarray
    dog_actuators: np.ndarray
    dog_init_qpos: np.ndarray
    dog_qpos_slice: slice
    dog_qvel_slice: slice
    harness_body: int
    harness_joints: list
    bushing: dict
    tendons: list
    seatback_joint: int
    prescribed: list
    airbags: list
    cargo: list
    cargo_restraint: dict
    mocap: dict
    geom_dog_part: np.ndarray
    geom_veh_part: np.ndarray
    dynamic_bodies: np.ndarray
    sensors: dict
    cameras: list
    ring_site: str = "harness_ring"
    attachment: str = "none"
    notes: list = field(default_factory=list)


class SceneBuilder:
    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg
        self.lay = compute_layout(cfg)
        self.notes = []

    # ------------------------------------------------------------------
    def build(self) -> Scene:
        cfg = self.cfg
        rp = cfg.rear_passenger
        mass, height = dog_mass_and_height(rp)
        spec = load_dog_spec(rp.model_path)
        self.spec = spec
        self.sc = scale_dog_spec(spec, mass, height, rp.model_path, rp.mass_distribution)
        self._options()
        self._assets()
        # neutral-pose world positions of the scaled dog bodies (identity orientations in dog.xml)
        m0 = spec.compile()
        d0 = mujoco.MjData(m0)
        mujoco.mj_forward(m0, d0)
        self.body_pos0 = {m0.body(i).name: d0.xpos[i].copy() for i in range(m0.nbody)}
        self.body_quat0 = {m0.body(i).name: d0.xquat[i].copy() for i in range(m0.nbody)}
        self.dog_body_names = [m0.body(i).name for i in range(1, m0.nbody)]
        self.dog_joint_names = [m0.joint(j).name for j in range(m0.njnt) if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.dog_act_names = [m0.actuator(a).name for a in range(m0.nu)]
        skin_vert = np.asarray(spec.skin("skin").vert)
        self.hgeo = build_geometry(skin_vert, self.sc, cfg.harness)
        self.bush = bushing_params(cfg.harness, self.sc)

        self.anchors = belt_anchor_points(cfg, self.lay)
        self.veh_part_of_geom = {}
        self.prescribed = []
        self.airbags = []
        self.cargo = []
        self.cargo_restraint = {"method": cfg.trunk_cargo.restraint_method}
        self.tendons = []

        for leg in ("upper_leg_L", "upper_leg_R"):
            for b in ("torso", "L_3", "L_4", "L_5", "L_6", "L_7"):
                spec.add_exclude(bodyname1=leg, bodyname2=b)
        self._sensors_on_dog()
        self._interior()
        self._seat_back()
        self._trunk()
        self._doors()
        self._airbags()
        self._harness()
        self._restraint_tendons()
        self._cargo()
        self._exterior_and_world()
        self._cameras()
        model = spec.compile()
        return self._finish(model)

    # ------------------------------------------------------------------
    def _options(self):
        o = self.spec.option
        o.timestep = self.cfg.simulation.timestep_s
        o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        o.noslip_iterations = 0
        o.iterations = 100
        o.ls_iterations = 50
        o.tolerance = 1e-8
        self.spec.visual.global_.offwidth = max(self.cfg.simulation.render.width, 1280)
        self.spec.visual.global_.offheight = max(self.cfg.simulation.render.height, 720)
        self.spec.visual.map.znear = 0.005
        self.spec.visual.map.zfar = 80.0
        self.spec.stat.extent = 3.0
        self.spec.stat.meansize = 0.05

    def _assets(self):
        s = self.spec
        s.add_texture(name="road", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                      rgb1=[0.32, 0.32, 0.34], rgb2=[0.26, 0.26, 0.28], width=256, height=256,
                      mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.9, 0.9, 0.9])
        s.add_material(name="road", textures=["", "road"], texrepeat=[20, 20], texuniform=True, reflectance=0.05)
        mats = {
            "trim": [0.20, 0.21, 0.23, 1.0], "seat": [0.16, 0.17, 0.20, 1.0], "headrest": [0.22, 0.22, 0.26, 1.0],
            "glass": [0.6, 0.75, 0.85, 0.12], "door": [0.30, 0.31, 0.34, 0.25], "paint": [0.70, 0.08, 0.06, 0.08],
            "tyre": [0.05, 0.05, 0.05, 0.35], "concrete": [0.55, 0.55, 0.52, 1.0], "alu": [0.75, 0.75, 0.78, 1.0],
            "partner": [0.15, 0.35, 0.75, 0.9], "cargo": [0.55, 0.38, 0.20, 1.0], "airbag": [0.92, 0.92, 0.88, 0.85],
            "net": [0.1, 0.1, 0.1, 0.5], "belt": [0.08, 0.08, 0.08, 1.0], "anchor": [0.8, 0.8, 0.1, 1.0],
            "roof": [0.25, 0.25, 0.27, 0.10], "floor": [0.12, 0.12, 0.13, 1.0],
        }
        for name, rgba in mats.items():
            s.add_material(name=name, rgba=rgba)
        h = self.cfg.harness
        s.add_material(name="harness", rgba=list(h.rgba), specular=0.2)
        s.add_material(name="hardware", rgba=[0.7, 0.7, 0.72, 1.0], specular=0.8, shininess=0.8)

    # ---- helpers -----------------------------------------------------
    def _veh_geom(self, body, part, name, gtype, size, pos, quat=None, material="trim", trim=None, friction=None,
                  contact=True, group=1):
        rs = self.cfg.restraints.seat
        trim = trim or rs.trim_stiffness
        solref, solimp = TRIM[trim]
        g = body.add_geom(name=name, type=gtype, size=list(size), pos=list(pos), material=material, group=group)
        if quat is not None:
            g.quat = list(quat)
        g.contype = VEH_CONTYPE if contact else 0
        g.conaffinity = VEH_CONAFF if contact else 0
        g.condim = 3
        g.priority = 3
        g.solref = list(solref)
        g.solimp = list(solimp) + [0.5, 2.0]
        g.friction = [friction if friction is not None else 0.5, 0.005, 0.0001]
        g.mass = 0.0 if body.name == "world" else g.mass
        self.veh_part_of_geom[name] = part
        return g

    def _cushion_contact(self):
        rs = self.cfg.restraints.seat
        m_eff = max(self.sc.body_mass / 2.0, 1.0)
        tc = math.sqrt(m_eff / (rs.cushion_stiffness_kN_per_m * 1e3))
        tc = max(tc, 3.0 * self.cfg.simulation.timestep_s)
        return (tc, max(rs.cushion_damping_ratio, 0.2)), (0.5, 0.95, 0.05, 0.5, 2.0)

    # ---- dog sensors ------------------------------------------------------
    def _sensors_on_dog(self):
        sp = self.spec
        s = self.sc.length_scale

        def axis_quat(z_axis):
            z = np.asarray(z_axis, float)
            z /= np.linalg.norm(z)
            x = np.array([1.0, 0, 0]) - z * z[0]
            if np.linalg.norm(x) < 1e-6:
                x = np.array([0, 0, 1.0]) - z * z[2]
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            R = np.column_stack([x, y, z])
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, R.flatten())
            return q.tolist()

        P = self.body_pos0
        lower_neck_axis = P["C_1"] - P["C_7"]
        upper_neck_axis = P["skull"] - P["C_1"]
        jpos_c7 = np.asarray(sp.joint("C_7_extend").pos)
        jpos_atlas = np.asarray(sp.joint("atlas").pos)
        sp.body("C_7").add_site(name="lower_neck_lc", pos=jpos_c7.tolist(), quat=axis_quat(lower_neck_axis),
                                size=[0.01 * s, 0, 0], group=5)
        sp.body("skull").add_site(name="upper_neck_lc", pos=jpos_atlas.tolist(), quat=axis_quat(upper_neck_axis),
                                  size=[0.01 * s, 0, 0], group=5)
        sp.body("L_1").add_site(name="thoracolumbar_lc", pos=(np.array([0.012, 0, 0.01]) * s).tolist(),
                                size=[0.01 * s, 0, 0], group=5)
        sp.body("pelvis").add_site(name="lumbosacral_lc", pos=[0, 0, 0], size=[0.01 * s, 0, 0], group=5)
        self.sensor_names = []
        for nm, body in (("head_acc", "skull"), ("chest_acc", "torso"), ("pelvis_acc", "pelvis")):
            b = sp.body(body)
            ipos = np.asarray(b.ipos) if b.explicitinertial else np.zeros(3)
            b.add_site(name=nm + "_site", pos=ipos.tolist(), size=[0.008 * s, 0, 0], group=5)
            sp.add_sensor(name=nm, type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE,
                          objname=nm + "_site")
            self.sensor_names.append(nm)
        for lc in ("lower_neck", "upper_neck", "thoracolumbar", "lumbosacral"):
            sp.add_sensor(name=lc + "_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                          objname=lc + "_lc")
            sp.add_sensor(name=lc + "_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                          objname=lc + "_lc")
            self.sensor_names += [lc + "_force", lc + "_torque"]

    # ---- interior -------------------------------------------------------
    def _interior(self):
        cfg = self.cfg
        rc = cfg.vehicle.rear_compartment
        lay = self.lay
        wb = self.spec.worldbody
        rs = cfg.restraints.seat
        cw = rc.cabin_width_m
        # rear cushion (one bench)
        ca = lay.cushion_angle
        L = rc.cushion_length_m + 0.06
        depth = max(rc.cushion_height_m - rc.floor_height_m, 0.15)
        center_top = lay.bight + 0.5 * L * np.array([math.cos(ca), math.sin(ca)])
        nrm = np.array([-math.sin(ca), math.cos(ca)])
        c = center_top - nrm * 0.5 * depth
        g = self._veh_geom(wb, "rear_cushion", "rear_cushion", GEOM.mjGEOM_BOX, [0.5 * L, 0.5 * rc.seat_width_m, 0.5 * depth],
                           [c[0], 0.0, c[1]], quat_y(-ca), material="seat", friction=rs.cushion_friction)
        solref, solimp = self._cushion_contact()
        g.solref = list(solref)
        g.solimp = list(solimp)
        # side bolsters / wheel-arch trim beside the bench
        for sgn, part in ((1, "door_left"), (-1, "door_right")):
            self._veh_geom(wb, part, f"bench_side_{'L' if sgn > 0 else 'R'}", GEOM.mjGEOM_BOX,
                           [0.5 * L, 0.5 * max(0.5 * cw - 0.5 * rc.seat_width_m, 0.02) + 0.01, 0.5 * depth],
                           [c[0], sgn * (0.5 * rc.seat_width_m + 0.5 * max(0.5 * cw - 0.5 * rc.seat_width_m, 0.02)),
                            c[1]], quat_y(-ca), material="trim")
        # rear floor / foot well
        x0 = lay.cushion_front[0] - 0.25
        x1 = rc.front_seat_back_x_m + 0.55
        self._veh_geom(wb, "floor", "rear_floor", GEOM.mjGEOM_BOX, [0.5 * (x1 - x0), 0.5 * cw, 0.05],
                       [0.5 * (x0 + x1), 0, rc.floor_height_m - 0.05], material="floor", friction=0.7)
        self._veh_geom(wb, "floor", "tunnel", GEOM.mjGEOM_BOX, [0.5 * (x1 - x0), 0.12, 0.5 * rc.tunnel_height_m + 0.02],
                       [0.5 * (x0 + x1), 0, rc.floor_height_m + 0.5 * rc.tunnel_height_m - 0.02], material="floor")
        # front seats
        fx = rc.front_seat_back_x_m
        top = rc.front_seat_back_top_m
        bottom = rc.floor_height_m + 0.22
        rec = math.radians(22.0)
        hgt = (top - bottom) / math.cos(rec)
        fw = rc.front_seat_width_m
        yc = 0.5 * rc.console_width_m + 0.5 * fw
        for sgn, tag in (((1, "L"), (-1, "R")) if rc.front_seats_present else ()):
            mid_z = 0.5 * (top + bottom)
            # rear face at fx at mid height, reclined (top further back)
            cx = fx + 0.06
            self._veh_geom(wb, "front_seat_backs", f"front_seat_back_{tag}", GEOM.mjGEOM_BOX, [0.06, 0.5 * fw, 0.5 * hgt],
                           [cx, sgn * yc, mid_z], quat_y(-rec), material="seat", trim="soft")
            self._veh_geom(wb, "front_seat_backs", f"front_headrest_{tag}", GEOM.mjGEOM_BOX, [0.06, 0.13, 0.11],
                           [cx - 0.5 * hgt * math.sin(rec) - 0.01, sgn * yc, top + 0.12], quat_y(-rec), material="headrest",
                           trim="soft")
            self._veh_geom(wb, "front_seat_backs", f"front_cushion_{tag}", GEOM.mjGEOM_BOX, [0.26, 0.5 * fw, 0.08],
                           [fx + 0.38, sgn * yc, bottom + 0.06], material="seat")
        if rc.front_seats_present:
            self._veh_geom(wb, "console", "console", GEOM.mjGEOM_BOX, [0.30, 0.5 * rc.console_width_m, 0.20],
                           [fx + 0.30, 0, rc.floor_height_m + 0.20], material="trim")
        # pillars and roof
        rz = rc.roof_height_m
        bl = rc.beltline_height_m
        for sgn, tag in (((1, "L"), (-1, "R")) if rc.roof_present else ()):
            y = sgn * (0.5 * cw + 0.03)
            self._veh_geom(wb, "pillars_roof", f"b_pillar_{tag}", GEOM.mjGEOM_BOX, [0.07, 0.04, 0.5 * (rz - rc.floor_height_m)],
                           [lay.b_pillar_x, y, 0.5 * (rz + rc.floor_height_m)], material="trim", trim="soft")
            self._veh_geom(wb, "pillars_roof", f"c_pillar_{tag}", GEOM.mjGEOM_BOX, [0.10, 0.04, 0.5 * (rz - bl)],
                           [lay.c_pillar_x - 0.04, y, 0.5 * (rz + bl)], material="trim", trim="soft")
            # window glass of the rear door
            xg0, xg1 = lay.c_pillar_x + 0.06, lay.b_pillar_x - 0.07
            self._veh_geom(wb, "window_left" if sgn > 0 else "window_right", f"rear_window_{tag}", GEOM.mjGEOM_BOX,
                           [0.5 * (xg1 - xg0), 0.006, 0.5 * (rz - bl)], [0.5 * (xg0 + xg1), sgn * (0.5 * cw + 0.05),
                                                                       0.5 * (rz + bl)],
                           material="glass", trim="hard", group=2)
        xr0 = lay.trunk_rear_x if not rc.parcel_shelf else lay.seat_back_top[0] - 0.25
        xr1 = rc.front_seat_back_x_m + 1.4
        if rc.roof_present:
            self._veh_geom(wb, "pillars_roof", "roof", GEOM.mjGEOM_BOX, [0.5 * (xr1 - xr0), 0.5 * cw + 0.05, 0.03],
                           [0.5 * (xr0 + xr1), 0, rz + 0.03], material="roof", group=2, trim="soft")

    def _seat_back(self):
        cfg = self.cfg
        rc = cfg.vehicle.rear_compartment
        rs = cfg.restraints.seat
        hr = cfg.restraints.head_restraints
        lay = self.lay
        beta = lay.seat_back_angle
        sb = self.spec.worldbody.add_body(name="rear_seat_back", pos=[lay.seat_back_hinge[0], 0, lay.seat_back_hinge[1]],
                                          quat=quat_y(-beta))
        t = rc.seat_back_thickness_m
        H = rc.seat_back_height_m
        W = rc.seat_width_m
        sb.mass = rs.seat_back_mass_kg
        sb.explicitinertial = True
        sb.ipos = [0.0, 0.0, 0.5 * H]
        msb = rs.seat_back_mass_kg
        sb.inertia = [msb * (W * W + H * H) / 12, msb * (H * H + t * t) / 12, msb * (W * W + t * t) / 12]
        j = sb.add_joint(name="seat_back_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], pos=[0, 0, 0])
        _clean_joint(j, True, [-0.6, 1.45])
        j.damping = [50.0, 0, 0]
        j.armature = 0.5
        g = self._veh_geom(sb, "rear_seat_back", "rear_seat_back", GEOM.mjGEOM_BOX, [0.5 * t, 0.5 * W, 0.5 * H],
                           [0.0, 0.0, 0.5 * H], material="seat", friction=rs.cushion_friction)
        solref, solimp = self._cushion_contact()
        g.solref = [solref[0] * 0.8, solref[1]]
        g.solimp = list(solimp)
        if hr.present:
            soft = TRIM[hr.softness]
            for name, y in (("L", lay.seat_y["rear_left"]), ("C", 0.0), ("R", lay.seat_y["rear_right"])):
                gh = self._veh_geom(sb, "head_restraint", f"head_restraint_{name}", GEOM.mjGEOM_BOX,
                                    [0.5 * hr.thickness_m, 0.5 * hr.width_m, 0.5 * hr.height_above_seatback_m + 0.03],
                                    [0.5 * t + hr.forward_offset_m - 0.5 * hr.thickness_m, y,
                                     H + 0.5 * hr.height_above_seatback_m - 0.02], material="headrest")
                gh.solref = list(soft[0])
                gh.solimp = list(soft[1]) + [0.5, 2.0]
        # anchor sites on the seat back, given relative to the hinge in vehicle axes
        Rb = np.array([[math.cos(-beta), 0, math.sin(-beta)], [0, 1, 0], [-math.sin(-beta), 0, math.cos(-beta)]])
        for key in ("upper_on_seatback", "headrest_on_seatback"):
            p = self.anchors.get(key)
            if p is not None:
                sb.add_site(name=key, pos=(Rb.T @ p).tolist(), size=[0.012, 0, 0], material="anchor", group=1)
        self.seat_back_rot = Rb

    def _trunk(self):
        cfg = self.cfg
        rc = cfg.vehicle.rear_compartment
        lay = self.lay
        wb = self.spec.worldbody
        if rc.trunk_length_m <= 0.05:
            return
        x0, x1 = lay.trunk_rear_x, lay.trunk_front_x
        hw = lay.trunk_half_width
        zf = lay.trunk_floor_z
        zt = lay.trunk_top_z if rc.parcel_shelf or cfg.vehicle.body_style == "sedan" else rc.roof_height_m
        zt = min(zt, rc.roof_height_m)
        self._veh_geom(wb, "trunk", "trunk_floor", GEOM.mjGEOM_BOX, [0.5 * (x1 - x0) + 0.02, hw + 0.05, 0.04],
                       [0.5 * (x0 + x1), 0, zf - 0.04], material="floor", friction=0.6)
        for sgn, tag in ((1, "L"), (-1, "R")):
            self._veh_geom(wb, "trunk", f"trunk_side_{tag}", GEOM.mjGEOM_BOX, [0.5 * (x1 - x0) + 0.02, 0.03, 0.5 * (zt - zf)],
                           [0.5 * (x0 + x1), sgn * (hw + 0.03), 0.5 * (zt + zf)], material="trim", group=2)
        if rc.parcel_shelf:
            xs0 = x0
            xs1 = lay.seat_back_top[0] - 0.5 * rc.seat_back_thickness_m
            zs = lay.seat_back_top[1] - 0.01
            self._veh_geom(wb, "trunk", "parcel_shelf", GEOM.mjGEOM_BOX, [0.5 * (xs1 - xs0) + 0.02, hw + 0.05, 0.015],
                           [0.5 * (xs0 + xs1), 0, max(zs, zt) + 0.015], material="trim", group=2)
            if cfg.vehicle.body_style == "sedan":
                self._veh_geom(wb, "trunk", "trunk_lid", GEOM.mjGEOM_BOX, [0.5 * (x1 - x0) + 0.02, hw + 0.05, 0.02],
                               [0.5 * (x0 + x1), 0, zt + 0.02], material="trim", group=2)
        # rear wall / tail gate: prescribed body (moves forward in rear impacts)
        rw = self.spec.worldbody.add_body(name="trunk_rear_wall", pos=[x0 - 0.04, 0, 0.5 * (zt + zf)])
        jt = _clean_joint(rw.add_joint(name="trunk_rear_wall_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0]))
        jt.armature = 5000.0
        jt.damping = [0.0, 0, 0]
        self._veh_geom(rw, "trunk", "trunk_rear_wall", GEOM.mjGEOM_BOX, [0.04, hw + 0.05, 0.5 * (zt - zf) + 0.02],
                       [0, 0, 0], material="trim", trim="hard", group=2)
        rw.mass = 50.0
        rw.explicitinertial = True
        rw.inertia = [1.0, 1.0, 1.0]
        self.prescribed.append({"name": "trunk_rear_wall", "joint": "trunk_rear_wall_slide", "signal": "intrusion_rear",
                                "sign": 1.0})

    def _doors(self):
        cfg = self.cfg
        rc = cfg.vehicle.rear_compartment
        lay = self.lay
        cw = rc.cabin_width_m
        x0 = lay.c_pillar_x + 0.0
        x1 = lay.b_pillar_x - 0.07
        zb = rc.floor_height_m + 0.02
        zt = rc.beltline_height_m
        for sgn, tag, part, sig in (((1, "L", "door_left", "intrusion_left"), (-1, "R", "door_right", "intrusion_right"))
                                    if rc.doors_present else ()):
            body = self.spec.worldbody.add_body(name=f"rear_door_{tag}", pos=[0.5 * (x0 + x1), sgn * (0.5 * cw + 0.04),
                                                                         0.5 * (zb + zt)])
            j = _clean_joint(body.add_joint(name=f"rear_door_{tag}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0]))
            j.armature = 5000.0
            body.mass = 25.0
            body.explicitinertial = True
            body.inertia = [1.0, 1.0, 1.0]
            self._veh_geom(body, part, f"rear_door_trim_{tag}", GEOM.mjGEOM_BOX, [0.5 * (x1 - x0), 0.04, 0.5 * (zt - zb)],
                           [0, 0, 0], material="door", group=2)
            # arm rest
            self._veh_geom(body, part, f"rear_armrest_{tag}", GEOM.mjGEOM_BOX, [0.18, 0.04, 0.03],
                           [0.5 * (lay.bight[0] + lay.cushion_front[0]) - 0.5 * (x0 + x1), -sgn * 0.06,
                            rc.cushion_height_m + 0.22 - 0.5 * (zb + zt)], material="door", group=2)
            self.prescribed.append({"name": f"rear_door_{tag}", "joint": f"rear_door_{tag}_slide", "signal": sig,
                                    "sign": -sgn})

    def _airbags(self):
        cfg = self.cfg
        rc = cfg.vehicle.rear_compartment
        lay = self.lay
        cw = rc.cabin_width_m
        for i, ab in enumerate(cfg.restraints.airbags):
            if not ab.enabled:
                continue
            th = ab.thickness_m
            name = f"airbag_{i}_{ab.type}_{ab.side}"
            if ab.type == "curtain":
                sgn = 1.0 if ab.side == "left" else -1.0
                x0, x1 = lay.c_pillar_x + 0.02, lay.b_pillar_x + 0.10
                z0, z1 = rc.beltline_height_m - 0.14, rc.roof_height_m - 0.03
                size = [0.5 * (x1 - x0), 0.5 * th, 0.5 * (z1 - z0)]
                deployed = [0.5 * (x0 + x1), sgn * (0.5 * cw - 0.5 * th + 0.02), 0.5 * (z0 + z1)]
                axis = [0, 0, -1]
                stroke = z1 - z0
            elif ab.type == "rear_side_thorax":
                sgn = 1.0 if ab.side == "left" else -1.0
                xc = float(lay.dog_ref[0]) if cfg.rear_passenger.seating_position.startswith("rear_") else \
                    lay.bight[0] + 0.25
                size = [0.22, 0.5 * th, 0.20]
                deployed = [xc, sgn * (0.5 * cw - 0.5 * th), rc.cushion_height_m + 0.30]
                axis = [0, -sgn, 0]
                stroke = th + 0.02
            elif ab.type == "front_seatback":
                y = {"left": lay.seat_y["rear_left"], "right": lay.seat_y["rear_right"], "center": 0.0}[ab.side]
                depth = max(0.18, th * 2.0)
                size = [0.5 * depth, 0.24, 0.24]
                deployed = [rc.front_seat_back_x_m - 0.5 * depth - 0.01, y, rc.cushion_height_m + 0.42]
                axis = [-1, 0, 0]
                stroke = depth + 0.02
            else:  # rear_center
                size = [0.20, 0.5 * th, 0.26]
                deployed = [lay.bight[0] + 0.22, 0.0, rc.cushion_height_m + 0.30]
                axis = [1, 0, 0]
                stroke = 0.42
            stowed = (np.asarray(deployed) - np.asarray(axis, float) * stroke).tolist()
            body = self.spec.worldbody.add_body(name=name, pos=stowed)
            j = _clean_joint(body.add_joint(name=name + "_deploy", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=list(axis)))
            j.armature = 2000.0
            body.mass = 0.5
            body.explicitinertial = True
            body.inertia = [0.01, 0.01, 0.01]
            self._veh_geom(body, "airbag", name, GEOM.mjGEOM_BOX, size, [0, 0, 0], material="airbag", trim="soft",
                           friction=0.4)
            self.airbags.append({"name": name, "cfg": ab, "joint": name + "_deploy", "geom": name, "stroke": stroke})

    # ---- harness ------------------------------------------------------
    def _harness(self):
        cfg = self.cfg
        h = cfg.harness
        hg = self.hgeo
        sp = self.spec
        P = self.body_pos0
        tor = P["torso"]
        self.harness_joint_names = []
        meshes_by_body = {}
        for name, (body, verts, faces) in hg.meshes.items():
            if body == "harness" and h.style == "collar_only":
                continue
            meshes_by_body.setdefault(body, []).append((name, verts, faces))
        if h.style != "collar_only":
            hb = sp.body("torso").add_body(name="harness", pos=[0, 0, 0])
            hb.mass = h.mass_kg
            hb.explicitinertial = True
            r = self.sc.chest_half_width
            dz = 0.5 * self.sc.chest_depth
            hb.ipos = (hg.girth_center - tor).tolist()
            ix = h.mass_kg * (r * r + dz * dz) / 2
            hb.inertia = [ix, 0.6 * ix + 0.002, 0.6 * ix + 0.002]
            b = self.bush
            jspec = [("harness_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0], b["x_range"]),
                     ("harness_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0], (-b["yz_range"], b["yz_range"])),
                     ("harness_z", mujoco.mjtJoint.mjJNT_SLIDE, [0, 0, 1], (-b["yz_range"], b["yz_range"])),
                     ("harness_roll", mujoco.mjtJoint.mjJNT_HINGE, [1, 0, 0], (-b["roll_range"], b["roll_range"])),
                     ("harness_pitch", mujoco.mjtJoint.mjJNT_HINGE, [0, 1, 0], (-b["pitch_range"], b["pitch_range"])),
                     ("harness_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1], (-b["yaw_range"], b["yaw_range"]))]
            for jn, jt, ax, rng in jspec:
                jj = _clean_joint(hb.add_joint(name=jn, type=jt, axis=ax, pos=(hg.girth_center - tor).tolist()), True, rng)
                jj.solref_limit = [0.004, 1.0]
                jj.solimp_limit = [0.9, 0.99, 0.002, 0.5, 2.0]
                jj.armature = 0.002
                mass_eff = 2.0 + 0.1 * self.sc.body_mass
                k = b["k_yz"] if jt == mujoco.mjtJoint.mjJNT_SLIDE and jn != "harness_x" else (
                    max(b["kx_rear"], b["kx_fwd"]) if jn == "harness_x" else
                    {"harness_roll": b["k_roll"], "harness_pitch": b["k_pitch"], "harness_yaw": b["k_yaw"]}[jn])
                inertia_eff = mass_eff if jt == mujoco.mjtJoint.mjJNT_SLIDE else mass_eff * self.sc.chest_half_width ** 2
                jj.damping = [2.0 * 0.6 * math.sqrt(max(k, 1.0) * inertia_eff), 0, 0]
                self.harness_joint_names.append(jn)
            hb.add_site(name="harness_ring", pos=(hg.dring_world - tor).tolist(), size=[0.008, 0, 0], group=3)
            hb.add_site(name="harness_neck_L", pos=(hg.neck_dorsal_L - tor).tolist(), size=[0.005, 0, 0], group=3)
            hb.add_site(name="harness_neck_R", pos=(hg.neck_dorsal_R - tor).tolist(), size=[0.005, 0, 0], group=3)
            hb.add_site(name="harness_front_L", pos=(hg.front_side_L - tor).tolist(), size=[0.005, 0, 0], group=3)
            hb.add_site(name="harness_front_R", pos=(hg.front_side_R - tor).tolist(), size=[0.005, 0, 0], group=3)
            self.ring_site = "harness_ring"
            if h.visual_mesh:
                me = sp.add_mesh(name="harness_user_mesh", file=str(Path(h.visual_mesh).resolve()))
                me.scale = [self.sc.length_scale] * 3
                me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                gg = hb.add_geom(name="harness_user_mesh", type=GEOM.mjGEOM_MESH, meshname="harness_user_mesh",
                                 material="harness", group=0)
                gg.contype = gg.conaffinity = 0
                gg.mass = 0.0
                gg.pos = (-tor).tolist()
        else:
            self.ring_site = "collar_ring"
            c5 = P["C_5"]
            sp.body("C_5").add_site(name="collar_ring", pos=(hg.collar_ring - c5).tolist(), size=[0.008, 0, 0], group=3)
        # strap via points on the dog
        c7 = P["C_7"]
        sp.body("C_7").add_site(name="neck_lat_L", pos=(hg.neck_lat_L - c7).tolist(), size=[0.005, 0, 0], group=3)
        sp.body("C_7").add_site(name="neck_lat_R", pos=(hg.neck_lat_R - c7).tolist(), size=[0.005, 0, 0], group=3)
        if self.bush["neck_on_throat"]:
            sp.body("C_7").add_site(name="neck_ventral", pos=(hg.neck_ventral - c7).tolist(), size=[0.005, 0, 0], group=3)
        else:
            # the neck opening of a vest bears on the prosternum (thorax), below the throat
            low = hg.prosternum + np.array([0.0, 0.0, 0.01 * self.sc.length_scale])
            sp.body("torso").add_site(name="neck_ventral", pos=(low - tor).tolist(), size=[0.005, 0, 0], group=3)
        sp.body("upper_arm_L").add_site(name="shoulder_front_L", pos=(hg.shoulder_front_L - P["upper_arm_L"]).tolist(),
                                        size=[0.005, 0, 0], group=3)
        sp.body("upper_arm_R").add_site(name="shoulder_front_R", pos=(hg.shoulder_front_R - P["upper_arm_R"]).tolist(),
                                        size=[0.005, 0, 0], group=3)
        sp.body("torso").add_site(name="prosternum", pos=(hg.prosternum - tor).tolist(), size=[0.005, 0, 0], group=3)
        # visual meshes
        for body, items in meshes_by_body.items():
            parent = sp.body(body)
            origin = P["torso"] if body == "harness" else P[body]
            for name, verts, faces in items:
                me = sp.add_mesh(name="mesh_" + name)
                me.uservert = (verts - origin).flatten().tolist()
                me.userface = faces.flatten().tolist()
                me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                gg = parent.add_geom(name="harness_" + name, type=GEOM.mjGEOM_MESH, meshname="mesh_" + name,
                                     material="hardware" if name == "ring" else "harness", group=0)
                gg.contype = gg.conaffinity = 0
                gg.mass = 0.0
                gg.density = 0.0

    def _add_tendon(self, name, sites, kind, rgba, vis_width, group, **info):
        t = self.spec.add_tendon(name=name)
        t.rgba = list(rgba)
        t.width = vis_width
        t.group = group
        for s_ in sites:
            t.wrap_site(s_)
        a = self.spec.add_actuator(name=name + "_force", trntype=mujoco.mjtTrn.mjTRN_TENDON, target=name)
        a.gainprm = [1.0] + [0.0] * 9
        a.biasprm = [0.0] * 10
        a.ctrllimited = False
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_NONE
        self.tendons.append(TendonInfo(name=name, kind=kind, **info))

    def _restraint_tendons(self):
        cfg = self.cfg
        rp = cfg.rear_passenger
        h = cfg.harness
        wb = self.spec.worldbody
        A = self.anchors
        s = self.sc.length_scale
        strap_w = h.strap_width_mm * 1e-3
        # body straps (always present when the harness is worn)
        b = self.bush
        if h.style != "collar_only":
            if b["neck_slack"] is not None:
                on_throat = b["neck_on_throat"]
                self._add_tendon("neck_loop", ["harness_neck_L", "neck_lat_L", "neck_ventral", "neck_lat_R", "harness_neck_R"],
                                 "strap", h.rgba, 0.006 * s, 3, width=strap_w,
                                 radius=self.sc.neck_radius if on_throat else self.sc.chest_half_width,
                                 strength=h.strap_rated_strength_kN * 1e3, connector_strength=h.buckle_strength_kN * 1e3,
                                 meta={"slack": b["neck_slack"], "body_site": "neck_ventral", "on_throat": on_throat})
            if b["front_strap"]:
                self._add_tendon("front_strap", ["harness_front_L", "shoulder_front_L", "prosternum", "shoulder_front_R",
                                                 "harness_front_R"], "strap", h.rgba, 0.006 * s, 3, width=strap_w,
                                 radius=self.sc.chest_half_width, strength=h.strap_rated_strength_kN * 1e3,
                                 connector_strength=h.buckle_strength_kN * 1e3,
                                 meta={"slack": 0.004 + h.fit_slack_mm * 1e-3 * 0.5, "body_site": "prosternum"})
        if not rp.seat_belt_used:
            self.notes.append("Dog unrestrained: the harness is not connected to the vehicle.")
            return
        att = cfg.restraints.attachment
        ring = self.ring_site
        tw = 0.025
        teth = h.tether

        def wsite(name, p):
            wb.add_site(name=name, pos=list(map(float, p)), size=[0.014, 0, 0], material="anchor", group=1)
            return name

        tinfo = dict(width=tw, radius=0.05, strength=teth.rated_strength_kN * 1e3,
                     connector_strength=teth.connector_strength_kN * 1e3)
        if self.lay.support == "cargo_floor" and att in ("belt_loop", "buckle_tether", "isofix_tether"):
            self.notes.append(f"Attachment {att!r} is not reachable from the cargo area; using cargo_hook_tether.")
            att = "cargo_hook_tether"
        self.attachment = att
        if att == "belt_loop":
            upper = "upper_on_seatback" if A["upper"] is None else wsite("belt_upper_anchor", A["upper"])
            buckle = wsite("belt_buckle", A["buckle"])
            wsite("belt_lap_anchor", A["lap"])
            web_strength = 26e3
            self._add_tendon("vehicle_belt", [upper, ring, buckle], "belt", [0.07, 0.07, 0.07, 1], 0.012, 1,
                             width=0.046, radius=0.05, strength=web_strength,
                             connector_strength=min(h.d_ring_strength_kN * 1e3, web_strength),
                             meta={"retractor": True})
            # the lap part is buckled but does not cross the dog
            self._add_tendon("vehicle_belt_lap", ["belt_buckle", "belt_lap_anchor"], "belt_lap", [0.07, 0.07, 0.07, 1],
                             0.012, 1, width=0.046)
        elif att == "buckle_tether":
            buckle = wsite("belt_buckle", A["buckle"])
            self._add_tendon("tether", [ring, buckle], "tether", h.rgba, 0.010, 1,
                             meta={"buckle_pretensioner": cfg.restraints.pretensioner.type == "buckle"}, **tinfo)
        elif att == "isofix_tether":
            l_ = wsite("isofix_L", A["isofix_l"])
            r_ = wsite("isofix_R", A["isofix_r"])
            self._add_tendon("tether", [l_, ring, r_], "tether", h.rgba, 0.010, 1, **tinfo)
        elif att == "headrest_tether":
            if not cfg.restraints.head_restraints.present:
                raise ValueError("headrest_tether needs rear head restraints (restraints.head_restraints.present)")
            self._add_tendon("tether", ["headrest_on_seatback", ring], "tether", h.rgba, 0.010, 1, **tinfo)
        elif att == "cargo_hook_tether":
            hooks = A["cargo_hooks"]
            d = self.lay.dog_ref
            # the two hooks closest to the dog
            order = np.argsort([np.linalg.norm(hk[:2] - d[:2]) for hk in hooks])
            a_ = wsite("cargo_hook_A", hooks[order[0]])
            b_ = wsite("cargo_hook_B", hooks[order[1]])
            self._add_tendon("tether", [a_, ring, b_], "tether", h.rgba, 0.010, 1, **tinfo)

    # ---- cargo --------------------------------------------------------
    def _cargo_positions(self):
        """Centre positions of the cargo items for the chosen packing arrangement."""
        cfg = self.cfg
        tc = cfg.trunk_cargo
        lay = self.lay
        t_sb = cfg.vehicle.rear_compartment.seat_back_thickness_m
        zf = lay.trunk_floor_z
        z_top = lay.trunk_top_z if cfg.vehicle.rear_compartment.parcel_shelf else cfg.vehicle.rear_compartment.roof_height_m
        hw = lay.trunk_half_width
        x_rear = lay.trunk_rear_x + 0.01
        x_front0 = lay.trunk_front_x - 0.005
        if lay.support == "cargo_floor":   # the dog occupies the front of the cargo area
            x_front0 = lay.dog_ref[0] - 0.45 * self.sc.length_scale

        def front_limit(top):
            if lay.support == "cargo_floor":
                return x_front0
            return min(x_front0, lay.seatback_rear_x(top, t_sb) - 0.015)

        pos = []
        placed = []   # (x0, x1, y0, y1, z_top)
        row_front, row_depth, y_cur = None, 0.0, hw
        stack_z = zf
        rear_cursor = x_rear
        for k, it in enumerate(tc.items):
            l, w, h = it.size_m
            if it.position_m is not None:
                pos.append(np.asarray(it.position_m, float))
                continue
            if w > 2 * hw + 1e-6 or h > z_top - zf + 1e-6:
                raise ValueError(f"cargo item {it.name!r} is larger than the trunk opening")
            if tc.packing == "stacked":
                if stack_z + h > z_top + 1e-6:
                    raise ValueError(f"cargo item {it.name!r} does not fit on the stack under the trunk lid")
                xf = front_limit(stack_z + h)
                p = np.array([xf - 0.5 * l, 0.0, stack_z + 0.5 * h + 0.003])
                stack_z += h + 0.005
            elif tc.packing == "loose_rear":
                if y_cur - w < -hw:
                    rear_cursor += row_depth + 0.01
                    y_cur, row_depth = hw, 0.0
                p = np.array([rear_cursor + 0.5 * l, y_cur - 0.5 * w, zf + 0.5 * h + 0.003])
                y_cur -= w + 0.01
                row_depth = max(row_depth, l)
            else:  # against_seatback / side_by_side / custom: rows across the width, starting at the seat back
                if row_front is None:
                    row_front = front_limit(zf + h)
                if y_cur - w < -hw:
                    if tc.packing == "side_by_side":
                        raise ValueError(f"cargo item {it.name!r} does not fit across the trunk")
                    row_front -= row_depth + 0.01
                    y_cur, row_depth = hw, 0.0
                xf = min(row_front, front_limit(zf + h))
                p = np.array([xf - 0.5 * l, y_cur - 0.5 * w, zf + 0.5 * h + 0.003])
                y_cur -= w + 0.01
                row_depth = max(row_depth, row_front - (xf - l))
            if p[0] - 0.5 * l < x_rear - 1e-6 or p[0] + 0.5 * l > lay.trunk_front_x + 0.3:
                # last resort: put it on top of an earlier item
                for (x0, x1, y0, y1, zt) in placed:
                    if x1 - x0 >= 0.5 * l and zt + h <= z_top:
                        p = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), zt + 0.5 * h + 0.003])
                        break
                else:
                    raise ValueError(f"cargo item {it.name!r} does not fit in the trunk/cargo area")
            placed.append((p[0] - 0.5 * l, p[0] + 0.5 * l, p[1] - 0.5 * w, p[1] + 0.5 * w, p[2] + 0.5 * h))
            pos.append(p)
        return pos

    def _cargo(self):
        cfg = self.cfg
        tc = cfg.trunk_cargo
        if not tc.items or cfg.vehicle.rear_compartment.trunk_length_m <= 0.05:
            if tc.items:
                self.notes.append("The vehicle has no trunk/cargo area; cargo items ignored.")
            return
        sp = self.spec
        positions = self._cargo_positions()
        for k, (it, p) in enumerate(zip(tc.items, positions)):
            l, w, hgt = it.size_m
            body = sp.worldbody.add_body(name=f"cargo_{k}", pos=p.tolist(), quat=quat_z(math.radians(it.yaw_deg)))
            body.add_freejoint(name=f"cargo_{k}_free")
            g = body.add_geom(name=f"cargo_{k}", type=GEOM.mjGEOM_BOX, size=[0.5 * l, 0.5 * w, 0.5 * hgt],
                              material="cargo", group=1)
            g.mass = it.mass_kg
            g.contype = CARGO_CONTYPE
            g.conaffinity = CARGO_CONAFF
            g.condim = 3
            g.priority = 4
            g.friction = [it.friction, 0.005, 0.0001]
            if it.construction == "soft":
                g.solref = [0.015, 0.8]
                g.solimp = [0.5, 0.95, 0.05, 0.5, 2.0]
            else:
                g.solref = [0.004, 1.0]
                g.solimp = [0.9, 0.98, 0.005, 0.5, 2.0]
            self.veh_part_of_geom[f"cargo_{k}"] = "cargo"
            self.cargo.append({"name": it.name, "body": f"cargo_{k}", "geom": f"cargo_{k}", "mass": it.mass_kg,
                               "size": [l, w, hgt]})
            if tc.restraint_method == "tie_down_straps":
                n = max(1, tc.straps_per_item)
                corners = [(0.5 * l, 0.5 * w), (0.5 * l, -0.5 * w), (-0.5 * l, 0.5 * w), (-0.5 * l, -0.5 * w)]
                if n == 1:
                    corners = [(0.0, 0.0)]
                elif n == 2:
                    corners = [(0.5 * l, 0.0), (-0.5 * l, 0.0)]
                for c_i, (cx, cy) in enumerate(corners[:n]):
                    sname = f"cargo_{k}_strap_{c_i}"
                    body.add_site(name=sname + "_top", pos=[cx, cy, 0.5 * hgt], size=[0.006, 0, 0], group=3)
                    out = np.array([cx, cy]) / max(np.linalg.norm([cx, cy]), 1e-6) if (cx or cy) else np.array([1.0, 0])
                    floor_pt = np.array([p[0] + cx + 0.12 * out[0], p[1] + cy + 0.12 * out[1], self.lay.trunk_floor_z])
                    sp.worldbody.add_site(name=sname + "_hook", pos=floor_pt.tolist(), size=[0.01, 0, 0],
                                          material="anchor", group=1)
                    self._add_tendon(sname, [sname + "_top", sname + "_hook"], "cargo_strap", [0.9, 0.6, 0.1, 1], 0.008, 1,
                                     width=0.025, strength=tc.strap_strength_kN * 1e3,
                                     meta={"stiffness": tc.strap_strength_kN * 1e3 / (tc.strap_elongation_pct_at_rated / 100.0)})
            elif tc.restraint_method == "locked_floor_box":
                eq = sp.add_equality(name=f"cargo_{k}_lock", type=mujoco.mjtEq.mjEQ_WELD, name1=f"cargo_{k}",
                                     objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.solref = [0.004, 1.0]
                self.cargo_restraint.setdefault("welds", []).append(f"cargo_{k}_lock")
        if tc.restraint_method in ("cargo_net", "cargo_barrier"):
            lay = self.lay
            rc = cfg.vehicle.rear_compartment
            net = tc.restraint_method == "cargo_net"
            z0 = lay.trunk_floor_z if net else lay.seat_back_top[1] - 0.05
            z1 = rc.roof_height_m - 0.02 if not rc.parcel_shelf else lay.trunk_top_z
            if z1 - z0 < 0.1:
                self.notes.append("No room for a cargo net/barrier above the seat back; ignored.")
                return
            x = lay.trunk_front_x - 0.03
            body = sp.worldbody.add_body(name="cargo_partition", pos=[x, 0, 0.5 * (z0 + z1)])
            j = _clean_joint(body.add_joint(name="cargo_partition_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0]),
                             True, [-0.05, 0.6])
            j.damping = [200.0, 0, 0]
            body.mass = 2.0 if net else 8.0
            body.explicitinertial = True
            body.inertia = [0.2, 0.2, 0.2]
            g = self._veh_geom(body, "cargo_net_barrier", "cargo_partition", GEOM.mjGEOM_BOX,
                               [0.01, lay.trunk_half_width, 0.5 * (z1 - z0)], [0, 0, 0], material="net",
                               trim="soft" if net else "medium")
            self.cargo_restraint.update({"partition": "cargo_partition", "joint": "cargo_partition_slide",
                                         "stiffness": (tc.net_stiffness_kN_per_m if net else tc.barrier_stiffness_kN_per_m) * 1e3,
                                         "strength": (tc.net_strength_kN if net else tc.barrier_strength_kN) * 1e3})

    # ---- exterior / world / cameras -------------------------------------
    def _exterior_and_world(self):
        cfg = self.cfg
        v = cfg.vehicle
        lay = self.lay
        sp = self.spec
        wb = sp.worldbody
        wb.add_light(name="sun", pos=[0, 0, 6], dir=[0.2, 0.1, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     diffuse=[0.7, 0.7, 0.7], specular=[0.2, 0.2, 0.2], castshadow=True)
        wb.add_light(name="cabin", pos=[lay.dog_ref[0], lay.dog_ref[1], cfg.vehicle.rear_compartment.roof_height_m - 0.05],
                     dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5], castshadow=False)
        hw = lay.half_width
        x0, x1 = lay.x_rear_bumper, lay.x_front_bumper
        bl = v.rear_compartment.beltline_height_m

        def vis(body, name, gtype, size, pos, mat, quat=None):
            g = body.add_geom(name=name, type=gtype, size=list(size), pos=list(pos), material=mat, group=2)
            if quat is not None:
                g.quat = list(quat)
            g.contype = g.conaffinity = 0
            g.mass = 0.0
            return g

        # body shell (centre section; the crushable ends are mocap bodies)
        xa, xb = x0 + 0.55, x1 - 0.65
        vis(wb, "shell_lower_L", GEOM.mjGEOM_BOX, [0.5 * (xb - xa), 0.02, 0.5 * (bl - 0.25)], [0.5 * (xa + xb), hw, 0.5 * (bl + 0.25)], "paint")
        vis(wb, "shell_lower_R", GEOM.mjGEOM_BOX, [0.5 * (xb - xa), 0.02, 0.5 * (bl - 0.25)], [0.5 * (xa + xb), -hw, 0.5 * (bl + 0.25)], "paint")
        for sgn, ax in ((1, lay.x_front_axle), (1, lay.x_rear_axle), (-1, lay.x_front_axle), (-1, lay.x_rear_axle)):
            vis(wb, f"wheel_{sgn}_{ax:.2f}", GEOM.mjGEOM_CYLINDER, [0.32, 0.11], [ax, sgn * 0.5 * v.track_m, 0.32], "tyre",
                quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])
        self.mocap = {}
        for end, xa_, xb_ in (("front_end", xb, x1), ("rear_end", x0, xa)):
            b = wb.add_body(name=end, mocap=True, pos=[0.5 * (xa_ + xb_), 0, 0])
            vis(b, end + "_box", GEOM.mjGEOM_BOX, [0.5 * (xb_ - xa_), hw, 0.5 * (bl - 0.25) - 0.08], [0, 0, 0.5 * (bl + 0.25) - 0.08], "paint")
            self.mocap[end] = end
        # road
        road = wb.add_body(name="road", mocap=True, pos=[0, 0, 0])
        g = road.add_geom(name="road", type=GEOM.mjGEOM_BOX, size=[40, 40, 0.02], pos=[0, 0, -0.02], material="road",
                          group=0)
        g.contype = g.conaffinity = 0
        g.mass = 0.0
        self.mocap["road"] = "road"
        # partner
        col = cfg.collision
        pb = wb.add_body(name="partner", mocap=True, pos=[1e3, 0, 0])
        self.mocap["partner"] = "partner"
        if col.mode == "crash":
            pd = partner_defaults(col.partner)
            Lp, Wp = pd["length_m"], pd["width_m"]
            tp = col.partner.type
            if tp == "pole":
                vis(pb, "partner_pole", GEOM.mjGEOM_CYLINDER, [0.5 * Wp, 2.0], [0, 0, 2.0], "concrete")
            elif tp == "rigid_barrier":
                vis(pb, "partner_block", GEOM.mjGEOM_BOX, [0.5 * Lp, 0.5 * Wp, 0.75], [0, 0, 0.75], "concrete")
            elif tp == "deformable_barrier":
                vis(pb, "partner_block", GEOM.mjGEOM_BOX, [1.5, 1.5, 1.0], [-1.5 - 0.5 * Lp + 0.54, 0, 1.0], "concrete")
                vis(pb, "partner_face", GEOM.mjGEOM_BOX, [0.27, 0.5 * Wp, 0.33], [0.5 * Lp - 0.27, 0, 0.53], "alu")
            else:
                hp = {"vehicle": 1.45, "moving_barrier": 1.2, "truck": 3.2}.get(tp, 1.4)
                zb_ = {"truck": 0.55}.get(tp, 0.22)
                vis(pb, "partner_body", GEOM.mjGEOM_BOX, [0.5 * Lp, 0.5 * Wp, 0.5 * (hp - zb_)], [0, 0, 0.5 * (hp + zb_)],
                    "partner")
                if tp == "moving_barrier":
                    vis(pb, "partner_face", GEOM.mjGEOM_BOX, [0.25, 0.5 * Wp, 0.25], [0.5 * Lp - 0.25, 0, 0.55], "alu")

    def _cameras(self):
        cfg = self.cfg
        lay = self.lay
        rc = cfg.vehicle.rear_compartment
        wb = self.spec.worldbody
        d = lay.dog_ref
        sgn = lay.outboard_sign
        tgt = np.array([d[0] + 0.05, d[1], d[2] + 0.35 * self.sc.length_scale + 0.1])
        cams = []

        def cam(name, pos, target, fovy=45.0):
            wb.add_camera(name=name, pos=list(map(float, pos)), xyaxes=lookat_xyaxes(pos, target), fovy=fovy)
            cams.append(name)

        cam("side", tgt + np.array([0.10, sgn * 1.9, 0.15]), tgt, 50)
        cam("dog", np.array([lay.cushion_front[0] + 0.05, d[1] - sgn * 0.75, rc.roof_height_m - 0.10]), tgt, 75)
        cam("top", np.array([d[0] - 0.15, d[1] - 0.01, rc.roof_height_m + 1.6]), tgt, 50)
        cam("front", np.array([rc.front_seat_back_x_m + 0.05, 0.0, rc.roof_height_m - 0.08]), tgt, 75)
        cam("world", np.array([-4.5, sgn * 7.5, 4.0]), np.array([0.5, 0, 0.5]), 45)
        self.cameras = cams

    # ---- finish ---------------------------------------------------------
    def _finish(self, m: mujoco.MjModel) -> Scene:
        cfg = self.cfg
        dog_bodies = np.array([m.body(n).id for n in self.dog_body_names])
        dog_joints = np.array([m.joint(n).id for n in self.dog_joint_names])
        dog_acts = np.array([m.actuator(n).id for n in self.dog_act_names])
        root = m.joint("root")
        qa = root.qposadr[0]
        va = root.dofadr[0]
        # dog tree extent in qpos/qvel (all dog joints are contiguous after the free joint)
        dog_q_end = qa + 7 + len(self.dog_joint_names)
        dog_v_end = va + 6 + len(self.dog_joint_names)
        hb = m.body("harness").id if cfg.harness.style != "collar_only" else -1
        hj = [m.joint(n).id for n in getattr(self, "harness_joint_names", [])]
        # teeth are enclosed by the skull/jaw primitives; skip their contacts (performance)
        for g in range(m.ngeom):
            nm = m.geom(g).name
            if any(k in nm for k in ("Molars", "Premolars", "Canine", "Incisors")):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        for t in self.tendons:
            t.tendon = m.tendon(t.name).id
            t.actuator = m.actuator(t.name + "_force").id
        geom_dog = np.full(m.ngeom, -1, int)
        geom_veh = np.full(m.ngeom, -1, int)
        dog_body_set = set(dog_bodies.tolist())
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            nm = m.geom(g).name
            if b in dog_body_set and m.geom_contype[g]:
                geom_dog[g] = DOG_PARTS.index(dog_part_of(m.body(b).name))
            elif nm in self.veh_part_of_geom:
                geom_veh[g] = VEH_PARTS.index(self.veh_part_of_geom[nm])
        prescribed = []
        for p in self.prescribed:
            j = m.joint(p["joint"]).id
            prescribed.append(dict(p, jid=j, qadr=int(m.jnt_qposadr[j]), dof=int(m.jnt_dofadr[j])))
        airbags = []
        for ab in self.airbags:
            j = m.joint(ab["joint"]).id
            airbags.append(dict(ab, jid=j, qadr=int(m.jnt_qposadr[j]), dof=int(m.jnt_dofadr[j]), gid=m.geom(ab["geom"]).id))
            m.geom_contype[airbags[-1]["gid"]] = 0
            m.geom_conaffinity[airbags[-1]["gid"]] = 0
            airbags[-1]["rgba"] = m.geom_rgba[airbags[-1]["gid"]].copy()
            m.geom_rgba[airbags[-1]["gid"], 3] = 0.0
        cargo = []
        for c in self.cargo:
            b = m.body(c["body"]).id
            cargo.append(dict(c, bid=b, gid=m.geom(c["geom"]).id, qadr=int(m.jnt_qposadr[m.body_jntadr[b]])))
        cr = dict(self.cargo_restraint)
        if "joint" in cr:
            j = m.joint(cr["joint"]).id
            cr.update(jid=j, qadr=int(m.jnt_qposadr[j]), dof=int(m.jnt_dofadr[j]), gid=m.geom("cargo_partition").id)
        if "welds" in cr:
            cr["weld_ids"] = [m.equality(n).id for n in cr["welds"]]
        prescribed_bodies = {m.body(p["name"]).id for p in self.prescribed} | {m.body(a["name"]).id for a in self.airbags}
        dyn = [b for b in range(1, m.nbody) if m.body_mocapid[b] < 0 and m.body_jntnum[b] > 0 or
               (m.body_mocapid[b] < 0 and b in dog_body_set)]
        dyn = np.array(sorted({b for b in dyn if b not in prescribed_bodies and m.body_mass[b] > 0}))
        sensors = {}
        for nm in self.sensor_names:
            sid = m.sensor(nm).id
            sensors[nm] = (int(m.sensor_adr[sid]), int(m.sensor_dim[sid]))
        mocap = {k: int(m.body_mocapid[m.body(v).id]) for k, v in self.mocap.items()}
        init = self._dog_initial_qpos(m, qa, dog_q_end)
        return Scene(cfg=cfg, model=m, layout=self.lay, scaling=self.sc, harness=self.hgeo, anchors=self.anchors,
                     dog_bodies=dog_bodies, dog_joints=dog_joints, dog_actuators=dog_acts, dog_init_qpos=init,
                     dog_qpos_slice=slice(qa, dog_q_end), dog_qvel_slice=slice(va, dog_v_end), harness_body=hb,
                     harness_joints=hj, bushing=self.bush, tendons=self.tendons,
                     seatback_joint=m.joint("seat_back_hinge").id, prescribed=prescribed, airbags=airbags,
                     cargo=cargo, cargo_restraint=cr, mocap=mocap, geom_dog_part=geom_dog, geom_veh_part=geom_veh,
                     dynamic_bodies=dyn, sensors=sensors, cameras=self.cameras, ring_site=self.ring_site,
                     attachment=getattr(self, "attachment", "none"), notes=self.notes)

    # ---- initial placement of the dog ----------------------------------
    def _dog_initial_qpos(self, m, qa, q_end):
        """Place the dog in its posture on its support without interpenetrating the interior.

        If the requested heading does not fit (e.g. a large dog standing forwards has its head in the
        front head restraint), the dog turns its head towards the gap between the front seats and lowers
        it, like a real dog would; if nothing fits, an error explains why.
        """
        rp = self.cfg.rear_passenger
        facing = rp.facing_deg if rp.facing_deg is not None else default_facing(rp.posture, rp.seating_position)
        post = POSTURES[rp.posture]
        lowered = {"C_7_extend": 18.0, "C_5_extend": 10.0, "C_3_extend": 12.0}
        y = self.lay.dog_ref[1]
        toward_center = [-1.0] if y > 0.05 else ([1.0] if y < -0.05 else [-1.0, 1.0])
        # (heading, extra joint angles, trunk pitch, trunk roll) in order of preference
        candidates = [(facing, {}, post.pitch_deg, post.roll_deg)]
        free_heading = rp.facing_deg is None
        if rp.posture == "sitting":
            for pitch in (post.pitch_deg, -35.0, -25.0):
                candidates.append((facing, lowered, pitch, post.roll_deg))
        if free_heading and rp.posture in ("sitting", "standing") and self.lay.support == "cushion":
            for extra in ({}, lowered):
                for ang in (15.0, 30.0, 45.0):
                    for sgn in toward_center:
                        candidates.append((facing + sgn * ang, extra, post.pitch_deg if not extra else
                                           max(post.pitch_deg, -35.0), post.roll_deg))
        if rp.posture.startswith("lying"):
            rolls = (post.roll_deg, -post.roll_deg) if post.roll_deg else (post.roll_deg,)
            for roll in rolls:
                for fac in ((facing, facing + 180.0) if free_heading else (facing,)):
                    for extra in ({}, lowered):
                        candidates.append((fac, extra, post.pitch_deg, roll))
            if free_heading:
                for fac in (facing + 20.0, facing - 20.0, 0.0, 180.0):
                    candidates.append((fac, lowered, post.pitch_deg, post.roll_deg))
        best = None
        for fac, extra, pitch, roll in candidates:
            q, resid = self._place_dog(m, qa, q_end, fac, extra, pitch, roll)
            if best is None or resid > best[1]:
                best = (q, resid, fac, extra)
            if resid > -0.003:
                break
        q, resid, fac, extra = best
        if resid < -0.02:
            raise ValueError(f"the {rp.posture} dog does not fit at {rp.seating_position} (residual interpenetration "
                             f"{-resid * 1000:.0f} mm); choose another posture, facing_deg or seating position")
        if fac != facing or extra:
            self.notes.append(f"Dog placement adjusted to fit the compartment: heading {fac:.0f} deg"
                              + (", head lowered / trunk lowered" if extra else "") + ".")
        return q

    def _place_dog(self, m, qa, q_end, facing, extra_joints, pitch_deg, roll_deg):
        cfg = self.cfg
        rp = cfg.rear_passenger
        lay = self.lay
        post = POSTURES[rp.posture]
        d = mujoco.MjData(m)
        q = d.qpos
        q[:] = m.qpos0
        for jn, deg in post.joints.items():
            q[m.joint(jn).qposadr[0]] = math.radians(deg)
        for jn, deg in extra_joints.items():
            adr = m.joint(jn).qposadr[0]
            q[adr] = min(q[adr] + math.radians(deg), m.jnt_range[m.joint(jn).id, 1] - 0.02)
        # curl the tail along the body
        for j in range(m.njnt):
            nm = m.joint(j).name
            if nm.startswith("Ca_") and nm.endswith("_bend"):
                q[m.jnt_qposadr[j]] = math.radians(14.0 if rp.posture == "sitting" else 10.0)
            if nm.startswith("Ca_") and nm.endswith("_extend"):
                q[m.jnt_qposadr[j]] = math.radians(8.0 if rp.posture == "sitting" else 4.0)
        quat = quat_from_euler(math.radians(roll_deg), math.radians(pitch_deg), math.radians(facing))
        q[qa:qa + 3] = 0.0
        q[qa + 3:qa + 7] = quat
        mujoco.mj_forward(m, d)
        dog_geoms = set(g for g in range(m.ngeom) if self._is_dog_collision(m, g))
        body_geoms = [g for g in dog_geoms if not m.body(m.geom_bodyid[g]).name.startswith("Ca_")]
        lo_b, hi_b = self._aabb(m, d, body_geoms)
        ref = lay.dog_ref.copy()
        support_z = lay.cushion_z(ref[0]) if lay.support == "cushion" else ref[2]
        dz = support_z + 0.004 - lo_b[2]
        if rp.posture in ("sitting", "standing") and lay.support == "cushion" and abs(math.sin(math.radians(facing))) < 0.75:
            # rump against the seat back face
            z_rump = support_z + 0.10 * self.sc.length_scale
            x_face = lay.bight[0] - (z_rump - lay.bight[1]) * math.tan(lay.seat_back_angle)
            dx = x_face + 0.035 - lo_b[0]
        else:
            dx = ref[0] - 0.5 * (lo_b[0] + hi_b[0])
        dy = ref[1] - 0.5 * (lo_b[1] + hi_b[1])
        q[qa:qa + 3] = [dx, dy, dz]
        body_set = set(body_geoms)

        def worst_contact(include_tail):
            mujoco.mj_forward(m, d)
            worst = None
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = c.geom1, c.geom2
                if (g1 in dog_geoms) == (g2 in dog_geoms):
                    continue
                if not include_tail and not (g1 in body_set or g2 in body_set):
                    continue
                if c.dist < -0.002 and (worst is None or c.dist < worst[0]):
                    n = np.array(c.frame[:3])
                    worst = (c.dist, n * (-c.dist + 0.002) * (1.0 if g2 in dog_geoms else -1.0))
            return worst

        for _ in range(30):
            w = worst_contact(True)
            if w is None:
                break
            q[qa:qa + 3] += w[1]
        w = worst_contact(False)
        resid = 0.0 if w is None else float(w[0])
        self.dog_lowest = lo_b[2] + dz
        return q[qa:q_end].copy(), resid

    @staticmethod
    def _is_dog_collision(m, g):
        return m.geom_contype[g] == 1 and m.body_rootid[m.geom_bodyid[g]] == m.body("torso").id

    @staticmethod
    def _aabb(m, d, geoms):
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for g in geoms:
            R = d.geom_xmat[g].reshape(3, 3)
            sz = m.geom_size[g]
            t = m.geom_type[g]
            if t == GEOM.mjGEOM_SPHERE:
                ext = np.full(3, sz[0])
            elif t == GEOM.mjGEOM_CAPSULE:
                ext = np.abs(R[:, 2]) * sz[1] + sz[0]
            elif t == GEOM.mjGEOM_BOX:
                ext = np.abs(R) @ sz[:3]
            elif t == GEOM.mjGEOM_ELLIPSOID:
                ext = np.sqrt((R * sz[:3]) ** 2 @ np.ones(3))
            else:
                ext = np.full(3, m.geom_rbound[g])
            p = d.geom_xpos[g]
            lo = np.minimum(lo, p - ext)
            hi = np.maximum(hi, p + ext)
        return lo, hi


def build_scene(cfg: ScenarioConfig) -> Scene:
    return SceneBuilder(cfg).build()
