"""Loading and allometric scaling of the musculoskeletal dog model.

The MuJoCo dog in ``musculoskeletal_dog/models/dog.xml`` is a lean, deep
chested dog standing about 0.57 m at the withers with a total mass of
~9.7 kg (its soft tissue is only represented by light collision
primitives).  For crash work the body mass matters more than anything
else, so the model is rescaled to the requested body size:

* every length (body/geom/site/joint positions, primitive sizes, mesh and
  skin vertices) is multiplied by the length scale ``s = h / h0`` from
  the withers height;
* every density/mass is multiplied by ``k = M / (M0 s^3)`` so the total
  mass equals the requested body mass (mass distribution preserved);
* inertias follow (``k s^5``), joint stiffness and muscle torque capacity
  scale with ``M s`` (torque needed to support the body weight), joint
  damping with ``M s^1.5`` (gravitational time scale ~ sqrt(s)) and joint
  armature with ``M s^2``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "musculoskeletal_dog" / "models" / "dog.xml"

BASE_WITHERS_HEIGHT = 0.57      # m, measured on the skin of the unscaled model
BASE_CHEST_DEPTH = 0.26         # m, sternum to withers at the girth
BASE_CHEST_HALF_WIDTH = 0.082   # m
BASE_NECK_RADIUS = 0.055        # m, base of the neck

# Segment mass fractions used to redistribute the model's mass (canine body-segment data,
# rounded; the raw model carries 16 % of its mass in the neck and only 26 % in the thorax).
ANATOMICAL_FRACTIONS = {
    "head": 0.07, "neck": 0.06, "thorax": 0.32, "lumbar": 0.14, "pelvis": 0.08,
    "forelimbs": 0.12, "hindlimbs": 0.20, "tail": 0.01,
}


def segment_of(body_name: str) -> str:
    if body_name in ("skull", "jaw"):
        return "head"
    if body_name.startswith("C_"):
        return "neck"
    if body_name in ("torso", "scapula_L", "scapula_R"):
        return "thorax"
    if body_name.startswith("L_"):
        return "lumbar"
    if body_name == "pelvis":
        return "pelvis"
    if body_name.startswith("Ca_"):
        return "tail"
    if any(body_name.startswith(p) for p in ("upper_arm", "lower_arm", "hand", "finger")):
        return "forelimbs"
    return "hindlimbs"


PRIMITIVES_WITH_LENGTH_SIZE = {
    mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX,
}


def read_skn(path: Path):
    """Parse a MuJoCo binary skin file."""
    b = Path(path).read_bytes()
    nvert, ntex, nface, nbone = struct.unpack_from("4i", b, 0)
    off = 16
    vert = np.frombuffer(b, np.float32, 3 * nvert, off).astype(float)
    off += 12 * nvert
    tex = np.frombuffer(b, np.float32, 2 * ntex, off).astype(float)
    off += 8 * ntex
    face = np.frombuffer(b, np.int32, 3 * nface, off).astype(int)
    off += 12 * nface
    bones = []
    for _ in range(nbone):
        name = b[off:off + 40].split(b"\0")[0].decode()
        off += 40
        bp = np.frombuffer(b, np.float32, 3, off).astype(float)
        off += 12
        bq = np.frombuffer(b, np.float32, 4, off).astype(float)
        off += 16
        (vc,) = struct.unpack_from("i", b, off)
        off += 4
        vid = np.frombuffer(b, np.int32, vc, off).astype(int)
        off += 4 * vc
        vw = np.frombuffer(b, np.float32, vc, off).astype(float)
        off += 4 * vc
        bones.append((name, bp, bq, vid, vw))
    return vert, tex, face, bones


@dataclass
class DogScaling:
    length_scale: float
    mass_factor: float          # density multiplier
    body_mass: float            # kg
    withers_height: float       # m
    base_mass: float            # kg of the unscaled model

    @property
    def torque_scale(self) -> float:
        return (self.body_mass / self.base_mass) * self.length_scale

    @property
    def damping_scale(self) -> float:
        return (self.body_mass / self.base_mass) * self.length_scale ** 1.5

    @property
    def inertia_scale(self) -> float:
        return (self.body_mass / self.base_mass) * self.length_scale ** 2

    @property
    def chest_depth(self) -> float:
        return BASE_CHEST_DEPTH * self.length_scale

    @property
    def chest_half_width(self) -> float:
        return BASE_CHEST_HALF_WIDTH * self.length_scale

    @property
    def neck_radius(self) -> float:
        return BASE_NECK_RADIUS * self.length_scale


def load_dog_spec(model_path: Optional[str] = None, keep_muscle_skins: bool = False) -> mujoco.MjSpec:
    path = Path(model_path) if model_path else DEFAULT_MODEL
    spec = mujoco.MjSpec.from_file(str(path))
    skin_dir = path.parent
    for sk in list(spec.skins):
        if sk.name != "skin" and not keep_muscle_skins:
            spec.delete(sk)
    # embed the skin data so it can be scaled
    for sk in spec.skins:
        if not sk.file:
            continue
        vert, tex, face, bones = read_skn(skin_dir / sk.file)
        sk.file = ""
        sk.vert = vert.tolist()
        sk.texcoord = tex.tolist()
        sk.face = face.tolist()
        sk.bodyname = [bn[0] for bn in bones]
        sk.bindpos = np.concatenate([bn[1] for bn in bones]).tolist()
        sk.bindquat = np.concatenate([bn[2] for bn in bones]).tolist()
        sk.vertid = [bn[3].tolist() for bn in bones]
        sk.vertweight = [bn[4].tolist() for bn in bones]
    # make asset paths absolute so the spec compiles from anywhere
    for me in spec.meshes:
        if me.file and not Path(me.file).is_absolute():
            me.file = str((path.parent / me.file).resolve())
    for tx in spec.textures:
        if tx.file and not Path(tx.file).is_absolute():
            tx.file = str((path.parent / tx.file).resolve())
    return spec


_BASE_MASS_CACHE = {}


def base_mass(model_path: Optional[str] = None) -> float:
    key = str(model_path or DEFAULT_MODEL)
    if key not in _BASE_MASS_CACHE:
        m = load_dog_spec(model_path).compile()
        _BASE_MASS_CACHE[key] = float(m.body_subtreemass[1])
    return _BASE_MASS_CACHE[key]


def redistribute_mass(spec: mujoco.MjSpec, fractions: dict = ANATOMICAL_FRACTIONS) -> dict:
    """Rescale segment densities so the segment masses follow ``fractions`` (total mass unchanged)."""
    m = spec.compile()
    total = float(m.body_subtreemass[1])
    seg_mass = {}
    for i in range(1, m.nbody):
        seg = segment_of(m.body(i).name)
        seg_mass[seg] = seg_mass.get(seg, 0.0) + float(m.body_mass[i])
    factor = {seg: fractions[seg] * total / mass for seg, mass in seg_mass.items() if mass > 0}
    for body in spec.bodies:
        if body.name == "world":
            continue
        f = factor[segment_of(body.name)]
        if body.explicitinertial:
            body.mass = body.mass * f
            body.inertia = np.asarray(body.inertia) * f
        for g in body.geoms:
            g.density = g.density * f
            if not np.isnan(g.mass):
                g.mass = g.mass * f
    return factor


def scale_dog_spec(spec: mujoco.MjSpec, body_mass: float, withers_height: float,
                   model_path: Optional[str] = None, mass_distribution: str = "anatomical") -> DogScaling:
    """Scale the dog in ``spec`` in place; returns the scaling factors."""
    m0 = base_mass(model_path)
    s = withers_height / BASE_WITHERS_HEIGHT
    k = body_mass / (m0 * s ** 3)
    sc = DogScaling(length_scale=s, mass_factor=k, body_mass=body_mass, withers_height=withers_height, base_mass=m0)
    for body in spec.bodies:
        if body.name == "world":
            continue
        body.pos = np.asarray(body.pos) * s
        body.ipos = np.asarray(body.ipos) * s
        if body.explicitinertial:
            body.mass = body.mass * k * s ** 3
            body.inertia = np.asarray(body.inertia) * k * s ** 5
            if not np.all(np.isnan(body.fullinertia)):
                body.fullinertia = np.asarray(body.fullinertia) * k * s ** 5
    for g in spec.geoms:
        g.pos = np.asarray(g.pos) * s
        if g.type in PRIMITIVES_WITH_LENGTH_SIZE:
            g.size = np.asarray(g.size) * s
        if not np.isnan(g.mass):
            g.mass = g.mass * k * s ** 3
        g.density = g.density * k
        g.margin = g.margin * s
    for me in spec.meshes:
        me.scale = np.asarray(me.scale) * s
    for site in spec.sites:
        site.pos = np.asarray(site.pos) * s
        site.size = np.asarray(site.size) * s
    for j in spec.joints:
        j.pos = np.asarray(j.pos) * s
        j.stiffness = np.asarray(j.stiffness) * sc.torque_scale
        j.damping = np.asarray(j.damping) * sc.damping_scale
        j.armature = j.armature * sc.inertia_scale
        j.frictionloss = j.frictionloss * sc.torque_scale
    for a in spec.actuators:
        gp = np.array(a.gainprm)
        gp[0] *= sc.torque_scale
        a.gainprm = gp
    for t in spec.tendons:
        t.width = t.width * s
        if hasattr(t, "stiffness"):
            t.stiffness = np.asarray(t.stiffness) * sc.torque_scale
            t.damping = np.asarray(t.damping) * sc.damping_scale
    for cam in spec.cameras:
        cam.pos = np.asarray(cam.pos) * s
    for sk in spec.skins:
        sk.vert = (np.asarray(sk.vert) * s).tolist()
        sk.bindpos = (np.asarray(sk.bindpos) * s).tolist()
        sk.inflate = sk.inflate * s
    if mass_distribution == "anatomical":
        redistribute_mass(spec)
    return sc
