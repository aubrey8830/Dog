"""Parametric 3D harness: geometry, visual meshes and mechanical model.

The harness is represented as a light rigid *harness frame* (back pad,
girth band, chest plate and attachment ring) that sits on the thorax and
is connected to it through a six-degree-of-freedom bushing (three slides,
three hinges).  The bushing stiffness, travel and damping describe how the
fitted straps grip the chest: the travel is the fit slack plus padding
compression, the stiffness depends on the architecture (a padded chest
plate resists rearward pull much more than a step-in loop).  Body straps
that cross other body segments are modelled as tension-only webbing
tendons routed through via points on the dog:

* the **neck loop** passes from the harness at the withers around the base
  of the neck (C7) across the throat;
* the horizontal **front strap** (Norwegian style) runs across the points
  of the shoulders (upper arms) and the prosternum.

Strap visuals are watertight ribbon meshes generated from cross sections
of the dog's skin so they hug the 3D body of the scaled dog.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import HarnessConfig
from ..biophysics.dog_model import DogScaling

# Harness landmark coordinates on the *unscaled* dog in its neutral standing pose (world frame of dog.xml).
GIRTH_X = 0.06
BACK_PAD = (-0.03, 0.15)
DORSAL_RING = np.array([0.085, 0.0, 0.0])      # z filled from the skin
VENTRAL_RING = np.array([0.215, 0.0, 0.0])
NECK_DORSAL = np.array([0.150, 0.0, 0.600])
NECK_VENTRAL = np.array([0.262, 0.0, 0.440])
FRONT_STRAP_Z = 0.455

# style -> bushing/strap properties for a 20 kg reference dog
STYLE = {
    # kx_*: bushing stiffness along the body (N/m); neck_slack: free play of the neck loop (m);
    # neck_on_throat: the ventral part of the neck loop bears on the throat (else on the prosternum)
    "padded_vest": dict(kx_rear=26e3, kx_fwd=6e3, neck_slack=0.040, neck_on_throat=False, front=False, plate=True),
    "y_front": dict(kx_rear=9e3, kx_fwd=5e3, neck_slack=0.012, neck_on_throat=True, front=False, plate=False),
    "norwegian": dict(kx_rear=2.5e3, kx_fwd=4e3, neck_slack=None, neck_on_throat=False, front=True, plate=False),
    "step_in": dict(kx_rear=1.2e3, kx_fwd=3e3, neck_slack=0.004, neck_on_throat=True, front=False, plate=False),
    "collar_only": dict(kx_rear=0.0, kx_fwd=0.0, neck_slack=None, neck_on_throat=True, front=False, plate=False),
}


def section_outline(V, origin, normal, u_axis, tol=0.006, nbins=64, smooth=2):
    """Closed outline of the point cloud ``V`` cut by a plane, ordered by angle.

    Returns (points (n,3), centroid (3,)).
    """
    normal = normal / np.linalg.norm(normal)
    u = u_axis - normal * np.dot(u_axis, normal)
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    rel = V - origin
    dist = rel @ normal
    sel = rel[np.abs(dist) < tol]
    if len(sel) < 20:
        raise ValueError("too few skin vertices near the section plane")
    pu, pw = sel @ u, sel @ w
    cu, cw = pu.mean(), pw.mean()
    ang = np.arctan2(pw - cw, pu - cu)
    rad = np.hypot(pu - cu, pw - cw)
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    idx = np.clip(np.digitize(ang, bins) - 1, 0, nbins - 1)
    r = np.full(nbins, np.nan)
    for i in range(nbins):
        if np.any(idx == i):
            r[i] = rad[idx == i].max()
    good = ~np.isnan(r)
    centers = 0.5 * (bins[:-1] + bins[1:])
    r = np.interp(centers, centers[good], r[good], period=2 * np.pi)
    for _ in range(smooth):
        r = 0.25 * np.roll(r, 1) + 0.5 * r + 0.25 * np.roll(r, -1)
    c3 = origin + cu * u + cw * w
    pts = c3[None, :] + (r * np.cos(centers))[:, None] * u[None, :] + (r * np.sin(centers))[:, None] * w[None, :]
    return pts, c3


def ribbon_mesh(path, centroid, width_dir, width, thickness, offset, closed=True):
    """Watertight band (rectangular cross-section) following ``path``.

    ``width_dir`` is a unit vector (or per-point array) across the strap
    width; the band is pushed outward from ``centroid`` by ``offset``.
    """
    path = np.asarray(path, float)
    n = len(path)
    wd = np.broadcast_to(np.asarray(width_dir, float), (n, 3))
    out = path - np.asarray(centroid)[None, :] if np.ndim(centroid) == 1 else path - centroid
    out = out - wd * np.sum(out * wd, axis=1, keepdims=True)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    inner = path + out * offset
    outer = inner + out * thickness
    hw = 0.5 * width
    verts = []
    for i in range(n):
        verts += [inner[i] - wd[i] * hw, inner[i] + wd[i] * hw, outer[i] + wd[i] * hw, outer[i] - wd[i] * hw]
    verts = np.array(verts)
    faces = []
    segs = n if closed else n - 1
    for i in range(segs):
        j = (i + 1) % n
        a, b = 4 * i, 4 * j
        for k in range(4):
            k2 = (k + 1) % 4
            faces += [[a + k, b + k, b + k2], [a + k, b + k2, a + k2]]
    if not closed:
        faces += [[0, 2, 1], [0, 3, 2]]
        e = 4 * (n - 1)
        faces += [[e, e + 1, e + 2], [e, e + 2, e + 3]]
    faces = np.array(faces)
    # double-sided so the band renders correctly whatever the path orientation
    return verts, np.vstack([faces, faces[:, ::-1]])


def torus_mesh(center, axis, radius, tube, nu=24, nv=8):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    verts = []
    for i in range(nu):
        th = 2 * np.pi * i / nu
        c = center + radius * (np.cos(th) * e1 + np.sin(th) * e2)
        radial = np.cos(th) * e1 + np.sin(th) * e2
        for j in range(nv):
            ph = 2 * np.pi * j / nv
            verts.append(c + tube * (np.cos(ph) * radial + np.sin(ph) * axis))
    faces = []
    for i in range(nu):
        for j in range(nv):
            a0 = i * nv + j
            a1 = ((i + 1) % nu) * nv + j
            b0 = i * nv + (j + 1) % nv
            b1 = ((i + 1) % nu) * nv + (j + 1) % nv
            faces += [[a0, a1, b1], [a0, b1, b0]]
    faces = np.array(faces)
    return np.array(verts), np.vstack([faces, faces[:, ::-1]])


@dataclass
class HarnessGeometry:
    """Landmarks (world frame of the scaled, neutral dog) and visual meshes."""
    style: str
    dring_world: np.ndarray
    neck_dorsal_L: np.ndarray
    neck_dorsal_R: np.ndarray
    neck_lat_L: np.ndarray
    neck_lat_R: np.ndarray
    neck_ventral: np.ndarray
    front_side_L: np.ndarray
    front_side_R: np.ndarray
    shoulder_front_L: np.ndarray
    shoulder_front_R: np.ndarray
    prosternum: np.ndarray
    girth_center: np.ndarray
    collar_ring: np.ndarray
    meshes: dict = field(default_factory=dict)   # name -> (body, verts, faces)
    perimeter_girth: float = 0.0


def build_geometry(skin_vertices: np.ndarray, sc: DogScaling, h: HarnessConfig) -> HarnessGeometry:
    """Construct harness landmarks and ribbon meshes from the (scaled) skin vertices at the bind pose."""
    s = sc.length_scale
    V = np.asarray(skin_vertices).reshape(-1, 3)
    w = h.strap_width_mm * 1e-3 * min(1.0, max(0.6, s))
    pad = h.padding_thickness_mm * 1e-3
    thick = 0.003 + 0.6 * pad
    gap = 0.002
    meshes = {}

    # girth band
    tol = 0.006 * max(s, 1.0)
    girth, gc = section_outline(V, np.array([GIRTH_X * s, 0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), tol=tol)
    per = float(np.sum(np.linalg.norm(np.roll(girth, -1, axis=0) - girth, axis=1)))
    meshes["girth_band"] = ("harness", *ribbon_mesh(girth, gc, np.array([1.0, 0, 0]), max(w, 0.02 * s) * 1.3, thick, gap))

    # back pad: a series of sections along the back, upper arc only
    xs = np.linspace(BACK_PAD[0] * s, BACK_PAD[1] * s, 7)
    tops = []
    for x in xs:
        pts, c = section_outline(V, np.array([x, 0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), tol=tol)
        tops.append(pts[np.argmax(pts[:, 2])])
    tops = np.array(tops)
    pw = min(0.09 * s, 2 * sc.chest_half_width * 0.75)
    meshes["back_pad"] = ("harness", *ribbon_mesh(tops, tops.mean(axis=0) - np.array([0, 0, 0.2 * s]),
                                                    np.array([0, 1.0, 0]), pw, thick + 0.004, gap, closed=False))
    dorsal_top = tops[np.argmin(np.abs(xs - DORSAL_RING[0] * s))]

    # chest plate / Y strap along the ventral midline: girth -> brisket -> point of the chest
    prosternum = np.array([0.252 * s, 0.0, 0.0])
    sel = V[(np.abs(V[:, 1]) < 0.015 * s + 0.004) & (np.abs(V[:, 2] - 0.43 * s) < 0.015 * s)]
    prosternum[0] = sel[:, 0].max()
    prosternum[2] = 0.43 * s
    girth_bottom = girth[np.argmin(girth[:, 2])]
    brisket_x = 0.17 * s
    selb = V[(np.abs(V[:, 0] - brisket_x) < tol) & (np.abs(V[:, 1]) < 0.015 * s + 0.004)]
    brisket = np.array([brisket_x, 0.0, selb[:, 2].min()])
    knots = np.array([[girth_bottom[0], 0.0, girth_bottom[2]], brisket,
                      [0.5 * (brisket[0] + prosternum[0]) + 0.01 * s, 0.0, 0.5 * (brisket[2] + prosternum[2]) - 0.01 * s],
                      prosternum - np.array([0.004 * s, 0, 0])])
    seg = np.concatenate([np.linspace(knots[i], knots[i + 1], 4, endpoint=False) for i in range(3)] + [knots[-1:]])
    bots = seg
    if h.style in ("padded_vest", "y_front"):
        area = h.chest_plate_area_cm2 * 1e-4 if h.style == "padded_vest" else 0.0
        length = float(np.sum(np.linalg.norm(np.diff(bots, axis=0), axis=1)))
        width_plate = max(w, min(area / max(length, 1e-3), 2 * sc.chest_half_width * 0.8)) if area > 0 else w
        inside = np.array([0.08 * s, 0.0, 0.43 * s])  # a point inside the chest
        meshes["chest_plate"] = ("harness", *ribbon_mesh(bots, inside, np.array([0, 1.0, 0]), width_plate,
                                                           thick + 0.003, gap, closed=False))

    # neck loop on the base of the neck (attached to C7 for display)
    nd = NECK_DORSAL * s
    nv = NECK_VENTRAL * s
    v_dir = nv - nd
    n_plane = np.cross(np.array([0, 1.0, 0]), v_dir)
    n_plane /= np.linalg.norm(n_plane)
    neck, ncent = section_outline(V, 0.5 * (nd + nv), n_plane, v_dir / np.linalg.norm(v_dir), tol=0.007 * max(s, 1.0))
    if h.style not in ("norwegian", "collar_only"):
        meshes["neck_loop"] = ("C_7", *ribbon_mesh(neck, ncent, n_plane, w, thick, gap))
    yl = neck[:, 1]
    lat_L = neck[np.argmax(yl)]
    lat_R = neck[np.argmin(yl)]
    ventral = neck[np.argmin(neck[:, 2])]
    dorsal = neck[np.argmax(neck[:, 2])]

    # horizontal front strap (Norwegian)
    fz = FRONT_STRAP_Z * s
    front, fcent = section_outline(V, np.array([0, 0, fz]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), tol=tol)
    front_half = front[front[:, 0] > GIRTH_X * s]
    order = np.argsort(np.arctan2(front_half[:, 1] - fcent[1], front_half[:, 0] - fcent[0]))
    front_half = front_half[order]
    if h.style == "norwegian":
        meshes["front_strap"] = ("harness", *ribbon_mesh(front_half, fcent, np.array([0, 0, 1.0]), w * 1.2, thick,
                                                           gap + 0.006 * s, closed=False))
    side_L = front_half[np.argmax(front_half[:, 1])]
    side_R = front_half[np.argmin(front_half[:, 1])]
    frontmost = front_half[np.argmax(front_half[:, 0])]
    sh_L = front_half[np.argmin(np.abs(front_half[:, 1] - 0.6 * side_L[1]) + 10 * (front_half[:, 1] < 0))]
    sh_R = front_half[np.argmin(np.abs(front_half[:, 1] - 0.6 * side_R[1]) + 10 * (front_half[:, 1] > 0))]

    # attachment ring
    if h.attachment_point == "dorsal":
        ring = dorsal_top + np.array([0, 0, thick + 0.012 * s])
        ring_axis = np.array([0, 1.0, 0])
    else:
        ring = bots[-2] - np.array([0, 0, thick + 0.012 * s])
        ring_axis = np.array([0, 1.0, 0])
    r_ring = 0.016 * max(s, 0.6)
    if h.style != "collar_only":
        meshes["ring"] = ("harness", *torus_mesh(ring, ring_axis, r_ring, 0.0035 * max(s, 0.6)))

    # collar (reference case): ring around the neck higher up, attached to C_5
    collar_pts, ccent = section_outline(V, np.array([0.225 * s, 0, 0.60 * s]), np.array([0.5, 0, 0.87]),
                                        np.array([1.0, 0, -0.57]), tol=tol)
    collar_ring = collar_pts[np.argmax(collar_pts[:, 2])] + np.array([0, 0, 0.015 * s])
    if h.style == "collar_only":
        meshes["collar"] = ("C_5", *ribbon_mesh(collar_pts, ccent, np.array([0.5, 0, 0.87]) / np.linalg.norm([0.5, 0, 0.87]),
                                                w, thick, gap))
        meshes["ring"] = ("C_5", *torus_mesh(collar_ring, np.array([0, 1.0, 0]), r_ring, 0.0035 * max(s, 0.6)))

    return HarnessGeometry(
        style=h.style, dring_world=ring, neck_dorsal_L=dorsal + np.array([0, 0.25 * sc.neck_radius, 0]),
        neck_dorsal_R=dorsal - np.array([0, 0.25 * sc.neck_radius, 0]), neck_lat_L=lat_L, neck_lat_R=lat_R,
        neck_ventral=ventral, front_side_L=side_L, front_side_R=side_R, shoulder_front_L=sh_L, shoulder_front_R=sh_R,
        prosternum=frontmost, girth_center=gc, collar_ring=collar_ring, meshes=meshes, perimeter_girth=per)


def bushing_params(h: HarnessConfig, sc: DogScaling) -> dict:
    """Stiffness/travel of the harness-to-thorax bushing for the style and dog size."""
    st = STYLE[h.style]
    size = (sc.body_mass / 20.0) ** (2.0 / 3.0)
    slack = h.fit_slack_mm * 1e-3
    pad = h.padding_thickness_mm * 1e-3
    return dict(
        kx_rear=st["kx_rear"] * size, kx_fwd=st["kx_fwd"] * size,
        x_range=(-(slack + 0.6 * pad + 0.01 * sc.length_scale), slack + 0.03 * sc.length_scale),
        k_yz=25e3 * size, yz_range=slack + 0.6 * pad + 0.008,
        k_roll=60.0 * size * sc.length_scale, roll_range=math.radians(55.0),
        k_pitch=250.0 * size * sc.length_scale, pitch_range=math.radians(14.0),
        k_yaw=200.0 * size * sc.length_scale, yaw_range=math.radians(14.0),
        neck_slack=None if st["neck_slack"] is None else st["neck_slack"] + slack,
        front_strap=st["front"], chest_plate=st["plate"], neck_on_throat=st["neck_on_throat"],
    )
