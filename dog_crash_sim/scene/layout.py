"""Key positions of the rear compartment derived from the scenario.

Everything is expressed in the vehicle frame (x forward, y left, z up, origin
on the ground below the CG).  Both the vehicle crash model (which needs to
know where the dog sits to localise intrusion) and the MuJoCo scene builder
use this single source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import ScenarioConfig


@dataclass
class Layout:
    x_front_bumper: float
    x_rear_bumper: float
    x_front_axle: float
    x_rear_axle: float
    half_width: float
    # rear bench
    bight: np.ndarray                 # (x, z) of the seat bight
    cushion_front: np.ndarray         # (x, z) of the cushion front edge (top surface)
    cushion_angle: float              # rad
    seat_back_angle: float            # rad, recline from vertical
    seat_back_top: np.ndarray         # (x, z) of the top of the seat-back front face
    seat_back_hinge: np.ndarray       # (x, z)
    seat_y: dict = field(default_factory=dict)   # seating position -> y
    # trunk / cargo area
    trunk_front_x: float = 0.0
    trunk_rear_x: float = 0.0
    trunk_floor_z: float = 0.0
    trunk_top_z: float = 0.0
    trunk_half_width: float = 0.0
    # the dog
    dog_ref: np.ndarray = None        # reference point under the dog (x, y, z of the support surface)
    dog_seat: str = "rear_left"       # seating position whose belt/anchors the dog uses
    support: str = "cushion"          # cushion | floor | cargo_floor
    outboard_sign: float = 1.0        # +1 if the dog's outboard side is +y (left)
    b_pillar_x: float = 0.0
    c_pillar_x: float = 0.0

    def seatback_rear_x(self, z: float, thickness: float) -> float:
        """x of the rear (trunk-side) face of the upright seat back at height z."""
        hx, hz = self.seat_back_hinge
        return float(hx - 0.5 * thickness / math.cos(self.seat_back_angle) - max(z - hz, 0.0) * math.tan(self.seat_back_angle))

    def cushion_z(self, x: float) -> float:
        """Height of the cushion top surface at x."""
        return float(self.bight[1] + (x - self.bight[0]) * math.tan(self.cushion_angle))


def compute_layout(cfg: ScenarioConfig) -> Layout:
    v = cfg.vehicle
    rc = v.rear_compartment
    rp = cfg.rear_passenger
    x_fb = v.cg_to_front_axle_m + v.front_overhang_m
    x_rb = x_fb - v.length_m
    bight = np.array([rc.seat_bight_x_m, rc.cushion_height_m])
    ca = math.radians(rc.cushion_angle_deg)
    cushion_front = bight + rc.cushion_length_m * np.array([math.cos(ca), math.sin(ca)])
    sba = math.radians(rc.seat_back_angle_deg)
    seat_back_top = bight + rc.seat_back_height_m * np.array([-math.sin(sba), math.cos(sba)])
    hinge = bight + np.array([-0.5 * rc.seat_back_thickness_m, -0.04])
    outboard = 0.295 * rc.seat_width_m
    seat_y = {"rear_left": outboard, "rear_center": 0.0, "rear_right": -outboard}

    trunk_front_x = rc.seat_bight_x_m - rc.seat_back_thickness_m - 0.04
    trunk_rear_x = trunk_front_x - rc.trunk_length_m
    trunk_rear_x = max(trunk_rear_x, x_rb + 0.12)

    lay = Layout(
        x_front_bumper=x_fb, x_rear_bumper=x_rb,
        x_front_axle=v.cg_to_front_axle_m, x_rear_axle=v.cg_to_front_axle_m - v.wheelbase_m,
        half_width=0.5 * v.width_m,
        bight=bight, cushion_front=cushion_front, cushion_angle=ca, seat_back_angle=sba,
        seat_back_top=seat_back_top, seat_back_hinge=hinge, seat_y=seat_y,
        trunk_front_x=trunk_front_x, trunk_rear_x=trunk_rear_x,
        trunk_floor_z=rc.trunk_floor_height_m,
        trunk_top_z=rc.trunk_floor_height_m + rc.trunk_height_m,
        trunk_half_width=0.5 * rc.trunk_width_m,
        b_pillar_x=rc.front_seat_back_x_m, c_pillar_x=rc.c_pillar_x_m,
    )

    pos = rp.seating_position
    if pos in seat_y:
        x = bight[0] + 0.45 * rc.cushion_length_m * math.cos(ca)
        lay.dog_ref = np.array([x, seat_y[pos] + rp.lateral_offset_m, lay.cushion_z(x)])
        lay.dog_seat = pos
        lay.support = "cushion"
    elif pos == "rear_footwell":
        y = 0.5 * (rc.cabin_width_m * 0.5 + 0.5 * rc.console_width_m) + rp.lateral_offset_m
        x = 0.5 * (cushion_front[0] + rc.front_seat_back_x_m)
        lay.dog_ref = np.array([x, y, rc.floor_height_m])
        lay.dog_seat = "rear_left"
        lay.support = "floor"
    elif pos == "cargo_area":
        x = 0.5 * (trunk_front_x + trunk_rear_x)
        lay.dog_ref = np.array([x, rp.lateral_offset_m, rc.trunk_floor_height_m])
        lay.dog_seat = "rear_center"
        lay.support = "cargo_floor"
    else:  # pragma: no cover - validated by config
        raise ValueError(pos)
    lay.outboard_sign = 1.0 if lay.dog_ref[1] >= 0 else -1.0
    return lay


def belt_anchor_points(cfg: ScenarioConfig, lay: Layout) -> dict:
    """Vehicle-frame anchor points for the dog's seating position.

    Returns world-fixed points plus points expressed on the seat back
    (``*_on_seatback`` keys, relative to the seat-back hinge, in the
    un-rotated seat-back frame).
    """
    rc = cfg.vehicle.rear_compartment
    belt = cfg.restraints.belt
    seat = lay.dog_seat
    y = lay.seat_y[seat] if seat in lay.seat_y else lay.dog_ref[1]
    s = 1.0 if y > 1e-6 else (-1.0 if y < -1e-6 else 1.0)
    xb, zb = lay.bight
    pts = {}
    if belt.upper_anchor_m is not None:
        pts["upper"] = np.asarray(belt.upper_anchor_m, float)
        pts["upper_on_seatback"] = None
    elif seat == "rear_center":
        pts["upper"] = None
        # seat-integrated retractor at the top of the seat back
        top = lay.seat_back_top - lay.seat_back_hinge
        pts["upper_on_seatback"] = np.array([top[0] - 0.02, 0.0, top[1] + 0.02])
    else:
        pts["upper"] = np.array([rc.c_pillar_x_m + 0.06, s * (0.5 * rc.cabin_width_m - 0.07), zb + 0.62])
        pts["upper_on_seatback"] = None
    if belt.buckle_m is not None:
        pts["buckle"] = np.asarray(belt.buckle_m, float)
    else:
        inboard = -s if seat != "rear_center" else -1.0
        pts["buckle"] = np.array([xb + 0.04, y + inboard * 0.21, zb + 0.03])
    if belt.lap_anchor_m is not None:
        pts["lap"] = np.asarray(belt.lap_anchor_m, float)
    else:
        pts["lap"] = np.array([xb - 0.02, y + s * 0.25, zb - 0.04])
    half = 0.5 * belt.isofix_spacing_m
    pts["isofix_l"] = np.array([xb - 0.02, y + half, zb - 0.02])
    pts["isofix_r"] = np.array([xb - 0.02, y - half, zb - 0.02])
    top = lay.seat_back_top - lay.seat_back_hinge
    pts["headrest_on_seatback"] = np.array([top[0] - 0.05, y, top[1] + 0.03])
    hx0, hx1 = lay.trunk_front_x - 0.08, lay.trunk_rear_x + 0.08
    hy = lay.trunk_half_width - 0.08
    z = lay.trunk_floor_z + 0.01
    pts["cargo_hooks"] = [np.array([hx0, hy, z]), np.array([hx0, -hy, z]),
                          np.array([hx1, hy, z]), np.array([hx1, -hy, z])]
    return pts
