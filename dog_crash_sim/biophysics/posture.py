"""Postures of the dog at the start of an event.

A posture is a set of joint targets (degrees) plus the orientation of the
trunk.  Joint conventions of the dog model: positive ``*_extend`` at the
hip/shoulder/scapula swings the limb backwards; negative knee/ankle/elbow
values fold the distal segment; positive cervical ``*_extend`` lowers the
nose.  The trunk pitch is a rotation about the dog's lateral axis
(negative = nose up).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Posture:
    name: str
    joints: dict = field(default_factory=dict)   # joint name -> degrees
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    # where the dog sits on its support: offset (m, in the dog heading frame, unscaled dog) of the support
    # reference point from the root body, used to place it
    rear_anchor: bool = False                    # place the rump against the seat back


SITTING = Posture(
    "sitting",
    joints=dict(hip_L_extend=-95, hip_R_extend=-95, knee_L=-90, knee_R=-90, ankle_L=-75, ankle_R=-75,
                scapula_L_extend=15, scapula_R_extend=15, shoulder_L_extend=20, shoulder_R_extend=20,
                elbow_L=10, elbow_R=10, wrist_L=0, wrist_R=0,
                C_7_extend=20, C_5_extend=8, C_3_extend=10,
                L_1_extend=-5, L_3_extend=-5, L_5_extend=-5, L_7_extend=-5),
    pitch_deg=-45.0, rear_anchor=True)

LYING_STERNAL = Posture(
    "lying_sternal",
    joints=dict(hip_L_extend=-100, hip_R_extend=-100, knee_L=-95, knee_R=-95, ankle_L=-60, ankle_R=-60,
                hip_L_abduct=-20, hip_R_abduct=-20,
                shoulder_L_extend=15, shoulder_R_extend=15, elbow_L=-100, elbow_R=-100,
                scapula_L_extend=0, scapula_R_extend=0, wrist_L=5, wrist_R=5, C_7_extend=-10),
    pitch_deg=0.0)

LYING_LATERAL = Posture(
    "lying_lateral",
    joints=dict(hip_L_extend=-40, hip_R_extend=-30, knee_L=-50, knee_R=-45, ankle_L=-40, ankle_R=-40,
                shoulder_L_extend=-25, shoulder_R_extend=-20, elbow_L=-40, elbow_R=-35,
                wrist_L=-10, wrist_R=-10, C_7_extend=10, C_7_bend=-10),
    pitch_deg=0.0, roll_deg=-90.0)

STANDING = Posture(
    "standing",
    joints=dict(hip_L_extend=-8, hip_R_extend=-8, knee_L=-8, knee_R=-8, ankle_L=-5, ankle_R=-5,
                elbow_L=-3, elbow_R=-3),
    pitch_deg=0.0)

POSTURES = {p.name: p for p in (SITTING, LYING_STERNAL, LYING_LATERAL, STANDING)}


def default_facing(posture: str, seating_position: str) -> float:
    """Default heading (deg) of the dog in the vehicle."""
    if posture in ("lying_sternal", "lying_lateral"):
        if seating_position == "rear_footwell":
            return 90.0
        if seating_position == "cargo_area":
            return 0.0
        return -90.0 if seating_position == "rear_left" else 90.0
    return 0.0


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Quaternion (w, x, y, z) of R = Rz(yaw) Ry(pitch) Rx(roll), radians."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cy * cp * cr + sy * sp * sr,
                     cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr,
                     sy * cp * cr - cy * sp * sr])
