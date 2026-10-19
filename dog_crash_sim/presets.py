"""Named scenario presets (standard crash-test configurations).

Each preset is a partial scenario; anything not given keeps its default.
The test descriptions approximate the public protocols - they are not
certified reproductions of them.
"""

from __future__ import annotations

import copy

NO_BRAKING = {"braking": {"enabled": False}, "pre_impact_duration_s": 0.25}

PRESETS = {
    "cps_sled_frontal": {
        "description": "Harness crashworthiness sled test in the style of the Center for Pet Safety protocol: "
                       "FMVSS 213-like 48 km/h sled pulse, open bench-seat buck, 45 lb (20.4 kg) dog dummy.",
        "collision": {"mode": "sled", "sled": {"shape": "fmvss213", "delta_v_kph": 48.0}},
        "vehicle": {"rear_compartment": {"front_seats_present": False, "roof_present": False, "doors_present": False}},
        "rear_passenger": {"breed_size": "cps_45lb", "posture": "sitting", "neuromuscular": {"muscle_tone": 0.1,
                                                                                              "startle_reflex": False}},
        "restraints": {"airbags": [], "pretensioner": {"enabled": False}, "load_limiter": {"enabled": False}},
        "trunk_cargo": {"items": []},
        "environment": NO_BRAKING,
    },
    "frontal_full_width_56": {
        "description": "Full-width frontal impact into a rigid wall at 56 km/h (US NCAP style), driver braking beforehand.",
        "collision": {"impact_speed_kph": 56.0, "impact_direction": "frontal", "overlap": 1.0,
                      "partner": {"type": "rigid_barrier"}},
    },
    "frontal_offset_64": {
        "description": "40 % offset frontal impact into a deformable barrier at 64 km/h (Euro NCAP ODB style).",
        "collision": {"impact_speed_kph": 64.0, "overlap": 0.4, "overlap_side": "left",
                      "partner": {"type": "deformable_barrier"}},
        "environment": NO_BRAKING,
    },
    "small_overlap_64": {
        "description": "25 % small-overlap frontal impact into a rigid barrier at 64 km/h (IIHS style).",
        "collision": {"impact_speed_kph": 64.0, "overlap": 0.25, "overlap_side": "left",
                      "partner": {"type": "rigid_barrier"}},
        "environment": NO_BRAKING,
    },
    "oblique_mdb_90": {
        "description": "Oblique moving deformable barrier (2486 kg, 90 km/h, 15 deg, 35 % overlap) into a stationary car.",
        "collision": {"impact_speed_kph": 0.0, "impact_direction": "frontal", "impact_angle_deg": 15.0, "overlap": 0.35,
                      "overlap_side": "left", "partner": {"type": "moving_barrier", "mass_kg": 2486.0,
                                                          "speed_kph": 90.0, "engaged_face": "front"}},
        "environment": NO_BRAKING,
    },
    "car_to_car_frontal_50": {
        "description": "Head-on impact with a similar car, both at 50 km/h, 50 % overlap.",
        "collision": {"impact_speed_kph": 50.0, "overlap": 0.5, "partner": {"type": "vehicle", "speed_kph": 50.0}},
    },
    "side_mdb_60": {
        "description": "Side impact by a 1400 kg moving deformable barrier at 60 km/h into the dog's side at the rear row.",
        "collision": {"impact_speed_kph": 0.0, "impact_direction": "left",
                      "partner": {"type": "moving_barrier", "mass_kg": 1400.0, "speed_kph": 60.0}},
        "rear_passenger": {"seating_position": "rear_left"},
        "environment": NO_BRAKING,
    },
    "side_mdb_60_far_side": {
        "description": "As side_mdb_60 but struck on the side away from the dog (far-side impact).",
        "collision": {"impact_speed_kph": 0.0, "impact_direction": "right",
                      "partner": {"type": "moving_barrier", "mass_kg": 1400.0, "speed_kph": 60.0}},
        "rear_passenger": {"seating_position": "rear_left"},
        "environment": NO_BRAKING,
    },
    "side_pole_32": {
        "description": "Vehicle sliding sideways (75 deg) into a 254 mm rigid pole at 32 km/h at the dog's row.",
        "collision": {"impact_speed_kph": 32.0, "impact_direction": "left", "partner": {"type": "pole"}},
        "environment": {"braking": {"enabled": False}, "pre_impact_duration_s": 0.2, "motion": {"sideslip_deg": 75.0}},
    },
    "rear_mdb_80": {
        "description": "Rear impact by a 1368 kg moving barrier at 80 km/h, 70 % overlap (FMVSS 301 style), with cargo.",
        "collision": {"impact_speed_kph": 0.0, "impact_direction": "rear", "overlap": 0.7,
                      "partner": {"type": "moving_barrier", "mass_kg": 1368.0, "speed_kph": 80.0}},
        "environment": NO_BRAKING,
    },
    "rear_low_speed_16": {
        "description": "Low-speed rear-end collision: a 1500 kg car at 32 km/h strikes the stationary vehicle (~16 km/h delta-V).",
        "collision": {"impact_speed_kph": 0.0, "impact_direction": "rear",
                      "partner": {"type": "vehicle", "mass_kg": 1500.0, "speed_kph": 32.0}},
        "environment": {"braking": {"enabled": True, "onset_before_impact_s": 2.0, "deceleration_g": 0.3},
                        "pre_impact_duration_s": 0.3},
    },
    "suv_cargo_intrusion": {
        "description": "SUV, 56 km/h full-width frontal impact with a 35 kg unrestrained crate loose in the cargo area.",
        "vehicle": {"preset": "midsize_suv"},
        "collision": {"impact_speed_kph": 56.0},
        "trunk_cargo": {"items": [{"name": "crate", "mass_kg": 35.0, "size_m": [0.5, 0.6, 0.45]}],
                        "packing": "loose_rear", "restraint_method": "none"},
        "rear_passenger": {"seating_position": "rear_center"},
    },
    "truck_underride_56": {
        "description": "Frontal impact at 56 km/h into the rear of a stationary truck (high, stiff structure - underride).",
        "collision": {"impact_speed_kph": 56.0, "partner": {"type": "truck", "speed_kph": 0.0}},
    },
    "evasive_swerve_ice_pole": {
        "description": "Evasive steering on ice ending in a sideways slide into a pole at 40 km/h.",
        "collision": {"impact_speed_kph": 40.0, "impact_direction": "right", "partner": {"type": "pole"}},
        "environment": {"road_condition": "ice", "pre_impact_duration_s": 0.8,
                        "braking": {"enabled": True, "deceleration_g": 0.5, "abs": False},
                        "motion": {"sideslip_deg": -60.0, "initial_yaw_rate_deg_s": -30.0, "steer_angle_deg": -8.0}},
    },
}


def preset(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {', '.join(PRESETS)}")
    p = copy.deepcopy(PRESETS[name])
    desc = p.pop("description")
    p.setdefault("name", name)
    p.setdefault("description", desc)
    return p
