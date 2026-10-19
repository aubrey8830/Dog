"""Scenario configuration for the rear-seat dog harness crash simulator.

A scenario is described by nine groups of parameters.  The first six are the
test variables of a rear-seat crash and map one-to-one onto the categories the
simulator was designed around:

=================  ==========================================================
``collision``      impact speed, impact direction, impact angle, overlap and
                   collision partner (plus an optional sled-pulse mode)
``vehicle``        mass, dimensions, structural stiffness, crush zones and
                   rear-compartment geometry
``rear_passenger`` the dog: body size, seating position, posture, whether the
                   harness is connected to the vehicle (seat belt used) and
                   its neuromuscular state
``restraints``     belt geometry, pretensioner, load limiter, airbags, seat
                   design and head restraints
``trunk_cargo``    cargo mass, dimensions, position, packing arrangement and
                   restraint method
``environment``    road friction, pre-impact braking and vehicle movement
                   before impact
=================  ==========================================================

The remaining groups describe the device under test (``harness``), numerical
settings (``simulation``) and the injury/performance criteria
(``assessment``).

Coordinates: the vehicle frame has x forward, y to the left and z up, with
its origin on the ground directly below the vehicle centre of gravity.
Every field carries a unit and a description in its metadata; run
``python -m dog_crash_sim params`` to print them all.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import yaml

KPH = 1.0 / 3.6
G = 9.80665


def F(default, unit: str = "", help: str = "", choices: Optional[list] = None):
    """Dataclass field with documentation metadata."""
    meta = {"unit": unit, "help": help}
    if choices is not None:
        meta["choices"] = list(choices)
    if isinstance(default, (list, dict)) or dataclasses.is_dataclass(default):
        return field(default_factory=lambda d=default: copy.deepcopy(d), metadata=meta)
    return field(default=default, metadata=meta)


def S(factory, help: str = ""):
    """Nested-section dataclass field."""
    return field(default_factory=factory, metadata={"unit": "", "help": help})


# --------------------------------------------------------------------------
# Collision
# --------------------------------------------------------------------------

PARTNER_TYPES = ["rigid_barrier", "deformable_barrier", "pole", "vehicle", "moving_barrier", "truck"]
IMPACT_DIRECTIONS = ["frontal", "rear", "left", "right"]


@dataclass
class PartnerConfig:
    type: str = F("rigid_barrier", "", "Collision partner. rigid_barrier: flat concrete wall; "
                  "deformable_barrier: fixed honeycomb (ODB) face; pole: narrow rigid pole; vehicle: "
                  "another car; moving_barrier: test trolley with deformable face (MDB); truck: heavy, "
                  "stiff, high-riding vehicle", PARTNER_TYPES)
    mass_kg: Optional[float] = F(None, "kg", "Partner mass (moving partners only). None = type default")
    speed_kph: float = F(0.0, "km/h", "Partner speed at impact along its own heading (0 for fixed barriers)")
    width_m: Optional[float] = F(None, "m", "Width of the partner's engaging face (pole diameter for poles). None = type default")
    length_m: Optional[float] = F(None, "m", "Partner length (visualisation and yaw inertia). None = type default")
    stiffness_kN_per_m: Optional[float] = F(None, "kN/m", "Elastic stiffness of the partner face before it crushes. None = type default")
    crush_force_kN: Optional[float] = F(None, "kN", "Mean crush (plateau) force of the partner face. None = type default")
    crush_depth_m: Optional[float] = F(None, "m", "Crushable depth of the partner face before it bottoms out. None = type default")
    structure_height_m: Optional[float] = F(None, "m", "Height of the partner's main load path above ground; a mismatch with "
                                            "the subject vehicle reduces structural engagement (override/underride)")
    engaged_face: str = F("front", "", "Face of a vehicle partner that engages the subject vehicle", ["front", "side", "rear"])


@dataclass
class SledPulseConfig:
    shape: str = F("fmvss213", "", "Pulse shape for collision.mode = sled. fmvss213 approximates the FMVSS 213 "
                   "child-restraint sled corridor used by harness crash-test protocols",
                   ["fmvss213", "half_sine", "trapezoid", "csv"])
    delta_v_kph: float = F(48.0, "km/h", "Velocity change of the sled pulse")
    duration_ms: float = F(80.0, "ms", "Pulse duration (half_sine, trapezoid)")
    rise_ms: float = F(20.0, "ms", "Rise time (trapezoid)")
    fall_ms: float = F(25.0, "ms", "Fall time (trapezoid)")
    csv_path: Optional[str] = F(None, "", "CSV file with columns time_s, accel_g (deceleration positive) for shape = csv")


@dataclass
class CollisionConfig:
    mode: str = F("crash", "", "crash: full vehicle-to-partner crash model; sled: prescribed acceleration pulse "
                  "(laboratory sled test)", ["crash", "sled"])
    impact_speed_kph: float = F(56.0, "km/h", "Speed of the subject vehicle (the one carrying the dog) at the moment of impact")
    impact_direction: str = F("frontal", "", "Face of the subject vehicle that is struck", IMPACT_DIRECTIONS)
    impact_angle_deg: float = F(0.0, "deg", "Obliquity: rotation of the partner (or barrier face) relative to the struck "
                                "face, positive counter-clockwise seen from above")
    overlap: float = F(1.0, "-", "Fraction (0-1] of the struck face width that is engaged by the partner")
    overlap_side: str = F("left", "", "Side of the struck face on which the overlap sits (frontal/rear impacts)",
                          ["left", "right", "center"])
    impact_location_m: Optional[float] = F(None, "m", "Side impacts: x position (vehicle frame) of the partner centre "
                                           "line; frontal/rear pole impacts: y position. None = at the dog's seating row "
                                           "(side) or vehicle centre line (frontal/rear)")
    interface_friction: float = F(0.3, "-", "Friction coefficient between the two crushing structures")
    duration_after_impact_s: float = F(0.30, "s", "Simulated time after first contact")
    partner: PartnerConfig = S(PartnerConfig, "Collision partner")
    sled: SledPulseConfig = S(SledPulseConfig, "Sled pulse (collision.mode = sled)")


# --------------------------------------------------------------------------
# Vehicle
# --------------------------------------------------------------------------

@dataclass
class CrushZoneConfig:
    name: str = F("zone", "", "Structural member / zone name")
    depth_m: float = F(0.1, "m", "Crush depth available in this zone")
    force_kN: float = F(100.0, "kN", "Mean crush force of the zone at full engagement")


def _zones(*items):
    return [CrushZoneConfig(n, d, f) for n, d, f in items]


@dataclass
class StructureConfig:
    front_stiffness_kN_per_m: float = F(1500.0, "kN/m", "Elastic stiffness of the front structure before plastic crush")
    rear_stiffness_kN_per_m: float = F(1000.0, "kN/m", "Elastic stiffness of the rear structure")
    side_stiffness_kN_per_m: float = F(700.0, "kN/m", "Elastic stiffness of the side structure")
    compartment_stiffness_kN_per_m: float = F(7000.0, "kN/m", "Stiffness of the passenger cell once all crush zones are used")
    unloading_stiffness_factor: float = F(20.0, "-", "Unloading slope as a multiple of the elastic stiffness (controls restitution)")
    structure_height_m: float = F(0.50, "m", "Height of the main longitudinal load path above ground")
    front_rail_share: float = F(0.35, "-", "Share of the front/rear structural force carried by each longitudinal rail "
                                "(the rest is spread across the bumper width)")
    rail_lateral_position: float = F(0.42, "-", "Rail position as a fraction of the half width")
    b_pillar_share: float = F(0.35, "-", "Share of the side structural force carried by the B-pillar (C-pillar gets 0.2)")


@dataclass
class RearCompartmentConfig:
    seat_bight_x_m: float = F(-0.95, "m", "x of the seat bight (cushion/seat-back junction) of the rear bench")
    cushion_height_m: float = F(0.60, "m", "Height of the top of the rear cushion above ground at the bight")
    cushion_length_m: float = F(0.50, "m", "Fore-aft length of the rear cushion")
    cushion_thickness_m: float = F(0.14, "m", "Foam thickness of the rear cushion")
    cushion_angle_deg: float = F(10.0, "deg", "Cushion angle (front edge raised)")
    seat_width_m: float = F(1.36, "m", "Width of the rear bench")
    seat_back_angle_deg: float = F(25.0, "deg", "Rear seat-back recline from vertical")
    seat_back_height_m: float = F(0.62, "m", "Rear seat-back height along its surface")
    seat_back_thickness_m: float = F(0.12, "m", "Rear seat-back thickness")
    front_seat_back_x_m: float = F(-0.15, "m", "x of the rear face of the front seat backs")
    front_seat_back_top_m: float = F(1.22, "m", "Height of the top of the front seat backs")
    front_seat_width_m: float = F(0.54, "m", "Width of each front seat")
    console_width_m: float = F(0.24, "m", "Width of the centre console between the front seats")
    floor_height_m: float = F(0.30, "m", "Height of the rear foot-well floor")
    tunnel_height_m: float = F(0.07, "m", "Height of the transmission tunnel above the floor")
    cabin_width_m: float = F(1.42, "m", "Inner width between the rear door trims")
    roof_height_m: float = F(1.36, "m", "Inner roof height above ground at the rear row")
    beltline_height_m: float = F(1.00, "m", "Height of the window sill (door trim top)")
    door_crush_space_m: float = F(0.07, "m", "Side crush absorbed before the inner door trim starts to intrude")
    c_pillar_x_m: float = F(-1.15, "m", "x of the C-pillar (outboard belt upper anchorage)")
    trunk_length_m: float = F(1.00, "m", "Length of the trunk / cargo floor behind the rear seat back")
    trunk_floor_height_m: float = F(0.62, "m", "Height of the trunk / cargo floor")
    trunk_height_m: float = F(0.45, "m", "Clear height of the trunk (to the parcel shelf/deck lid or cargo-area roof)")
    trunk_width_m: float = F(1.10, "m", "Width of the trunk / cargo floor")
    rear_crush_space_m: float = F(0.30, "m", "Rear crush absorbed before the trunk rear wall moves into the trunk")
    parcel_shelf: bool = F(True, "", "Rigid parcel shelf / bulkhead above the seat back (sedans)")


@dataclass
class SuspensionConfig:
    pitch_frequency_hz: float = F(1.4, "Hz", "Body pitch natural frequency")
    roll_frequency_hz: float = F(1.6, "Hz", "Body roll natural frequency")
    damping_ratio: float = F(0.35, "-", "Pitch/roll damping ratio")
    bump_stop_deg: float = F(5.0, "deg", "Pitch/roll angle at which the bump stops engage")


VEHICLE_PRESETS = ["midsize_sedan", "compact_hatchback", "midsize_suv", "station_wagon", "minivan", "pickup_crew_cab"]


@dataclass
class VehicleConfig:
    preset: Optional[str] = F(None, "", "Optional body preset applied before the other vehicle values", VEHICLE_PRESETS)
    body_style: str = F("sedan", "", "Body style; hatchback/suv/wagon/minivan have an open cargo area behind the rear seat",
                        ["sedan", "hatchback", "suv", "wagon", "minivan", "pickup"])
    mass_kg: float = F(1600.0, "kg", "Test mass of the subject vehicle")
    length_m: float = F(4.85, "m", "Overall length")
    width_m: float = F(1.85, "m", "Overall width")
    height_m: float = F(1.45, "m", "Overall height")
    wheelbase_m: float = F(2.83, "m", "Wheelbase")
    track_m: float = F(1.58, "m", "Track width")
    cg_height_m: float = F(0.55, "m", "Centre-of-gravity height")
    cg_to_front_axle_m: float = F(1.30, "m", "Distance from the front axle back to the CG")
    front_overhang_m: float = F(0.95, "m", "Front bumper ahead of the front axle")
    yaw_inertia_kgm2: Optional[float] = F(None, "kg m^2", "Yaw moment of inertia. None = estimated from mass and size")
    pitch_inertia_kgm2: Optional[float] = F(None, "kg m^2", "Pitch moment of inertia. None = estimated")
    roll_inertia_kgm2: Optional[float] = F(None, "kg m^2", "Roll moment of inertia. None = estimated")
    tire_cornering_stiffness_N_per_rad: float = F(90000.0, "N/rad", "Cornering stiffness per axle")
    brake_balance_front: float = F(0.65, "-", "Fraction of brake force on the front axle")
    structure: StructureConfig = S(StructureConfig, "Structural stiffness")
    crush_zones_front: list = F(_zones(("bumper_beam", 0.08, 80.0), ("crash_boxes", 0.12, 180.0),
                                       ("front_rails", 0.35, 360.0), ("engine_firewall", 0.15, 600.0)),
                                "m, kN", "Front crush zones from the bumper rearwards (depth, mean force)")
    crush_zones_rear: list = F(_zones(("bumper_beam", 0.08, 60.0), ("crash_boxes", 0.10, 130.0),
                                      ("rear_rails_trunk_floor", 0.40, 250.0), ("rear_bulkhead", 0.15, 420.0)),
                               "m, kN", "Rear crush zones from the bumper forwards")
    crush_zones_side: list = F(_zones(("door_skin", 0.06, 55.0), ("door_beam", 0.12, 150.0),
                                      ("pillar_sill", 0.15, 280.0), ("floor_crossmember_tunnel", 0.15, 420.0)),
                               "m, kN", "Side crush zones from the outer skin inwards")
    rear_compartment: RearCompartmentConfig = S(RearCompartmentConfig, "Rear-compartment geometry")
    suspension: SuspensionConfig = S(SuspensionConfig, "Pitch/roll suspension")


# --------------------------------------------------------------------------
# Rear passenger (the dog)
# --------------------------------------------------------------------------

BREED_SIZES = {
    # name: (body mass kg, withers height m)
    "toy": (3.0, 0.23),
    "small": (9.0, 0.36),
    "medium": (20.0, 0.50),
    "large": (32.0, 0.60),
    "giant": (60.0, 0.78),
    # weighted crash-test dog dummies used in harness test protocols
    "cps_25lb": (11.3, 0.40),
    "cps_45lb": (20.4, 0.50),
    "cps_75lb": (34.0, 0.60),
}

SEATING_POSITIONS = ["rear_left", "rear_center", "rear_right", "rear_footwell", "cargo_area"]
POSTURES = ["sitting", "lying_sternal", "lying_lateral", "standing"]


@dataclass
class NeuromuscularConfig:
    muscle_tone: float = F(0.35, "-", "Baseline postural muscle activation (0 = flaccid/asleep, 1 = maximal)")
    startle_reflex: bool = F(True, "", "Enable the startle / bracing reflex after the crash onset")
    reflex_delay_ms: float = F(70.0, "ms", "Latency of the startle / long-latency stretch reflex")
    reflex_activation: float = F(0.8, "-", "Muscle activation reached during the reflex (0-1)")
    stretch_reflex_delay_ms: float = F(25.0, "ms", "Latency of the short-latency (spinal) stretch reflex")
    stretch_reflex_gain: float = F(0.4, "-", "Gain of the delayed stretch reflex relative to the postural gain")
    activation_time_ms: float = F(50.0, "ms", "Muscle activation/deactivation time constant")
    ligament_stiffening: bool = F(True, "", "Exponential passive tissue torque in the last part of each joint range")


@dataclass
class RearPassengerConfig:
    breed_size: str = F("medium", "", "Body-size preset (mass, withers height); 'custom' uses mass_kg and withers_height_m",
                        list(BREED_SIZES) + ["custom"])
    mass_kg: Optional[float] = F(None, "kg", "Body mass. None = from breed_size")
    withers_height_m: Optional[float] = F(None, "m", "Height at the withers (standing). None = from breed_size or allometry")
    seating_position: str = F("rear_left", "", "Where the dog rides", SEATING_POSITIONS)
    posture: str = F("sitting", "", "Posture at the start of the event", POSTURES)
    facing_deg: Optional[float] = F(None, "deg", "Heading of the dog in the vehicle (0 = facing forward, 90 = facing the left "
                                    "door). None = posture default")
    lateral_offset_m: float = F(0.0, "m", "Extra lateral offset from the seating-position centre (+left)")
    seat_belt_used: bool = F(True, "", "Whether the harness is connected to the vehicle restraint system (via the seat belt "
                             "or an anchor). False = unrestrained dog wearing the harness")
    neuromuscular: NeuromuscularConfig = S(NeuromuscularConfig, "Muscle tone and reflexes")
    model_path: Optional[str] = F(None, "", "Override the MuJoCo dog model (defaults to musculoskeletal_dog/models/dog.xml)")


# --------------------------------------------------------------------------
# Harness (device under test)
# --------------------------------------------------------------------------

HARNESS_STYLES = ["padded_vest", "y_front", "norwegian", "step_in", "collar_only"]


@dataclass
class TetherConfig:
    length_m: Optional[float] = F(None, "m", "Free length of the tether/belt-loop strap. None = taut length at rest plus slack_m")
    slack_m: float = F(0.04, "m", "Free play beyond the taut length when length_m is None")
    elongation_pct_at_rated: float = F(12.0, "%", "Webbing elongation at its rated load (nylon ~10-20 %, polyester ~6-10 %)")
    rated_strength_kN: float = F(10.0, "kN", "Breaking strength of the tether webbing")
    connector: str = F("carabiner", "", "Hardware between tether and anchor", ["carabiner", "snap_hook", "buckle_tongue",
                                                                              "isofix_clip", "loop"])
    connector_strength_kN: float = F(12.0, "kN", "Breaking strength of the connector")
    energy_absorber_force_kN: Optional[float] = F(None, "kN", "Tear-webbing energy absorber activation force (None = none)")
    energy_absorber_stroke_m: float = F(0.0, "m", "Available energy absorber payout")
    damping_ratio: float = F(0.08, "-", "Webbing damping ratio")


@dataclass
class HarnessConfig:
    name: str = F("generic padded vest", "", "Label used in reports")
    style: str = F("padded_vest", "", "Load-path architecture. padded_vest: wide straps + padded chest plate; y_front: "
                   "Y-shaped chest strap; norwegian: horizontal strap across the front of the chest; step_in: neck loop + "
                   "girth loop; collar_only: tether to a collar (reference case)", HARNESS_STYLES)
    attachment_point: str = F("dorsal", "", "Where the tether/belt attaches on the harness", ["dorsal", "ventral"])
    strap_width_mm: float = F(38.0, "mm", "Webbing width of the body straps")
    chest_plate_area_cm2: float = F(160.0, "cm^2", "Area of the padded chest plate (0 = none)")
    padding_thickness_mm: float = F(12.0, "mm", "Foam padding under the straps/plates")
    fit_slack_mm: float = F(10.0, "mm", "Looseness of the fitted harness")
    strap_elongation_pct_at_rated: float = F(10.0, "%", "Body-strap webbing elongation at rated load")
    strap_rated_strength_kN: float = F(8.0, "kN", "Body-strap breaking strength")
    d_ring_strength_kN: float = F(12.0, "kN", "Breaking strength of the attachment ring")
    buckle_strength_kN: float = F(6.0, "kN", "Breaking strength of the body-strap buckles (side-release buckles fail first)")
    mass_kg: float = F(0.45, "kg", "Harness mass")
    rgba: list = F([0.15, 0.35, 0.75, 1.0], "", "Colour of the 3D harness")
    visual_mesh: Optional[str] = F(None, "", "Optional STL of your own harness design, attached to the harness frame for display "
                                   "(metres, dog torso frame)")
    tether: TetherConfig = S(TetherConfig, "Tether / belt-loop strap")


# --------------------------------------------------------------------------
# Restraints (vehicle side)
# --------------------------------------------------------------------------

ATTACHMENTS = ["belt_loop", "buckle_tether", "isofix_tether", "headrest_tether", "cargo_hook_tether"]


@dataclass
class BeltGeometryConfig:
    upper_anchor_m: Optional[list] = F(None, "m", "Shoulder-belt upper anchorage (D-ring) in the vehicle frame. None = at the "
                                       "C-pillar for outboard seats / roof for the centre seat")
    buckle_m: Optional[list] = F(None, "m", "Buckle position in the vehicle frame. None = inboard at the seat bight")
    lap_anchor_m: Optional[list] = F(None, "m", "Outboard lap anchor in the vehicle frame. None = outboard at the seat bight")
    isofix_spacing_m: float = F(0.28, "m", "Lateral spacing of the ISOFIX/LATCH lower anchors")
    webbing_elongation_pct_at_10kN: float = F(8.0, "%", "Vehicle belt webbing elongation at 10 kN")
    slack_mm: float = F(15.0, "mm", "Slack in the belt / tether at buckle-up")
    retractor_lock_g: float = F(0.45, "g", "Vehicle-sensitive retractor locking threshold")
    film_spool_mm: float = F(15.0, "mm", "Webbing pulled out of the retractor after locking (film-spool effect)")


@dataclass
class PretensionerConfig:
    enabled: bool = F(True, "", "Pyrotechnic pretensioner present")
    type: str = F("retractor", "", "retractor: pulls the shoulder belt; buckle: pulls the buckle (also tensions buckle tethers)",
                  ["retractor", "buckle"])
    fire_on: list = F(["frontal", "rear"], "", "Impact directions that fire the pretensioner")
    fire_threshold_g: float = F(8.0, "g", "Vehicle deceleration that triggers firing")
    fire_delay_ms: float = F(8.0, "ms", "Time from threshold crossing to firing")
    stroke_mm: float = F(80.0, "mm", "Maximum webbing retraction")
    max_force_kN: float = F(2.0, "kN", "Stall force of the pretensioner")
    speed_m_per_s: float = F(12.0, "m/s", "Retraction speed")


@dataclass
class LoadLimiterConfig:
    enabled: bool = F(True, "", "Retractor load limiter (torsion bar) present")
    force_kN: float = F(3.0, "kN", "Belt force at which webbing pays out")
    max_payout_mm: float = F(150.0, "mm", "Maximum payout")


AIRBAG_TYPES = ["curtain", "rear_side_thorax", "front_seatback", "rear_center"]


@dataclass
class AirbagConfig:
    type: str = F("curtain", "", "curtain: roof-rail window bag; rear_side_thorax: door/bolster bag; front_seatback: bag "
                  "deploying rearwards from the front seat back; rear_center: bag between rear seats", AIRBAG_TYPES)
    side: str = F("left", "", "Vehicle side (curtain/thorax) or seating position", ["left", "right", "center"])
    enabled: bool = F(True, "", "Bag fitted")
    fire_on: list = F(["left", "right"], "", "Impact directions that deploy the bag")
    fire_threshold_g: float = F(6.0, "g", "Vehicle acceleration that triggers deployment")
    fire_delay_ms: float = F(6.0, "ms", "Sensing + firing delay after the threshold")
    inflation_time_ms: float = F(25.0, "ms", "Time to full deployment")
    peak_pressure_kPa: float = F(45.0, "kPa", "Peak internal over-pressure")
    vent_time_constant_ms: float = F(80.0, "ms", "Pressure decay time constant after full deployment")
    thickness_m: float = F(0.12, "m", "Deployed thickness")


@dataclass
class SeatDesignConfig:
    cushion_stiffness_kN_per_m: float = F(25.0, "kN/m", "Cushion foam stiffness under a dog-sized contact patch")
    cushion_damping_ratio: float = F(0.6, "-", "Cushion damping ratio")
    cushion_friction: float = F(0.6, "-", "Friction between dog/harness and seat cover")
    seat_back_stiffness_kNm_per_rad: float = F(40.0, "kN m/rad", "Elastic rotational stiffness of the rear seat back")
    seat_back_forward_yield_Nm: float = F(1800.0, "N m", "Moment at which the seat back yields forward (cargo loading)")
    seat_back_rearward_yield_Nm: float = F(2500.0, "N m", "Moment at which the seat back yields rearward")
    seat_back_latch_strength_Nm: float = F(3200.0, "N m", "Forward moment at which the folding seat-back latch fails")
    seat_back_mass_kg: float = F(14.0, "kg", "Mass of the rear seat back")
    trim_stiffness: str = F("medium", "", "Interior trim softness (door panels, front seat backs)", ["soft", "medium", "hard"])


@dataclass
class HeadRestraintConfig:
    present: bool = F(True, "", "Rear head restraints fitted")
    height_above_seatback_m: float = F(0.18, "m", "Head restraint height above the seat-back top")
    width_m: float = F(0.26, "m", "Head restraint width")
    thickness_m: float = F(0.10, "m", "Head restraint thickness")
    forward_offset_m: float = F(0.03, "m", "Forward offset of the head restraint face from the seat-back face")
    softness: str = F("medium", "", "Foam softness", ["soft", "medium", "hard"])


@dataclass
class RestraintsConfig:
    attachment: str = F("belt_loop", "", "How the harness is connected when rear_passenger.seat_belt_used is true. belt_loop: "
                        "shoulder belt threaded through the harness loop; buckle_tether: tether clipped into the belt buckle; "
                        "isofix_tether: Y-tether to both ISOFIX lower anchors; headrest_tether: tether around the head "
                        "restraint posts; cargo_hook_tether: tether to the cargo-floor tie-down hooks", ATTACHMENTS)
    belt: BeltGeometryConfig = S(BeltGeometryConfig, "Belt geometry")
    pretensioner: PretensionerConfig = S(PretensionerConfig, "Pretensioner")
    load_limiter: LoadLimiterConfig = S(LoadLimiterConfig, "Load limiter")
    airbags: list = F([AirbagConfig(type="curtain", side="left"), AirbagConfig(type="curtain", side="right")],
                      "", "Airbags in the rear compartment")
    seat: SeatDesignConfig = S(SeatDesignConfig, "Rear seat design")
    head_restraints: HeadRestraintConfig = S(HeadRestraintConfig, "Rear head restraints")


# --------------------------------------------------------------------------
# Trunk / cargo
# --------------------------------------------------------------------------

@dataclass
class CargoItemConfig:
    name: str = F("suitcase", "", "Label")
    mass_kg: float = F(18.0, "kg", "Mass")
    size_m: list = F([0.70, 0.45, 0.26], "m", "Dimensions (length along x, width along y, height)")
    position_m: Optional[list] = F(None, "m", "Centre position in the vehicle frame. None = placed by the packing arrangement")
    yaw_deg: float = F(0.0, "deg", "Rotation about the vertical axis")
    friction: float = F(0.4, "-", "Friction against the trunk floor")
    construction: str = F("rigid", "", "rigid: hard case/box; soft: bag", ["rigid", "soft"])


PACKING = ["against_seatback", "stacked", "side_by_side", "loose_rear", "custom"]
CARGO_RESTRAINTS = ["none", "tie_down_straps", "cargo_net", "cargo_barrier", "locked_floor_box"]


@dataclass
class TrunkCargoConfig:
    items: list = F([CargoItemConfig()], "", "Cargo items")
    packing: str = F("against_seatback", "", "Packing arrangement for items without an explicit position", PACKING)
    restraint_method: str = F("none", "", "How the cargo is secured", CARGO_RESTRAINTS)
    strap_strength_kN: float = F(5.0, "kN", "Breaking strength of each tie-down strap")
    straps_per_item: int = F(2, "-", "Tie-down straps per item")
    strap_elongation_pct_at_rated: float = F(15.0, "%", "Tie-down strap elongation at rated load")
    net_stiffness_kN_per_m: float = F(60.0, "kN/m", "Cargo net stiffness")
    net_strength_kN: float = F(12.0, "kN", "Cargo net strength")
    barrier_stiffness_kN_per_m: float = F(600.0, "kN/m", "Cargo barrier / dog guard stiffness")
    barrier_strength_kN: float = F(35.0, "kN", "Cargo barrier strength")
    floor_box_strength_kN: float = F(20.0, "kN", "Strength of the locked floor-box fixing")


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

ROAD_CONDITIONS = {"dry": 0.85, "wet": 0.55, "snow": 0.25, "ice": 0.10, "gravel": 0.55}


@dataclass
class BrakingConfig:
    enabled: bool = F(True, "", "Driver brakes before the impact")
    deceleration_g: float = F(0.8, "g", "Requested deceleration (limited by road friction)")
    onset_before_impact_s: float = F(0.6, "s", "How long before the impact the driver starts braking")
    ramp_time_s: float = F(0.15, "s", "Brake pressure build-up time")
    abs: bool = F(True, "", "Anti-lock brakes (without ABS a demand above the friction limit locks the wheels)")
    continue_after_impact: bool = F(True, "", "Brakes stay applied during and after the impact")


@dataclass
class PreImpactMotionConfig:
    initial_yaw_rate_deg_s: float = F(0.0, "deg/s", "Yaw rate at the start of the simulation (+ = turning left)")
    sideslip_deg: float = F(0.0, "deg", "Sideslip of the velocity vector relative to the heading at impact "
                            "(e.g. 75-90 for a vehicle sliding sideways into a pole)")
    steer_angle_deg: float = F(0.0, "deg", "Evasive front-wheel steer angle (+ = left)")
    steer_onset_before_impact_s: float = F(0.5, "s", "When the evasive steering starts")
    steer_ramp_s: float = F(0.2, "s", "Steering ramp time")


@dataclass
class EnvironmentConfig:
    road_condition: str = F("dry", "", "Road surface preset; sets road_friction unless it is given explicitly",
                            list(ROAD_CONDITIONS) + ["custom"])
    road_friction: Optional[float] = F(None, "-", "Tyre-road friction coefficient. None = from road_condition")
    pre_impact_duration_s: float = F(1.0, "s", "Simulated time before the impact")
    braking: BrakingConfig = S(BrakingConfig, "Pre-impact braking")
    motion: PreImpactMotionConfig = S(PreImpactMotionConfig, "Vehicle movement before impact")
    gravity: float = F(9.81, "m/s^2", "Gravitational acceleration")


# --------------------------------------------------------------------------
# Simulation & assessment
# --------------------------------------------------------------------------

@dataclass
class RenderConfig:
    enabled: bool = F(False, "", "Render a video of the event")
    cameras: list = F(["side", "dog"], "", "Cameras to render", ["side", "dog", "top", "front", "world"])
    width: int = F(960, "px", "Video width")
    height: int = F(540, "px", "Video height")
    fps: int = F(30, "1/s", "Output frame rate")
    slow_motion: float = F(10.0, "x", "Slow-motion factor during the impact phase")
    show_skeleton: bool = F(False, "", "Show bones instead of the skin")


@dataclass
class SimulationConfig:
    timestep_s: float = F(1.0e-4, "s", "Occupant simulation time step")
    settle_time_s: float = F(0.8, "s", "Time the dog settles into its posture before the scenario starts")
    vehicle_timestep_s: float = F(5.0e-5, "s", "Vehicle crash model time step")
    record_every: int = F(1, "-", "Record every n-th occupant step")
    state_rate_hz: float = F(500.0, "Hz", "Rate at which full states are stored for replay/rendering")
    output_dir: str = F("results", "", "Output directory")
    render: RenderConfig = S(RenderConfig, "Video rendering")


@dataclass
class AssessmentConfig:
    reference_scaling: str = F("geometric", "", "How human injury assessment reference values are transferred to the dog: "
                               "geometric similitude by mass ratio, or none", ["geometric", "none"])
    max_head_excursion_m: Optional[float] = F(None, "m", "Allowed forward head excursion. None = 1.1 x withers height")
    max_lateral_excursion_m: Optional[float] = F(None, "m", "Allowed lateral torso excursion. None = 0.6 x withers height")
    hardware_safety_factor: float = F(1.5, "-", "Required margin between peak load and rated strength")
    strap_pressure_limit_kPa: float = F(250.0, "kPa", "Screening limit for strap/plate contact pressure on soft tissue")
    throat_force_limit_N: Optional[float] = F(None, "N", "Screening limit for the ventral neck (throat) strap force. "
                                              "None = 1.5 x body weight")
    overrides: dict = F({}, "", "Override any reference value by criterion key, e.g. {hic15: 1200}")


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    name: str = F("scenario", "", "Scenario name")
    description: str = F("", "", "Free text")
    collision: CollisionConfig = S(CollisionConfig, "Collision")
    vehicle: VehicleConfig = S(VehicleConfig, "Vehicle")
    rear_passenger: RearPassengerConfig = S(RearPassengerConfig, "Rear passenger (dog)")
    harness: HarnessConfig = S(HarnessConfig, "Harness under test")
    restraints: RestraintsConfig = S(RestraintsConfig, "Vehicle restraints")
    trunk_cargo: TrunkCargoConfig = S(TrunkCargoConfig, "Trunk / cargo")
    environment: EnvironmentConfig = S(EnvironmentConfig, "Environment")
    simulation: SimulationConfig = S(SimulationConfig, "Numerics and output")
    assessment: AssessmentConfig = S(AssessmentConfig, "Assessment criteria")

    # ---- convenience ---------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        data = self.to_dict()
        with open(path, "w") as f:
            if path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                yaml.safe_dump(data, f, sort_keys=False)

    def copy(self) -> "ScenarioConfig":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------
# Presets for groups
# --------------------------------------------------------------------------

def vehicle_preset(name: str) -> dict:
    """Return a vehicle dict (partial) for a named body preset."""
    sedan = {}
    presets = {
        "midsize_sedan": sedan,
        "compact_hatchback": {
            "body_style": "hatchback", "mass_kg": 1300.0, "length_m": 4.30, "width_m": 1.79, "height_m": 1.48,
            "wheelbase_m": 2.64, "cg_to_front_axle_m": 1.05, "front_overhang_m": 0.88, "cg_height_m": 0.54,
            "rear_compartment": {"seat_bight_x_m": -0.85, "front_seat_back_x_m": -0.10, "cabin_width_m": 1.38,
                                 "seat_width_m": 1.30, "trunk_length_m": 0.75, "trunk_floor_height_m": 0.68,
                                 "trunk_height_m": 0.60, "trunk_width_m": 1.00, "parcel_shelf": False,
                                 "roof_height_m": 1.40, "c_pillar_x_m": -1.10, "rear_crush_space_m": 0.25},
        },
        "midsize_suv": {
            "body_style": "suv", "mass_kg": 2000.0, "length_m": 4.75, "width_m": 1.92, "height_m": 1.72,
            "wheelbase_m": 2.85, "cg_to_front_axle_m": 1.30, "front_overhang_m": 0.92, "cg_height_m": 0.70,
            "structure": {"structure_height_m": 0.58, "front_stiffness_kN_per_m": 1800.0},
            "rear_compartment": {"seat_bight_x_m": -0.95, "cushion_height_m": 0.78, "floor_height_m": 0.45,
                                 "front_seat_back_top_m": 1.42, "roof_height_m": 1.62, "beltline_height_m": 1.20,
                                 "cabin_width_m": 1.50, "seat_width_m": 1.44, "trunk_length_m": 1.05,
                                 "trunk_floor_height_m": 0.80, "trunk_height_m": 0.80, "trunk_width_m": 1.15,
                                 "parcel_shelf": False, "c_pillar_x_m": -1.25},
        },
        "station_wagon": {
            "body_style": "wagon", "mass_kg": 1700.0, "length_m": 4.90, "height_m": 1.50,
            "rear_compartment": {"trunk_length_m": 1.10, "trunk_floor_height_m": 0.64, "trunk_height_m": 0.70,
                                 "parcel_shelf": False, "roof_height_m": 1.40},
        },
        "minivan": {
            "body_style": "minivan", "mass_kg": 2100.0, "length_m": 5.10, "width_m": 1.98, "height_m": 1.78,
            "wheelbase_m": 3.05, "cg_to_front_axle_m": 1.40, "front_overhang_m": 0.95, "cg_height_m": 0.72,
            "rear_compartment": {"seat_bight_x_m": -1.05, "cushion_height_m": 0.80, "floor_height_m": 0.42,
                                 "front_seat_back_x_m": -0.20, "front_seat_back_top_m": 1.45, "roof_height_m": 1.70,
                                 "beltline_height_m": 1.15, "cabin_width_m": 1.58, "seat_width_m": 1.50,
                                 "trunk_length_m": 0.80, "trunk_floor_height_m": 0.62, "trunk_height_m": 0.95,
                                 "parcel_shelf": False, "c_pillar_x_m": -1.35},
        },
        "pickup_crew_cab": {
            "body_style": "pickup", "mass_kg": 2400.0, "length_m": 5.80, "width_m": 2.03, "height_m": 1.90,
            "wheelbase_m": 3.70, "cg_to_front_axle_m": 1.55, "front_overhang_m": 1.00, "cg_height_m": 0.80,
            "structure": {"structure_height_m": 0.65, "front_stiffness_kN_per_m": 2000.0},
            "rear_compartment": {"seat_bight_x_m": -0.55, "cushion_height_m": 0.95, "floor_height_m": 0.60,
                                 "front_seat_back_x_m": 0.20, "front_seat_back_top_m": 1.62, "roof_height_m": 1.82,
                                 "beltline_height_m": 1.35, "cabin_width_m": 1.62, "seat_width_m": 1.55,
                                 "trunk_length_m": 0.0, "parcel_shelf": True, "c_pillar_x_m": -0.70},
        },
    }
    if name not in presets:
        raise ValueError(f"unknown vehicle preset {name!r}; choose from {list(presets)}")
    out = copy.deepcopy(presets[name])
    mass_ratio = out.get("mass_kg", 1600.0) / 1600.0
    if name != "midsize_sedan":
        # crush forces scale roughly with vehicle mass for similar crash pulses
        base = VehicleConfig()
        for face in ("front", "rear", "side"):
            zones = getattr(base, f"crush_zones_{face}")
            out[f"crush_zones_{face}"] = [
                {"name": z.name, "depth_m": z.depth_m, "force_kN": round(z.force_kN * mass_ratio, 1)} for z in zones]
    return out


# --------------------------------------------------------------------------
# dict <-> dataclass
# --------------------------------------------------------------------------

LIST_ITEM_TYPES = {
    ("VehicleConfig", "crush_zones_front"): CrushZoneConfig,
    ("VehicleConfig", "crush_zones_rear"): CrushZoneConfig,
    ("VehicleConfig", "crush_zones_side"): CrushZoneConfig,
    ("RestraintsConfig", "airbags"): AirbagConfig,
    ("TrunkCargoConfig", "items"): CargoItemConfig,
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build(cls, data: Optional[dict], path: str):
    if data is None:
        return cls()
    if dataclasses.is_dataclass(data):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise KeyError(f"{path}: unknown parameter(s) {sorted(unknown)}; valid: {sorted(names)}")
    kwargs = {}
    for name, value in data.items():
        tp = hints[name]
        origin = get_origin(tp)
        inner = [a for a in get_args(tp) if a is not type(None)] if origin is Union else []
        target = inner[0] if inner else tp
        if dataclasses.is_dataclass(target) and isinstance(value, dict):
            kwargs[name] = _build(target, value, f"{path}.{name}")
        elif (cls.__name__, name) in LIST_ITEM_TYPES and isinstance(value, list):
            item_cls = LIST_ITEM_TYPES[(cls.__name__, name)]
            kwargs[name] = [_build(item_cls, v, f"{path}.{name}[{i}]") if isinstance(v, dict) else v
                            for i, v in enumerate(value)]
        else:
            if target is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            kwargs[name] = value
    obj = cls(**kwargs)
    for name, f in names.items():
        choices = f.metadata.get("choices")
        val = getattr(obj, name)
        if choices and val is not None and not isinstance(val, list) and val not in choices:
            raise ValueError(f"{path}.{name}: {val!r} is not one of {choices}")
    return obj


def config_from_dict(data: Optional[dict]) -> ScenarioConfig:
    data = copy.deepcopy(data or {})
    veh = data.get("vehicle") or {}
    if veh.get("preset"):
        data["vehicle"] = deep_merge(vehicle_preset(veh["preset"]), veh)
    cfg = _build(ScenarioConfig, data, "scenario")
    validate(cfg)
    return cfg


def load_config(path: Union[str, Path]) -> ScenarioConfig:
    path = Path(path)
    with open(path) as f:
        data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    return config_from_dict(data)


def validate(cfg: ScenarioConfig) -> None:
    c = cfg.collision
    if not 0.0 < c.overlap <= 1.0:
        raise ValueError("collision.overlap must be in (0, 1]")
    if c.impact_speed_kph < 0 or c.partner.speed_kph < 0:
        raise ValueError("speeds must be >= 0")
    if c.mode == "crash" and c.impact_speed_kph == 0 and c.partner.speed_kph == 0:
        raise ValueError("either the subject vehicle or the partner must be moving")
    if c.mode == "crash" and c.partner.type in ("rigid_barrier", "deformable_barrier", "pole") and c.impact_speed_kph <= 0:
        raise ValueError("fixed barriers need impact_speed_kph > 0")
    if c.mode == "sled" and c.sled.shape == "csv" and not c.sled.csv_path:
        raise ValueError("collision.sled.csv_path is required for shape = csv")
    rp = cfg.rear_passenger
    if rp.breed_size == "custom" and rp.mass_kg is None:
        raise ValueError("rear_passenger.mass_kg is required for breed_size = custom")
    if rp.seating_position == "cargo_area" and cfg.vehicle.body_style == "sedan" and cfg.vehicle.rear_compartment.parcel_shelf:
        raise ValueError("a sedan with a parcel shelf has no cargo area for the dog; use a hatchback/suv/wagon body style")
    if rp.seating_position == "cargo_area" and cfg.vehicle.rear_compartment.trunk_length_m < 0.4:
        raise ValueError("the cargo area is too short for a dog")
    if cfg.simulation.timestep_s > 5e-4:
        raise ValueError("simulation.timestep_s should be <= 5e-4 s for crash loads")
    for z in cfg.vehicle.crush_zones_front + cfg.vehicle.crush_zones_rear + cfg.vehicle.crush_zones_side:
        if z.depth_m <= 0 or z.force_kN <= 0:
            raise ValueError(f"crush zone {z.name}: depth and force must be positive")
    for item in cfg.trunk_cargo.items:
        if item.mass_kg <= 0 or len(item.size_m) != 3 or min(item.size_m) <= 0:
            raise ValueError(f"cargo item {item.name}: invalid mass or size")


# --------------------------------------------------------------------------
# Resolved values
# --------------------------------------------------------------------------

def dog_mass_and_height(rp: RearPassengerConfig) -> tuple:
    """Body mass (kg) and withers height (m) after applying the breed preset."""
    mass, height = BREED_SIZES.get(rp.breed_size, (None, None))
    if rp.mass_kg is not None:
        mass = float(rp.mass_kg)
        if rp.withers_height_m is None and rp.breed_size == "custom":
            height = None
    if rp.withers_height_m is not None:
        height = float(rp.withers_height_m)
    if height is None:
        # allometric fit through typical breeds: h = 0.134 m^0.436
        height = 0.134 * mass ** 0.436
    return mass, height


def road_friction(env: EnvironmentConfig) -> float:
    if env.road_friction is not None:
        return float(env.road_friction)
    return ROAD_CONDITIONS.get(env.road_condition, 0.8)


def partner_defaults(p: PartnerConfig) -> dict:
    """Fill the type-dependent partner defaults."""
    table = {
        # mass, width, length, stiffness kN/m, crush force kN, depth m, structure height m
        "rigid_barrier": (None, 4.0, 1.5, None, None, 0.0, 0.5),
        "deformable_barrier": (None, 1.0, 1.0, 6000.0, None, 0.54, 0.5),
        "pole": (None, 0.254, 0.254, None, None, 0.0, 0.5),
        "vehicle": (1600.0, 1.85, 4.85, 1500.0, None, 0.65, 0.5),
        "moving_barrier": (1400.0, 1.70, 4.0, 2500.0, 260.0, 0.50, 0.55),
        "truck": (12000.0, 2.50, 9.0, 6000.0, 2000.0, 0.30, 0.80),
    }
    mass, width, length, k, fc, depth, h = table[p.type]
    out = {
        "mass_kg": p.mass_kg if p.mass_kg is not None else mass,
        "width_m": p.width_m if p.width_m is not None else width,
        "length_m": p.length_m if p.length_m is not None else length,
        "stiffness_kN_per_m": p.stiffness_kN_per_m if p.stiffness_kN_per_m is not None else k,
        "crush_force_kN": p.crush_force_kN if p.crush_force_kN is not None else fc,
        "crush_depth_m": p.crush_depth_m if p.crush_depth_m is not None else depth,
        "structure_height_m": p.structure_height_m if p.structure_height_m is not None else h,
    }
    return out


# --------------------------------------------------------------------------
# Documentation helpers (params table / commented template)
# --------------------------------------------------------------------------

def iter_fields(cls, prefix: str = ""):
    """Yield (dotted_name, field, default_value) recursively."""
    obj = cls()
    for f in dataclasses.fields(cls):
        val = getattr(obj, f.name)
        name = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(val):
            yield name, f, None
            yield from iter_fields(type(val), name + ".")
        else:
            yield name, f, val


def params_table() -> str:
    rows = []
    for name, f, val in iter_fields(ScenarioConfig):
        if f.default_factory is not dataclasses.MISSING and dataclasses.is_dataclass(f.default_factory()):
            rows.append(f"\n[{name}]  {f.metadata.get('help', '')}")
            continue
        unit = f.metadata.get("unit", "")
        choices = f.metadata.get("choices")
        txt = f"  {name:<58} = {val!r:<22} {('[' + unit + ']') if unit else ''}  {f.metadata.get('help', '')}"
        if choices:
            txt += f"  options: {', '.join(map(str, choices))}"
        rows.append(txt)
    return "\n".join(rows)


def _yaml_value(v) -> str:
    if dataclasses.is_dataclass(v):
        v = dataclasses.asdict(v)
    s = yaml.safe_dump(v, default_flow_style=True, sort_keys=False).strip()
    if s.endswith("..."):
        s = s[:-3].strip()
    return s


def commented_template(cfg: Optional[ScenarioConfig] = None) -> str:
    """YAML text of a scenario where every value is preceded by its documentation."""
    cfg = cfg or ScenarioConfig()
    lines = ["# Rear-seat dog harness crash scenario", "# Coordinates: vehicle frame, x forward, y left, z up, "
             "origin on the ground below the CG.", ""]

    def emit(obj, indent):
        pad = "  " * indent
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            unit = f.metadata.get("unit", "")
            help_ = f.metadata.get("help", "")
            choices = f.metadata.get("choices")
            comment = help_ + (f" [{unit}]" if unit else "")
            if choices:
                comment += f" (options: {', '.join(map(str, choices))})"
            if dataclasses.is_dataclass(val):
                lines.append(f"{pad}# {comment}" if indent == 0 else f"{pad}# {comment}")
                lines.append(f"{pad}{f.name}:")
                emit(val, indent + 1)
                if indent == 0:
                    lines.append("")
            elif isinstance(val, list) and val and dataclasses.is_dataclass(val[0]):
                lines.append(f"{pad}# {comment}")
                lines.append(f"{pad}{f.name}:")
                for item in val:
                    first = True
                    for sf in dataclasses.fields(item):
                        prefix = f"{pad}  - " if first else f"{pad}    "
                        lines.append(f"{prefix}{sf.name}: {_yaml_value(getattr(item, sf.name))}")
                        first = False
            else:
                lines.append(f"{pad}# {comment}")
                lines.append(f"{pad}{f.name}: {_yaml_value(val)}")

    emit(cfg, 0)
    return "\n".join(lines) + "\n"
