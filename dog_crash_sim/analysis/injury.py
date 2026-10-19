"""Injury and harness-performance assessment.

There are no validated injury criteria for dogs.  This module therefore
combines three kinds of criteria and labels each with its basis:

* ``scaled-human``: human injury assessment reference values (Hybrid III
  50th percentile male, FMVSS 208 / NHTSA Nij) transferred to the dog by
  geometric similitude with the mass ratio ``lambda = (m_dog / m_ref)^(1/3)``
  (force ~ lambda^2, moment ~ lambda^3, acceleration ~ 1/lambda,
  HIC ~ lambda^-1.5, compression ratio and VC unchanged).  These are
  indicative only.
* ``hardware``: peak loads compared with the rated strengths of the
  tether, connector, attachment ring and buckles (with a safety factor).
* ``kinematic`` / ``screening``: excursions, staying on the seat, harness
  migration and rotation, throat loading and strap pressure.  The pressure
  and throat values are screening thresholds meant for comparing designs
  and can be overridden in ``assessment``.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import G
from ..scene.builder import DOG_PARTS, VEH_PARTS
from .signals import cfc_filter, clip_3ms, hic, peak

HUMAN = {
    "mass": 78.0, "head_mass": 4.54,
    "hic15": 700.0, "head_3ms_g": 80.0, "chest_3ms_g": 60.0,
    "neck_tension_N": 4170.0, "neck_compression_N": 4000.0,
    "nij_tension_N": 6806.0, "nij_compression_N": 6160.0, "nij_flexion_Nm": 310.0, "nij_extension_Nm": 135.0,
    "lumbar_compression_N": 6670.0, "chest_compression": 0.30, "vc_m_s": 1.0, "thorax_stiffness_N_m": 70e3,
}

STATUS_ORDER = ["good", "acceptable", "marginal", "poor"]


def status_of(ratio: float) -> str:
    if ratio < 0.6:
        return "good"
    if ratio < 0.85:
        return "acceptable"
    if ratio < 1.0:
        return "marginal"
    return "poor"


def _crit(key, label, value, unit, limit, basis, region, higher_is_worse=True, note=""):
    ratio = (value / limit) if (limit and limit > 0) else 0.0
    return {"key": key, "label": label, "value": float(value), "unit": unit, "limit": float(limit) if limit else None,
            "ratio": float(ratio), "status": status_of(ratio), "basis": basis, "region": region, "note": note}


def risk_ais3_hic(hic_equiv: float) -> float:
    """NHTSA lognormal head risk curve (AIS3+), human-equivalent HIC15."""
    if hic_equiv <= 0:
        return 0.0
    z = (math.log(hic_equiv) - 7.45231) / 0.73998
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def risk_ais3_nij(nij: float) -> float:
    return 1.0 / (1.0 + math.exp(3.2269 - 1.9688 * nij))


def risk_ais3_chest_deflection(d_mm_equiv: float) -> float:
    return 1.0 / (1.0 + math.exp(10.5456 - 1.568 * max(d_mm_equiv, 0.0) ** 0.4612))


def assess(res) -> dict:
    cfg = res.cfg
    sc = res.scene
    C = res.channels
    t = res.t
    if len(t) < 10:
        return {"criteria": [], "summary": {"overall": "incomplete"}, "events": res.events}
    dt = float(np.median(np.diff(t)))
    A = cfg.assessment
    m = sc.model
    mass = sc.scaling.body_mass
    head_mass = float(m.body_subtreemass[m.body("skull").id])
    geo = A.reference_scaling == "geometric"
    lam = (mass / HUMAN["mass"]) ** (1 / 3) if geo else 1.0
    lam_h = (head_mass / HUMAN["head_mass"]) ** (1 / 3) if geo else 1.0
    ov = A.overrides or {}

    def ref(key, value):
        return float(ov.get(key, value))

    crit = []
    post = t >= -0.05  # crash-relevant window (keep a little pre-impact)
    # ---- head -----------------------------------------------------------
    ah = np.linalg.norm(cfc_filter(C["head_acc"], dt, 1000.0), axis=1) / G
    ah_post = ah[post]
    tp = t[post]
    hic15, t1, t2 = hic(tp, ah_post, 0.015)
    lim_hic = ref("hic15", HUMAN["hic15"] * lam_h ** -1.5)
    crit.append(_crit("hic15", "Head Injury Criterion (HIC15)", hic15, "-", lim_hic, "scaled-human", "head",
                      note=f"t1={t1 * 1e3:.0f} ms, t2={t2 * 1e3:.0f} ms"))
    h3 = clip_3ms(tp, ah_post)
    crit.append(_crit("head_3ms_g", "Head resultant acceleration, 3 ms clip", h3, "g",
                      ref("head_3ms_g", HUMAN["head_3ms_g"] / lam_h), "scaled-human", "head"))
    # ---- chest ----------------------------------------------------------
    ac = np.linalg.norm(cfc_filter(C["chest_acc"], dt, 180.0), axis=1) / G
    c3 = clip_3ms(tp, ac[post])
    crit.append(_crit("chest_3ms_g", "Thorax resultant acceleration, 3 ms clip", c3, "g",
                      ref("chest_3ms_g", HUMAN["chest_3ms_g"] / lam), "scaled-human", "thorax"))
    # thoracic load from the harness and from hard contacts
    hf = C["harness_force"]
    harness_load = np.linalg.norm(hf[:, :3], axis=1) if hf.size else np.zeros(len(t))
    cf = C["contact_force"]
    thor = DOG_PARTS.index("thorax")
    hard = [VEH_PARTS.index(p) for p in ("front_seat_backs", "door_left", "door_right", "window_left", "window_right",
                                          "airbag", "cargo", "console", "cargo_net_barrier", "pillars_roof")]
    thorax_contact = cf[:, thor, hard].sum(axis=1)
    chest_load = harness_load + C["prosternum_strap_force"] + thorax_contact
    chest_load_f = cfc_filter(chest_load, dt, 180.0)
    k_th = HUMAN["thorax_stiffness_N_m"] * lam
    depth = sc.scaling.chest_depth
    defl = np.maximum(chest_load_f, 0.0) / k_th
    Cc = defl / depth
    V = np.gradient(cfc_filter(defl, dt, 180.0), t)
    VC = np.maximum(V, 0.0) * Cc
    crit.append(_crit("chest_compression", "Estimated thoracic compression (load/stiffness/depth)", Cc[post].max(), "-",
                      ref("chest_compression", HUMAN["chest_compression"]), "scaled-human", "thorax"))
    crit.append(_crit("viscous_criterion", "Estimated viscous criterion VC", VC[post].max(), "m/s",
                      ref("viscous_criterion", HUMAN["vc_m_s"]), "scaled-human", "thorax"))
    # ---- neck -----------------------------------------------------------
    for lc, region in (("upper_neck", "neck"), ("lower_neck", "neck")):
        Fz = -cfc_filter(C[lc + "_force"][:, 2], dt, 1000.0)    # + tension
        My = cfc_filter(C[lc + "_torque"][:, 1], dt, 600.0)
        lab = "Upper neck (C1-skull)" if lc == "upper_neck" else "Lower neck (C7-T1)"
        crit.append(_crit(f"{lc}_tension", f"{lab} tension", max(Fz[post].max(), 0.0), "N",
                          ref(f"{lc}_tension", HUMAN["neck_tension_N"] * lam ** 2), "scaled-human", region))
        crit.append(_crit(f"{lc}_compression", f"{lab} compression", max(-Fz[post].min(), 0.0), "N",
                          ref(f"{lc}_compression", HUMAN["neck_compression_N"] * lam ** 2), "scaled-human", region))
        if lc == "upper_neck":
            Ft = np.where(Fz > 0, Fz / (HUMAN["nij_tension_N"] * lam ** 2), -Fz / (HUMAN["nij_compression_N"] * lam ** 2))
            Mm = np.where(My > 0, My / (HUMAN["nij_flexion_Nm"] * lam ** 3), -My / (HUMAN["nij_extension_Nm"] * lam ** 3))
            nij = (np.abs(Ft) + np.abs(Mm))[post].max()
            crit.append(_crit("nij", "Neck injury criterion Nij (scaled intercepts)", nij, "-", ref("nij", 1.0),
                              "scaled-human", "neck"))
    # ---- spine --------------------------------------------------------------
    # force on the hindquarters from the thorax along the spine (+x cranial): compression is negative
    Fx = cfc_filter(C["thoracolumbar_force"][:, 0], dt, 600.0)
    crit.append(_crit("lumbar_compression", "Thoracolumbar axial compression", max(-Fx[post].min(), 0.0), "N",
                      ref("lumbar_compression", HUMAN["lumbar_compression_N"] * lam ** 2), "scaled-human", "spine"))
    # ---- joints -------------------------------------------------------------
    over = C["joint_overrange_deg"]
    for gi, g in enumerate(("neck", "spine", "forelimb", "hindlimb")):
        crit.append(_crit(f"overrange_{g}", f"Joint motion beyond physiological range ({g})", over[post, gi].max(),
                          "deg", ref(f"overrange_{g}", 10.0), "kinematic", "joints",
                          note="ligament/capsule loading; 10 deg beyond the limit treated as injurious"))
    # ---- harness: throat and pressure ----------------------------------------
    W = mass * cfg.environment.gravity
    throat_lim = A.throat_force_limit_N if A.throat_force_limit_N else 1.5 * W
    throat = cfc_filter(C["throat_force"], dt, 180.0)
    if cfg.harness.style == "collar_only":
        # the whole tether load goes through the collar onto the throat/neck
        ring_T = C["tendon_tension"].max(axis=1) if C["tendon_tension"].size else np.zeros(len(t))
        throat = np.maximum(throat, ring_T)
    crit.append(_crit("throat_force", "Ventral neck (throat) strap load", throat[post].max(), "N", ref("throat_force", throat_lim),
                      "screening", "neck", note="tracheal / carotid loading"))
    h = cfg.harness
    w = h.strap_width_mm * 1e-3
    pressures = {}
    # belt-on-drum (capstan) pressure under a strap of tension T wrapped over a body radius R: p = T / (w R)
    for j, nm in enumerate(C.get("tendon_names", [])):
        ti = next(x for x in sc.tendons if x.name == nm)
        if ti.kind == "strap":
            pressures[nm.replace("_", " ")] = cfc_filter(C["tendon_tension"][:, j], dt, 180.0) / (w * ti.radius)
    if cfg.harness.style == "collar_only":
        pressures["collar"] = throat / (2.0 * w * sc.scaling.neck_radius)
    if hf.size:
        Fx_rear = cfc_filter(np.maximum(-hf[:, 0], 0.0), dt, 180.0)
        if h.style == "padded_vest" and h.chest_plate_area_cm2 > 0:
            area = h.chest_plate_area_cm2 * 1e-4
            pressures["chest plate"] = Fx_rear / area
        elif h.style == "y_front":
            pressures["Y strap"] = Fx_rear / (w * 0.16 * sc.scaling.length_scale)
        Flat = cfc_filter(np.linalg.norm(hf[:, 1:3], axis=1), dt, 180.0)
        pressures["girth strap"] = Flat / (w * 0.5 * res.scene.harness.perimeter_girth)
    pmax_name, pmax = "", 0.0
    for nm, p in pressures.items():
        v = float(np.max(p[post])) / 1e3
        if v > pmax:
            pmax_name, pmax = nm, v
    crit.append(_crit("strap_pressure", f"Peak strap/plate contact pressure ({pmax_name or 'n/a'})", pmax, "kPa",
                      ref("strap_pressure", A.strap_pressure_limit_kPa), "screening", "soft tissue"))
    # ---- hardware ---------------------------------------------------------------
    SF = A.hardware_safety_factor
    names = C.get("tendon_names", [])
    TT = C["tendon_tension"]
    for j, nm in enumerate(names):
        ti = next(x for x in sc.tendons if x.name == nm)
        if ti.kind not in ("tether", "belt", "strap", "cargo_strap"):
            continue
        Tpk = float(TT[post, j].max()) if TT.size else 0.0
        if ti.kind in ("tether", "belt"):
            strength = min(ti.strength, ti.connector_strength, h.d_ring_strength_kN * 1e3)
            label = {"tether": "Tether / connector / ring load", "belt": "Belt load through the harness loop"}[ti.kind]
            crit.append(_crit(f"{nm}_load", label, Tpk, "N", strength / SF, "hardware", "hardware",
                              note=f"rated {strength / 1e3:.1f} kN, safety factor {SF}"))
        elif ti.kind == "strap":
            strength = min(ti.strength, ti.connector_strength)
            crit.append(_crit(f"{nm}_load", f"Body strap load ({nm.replace('_', ' ')})", Tpk, "N", strength / SF,
                              "hardware", "hardware", note=f"buckle {h.buckle_strength_kN:.1f} kN"))
    if hf.size:
        Fb = float(harness_load[post].max())
        crit.append(_crit("harness_body_load", "Harness frame load into the thorax", Fb, "N",
                          min(h.strap_rated_strength_kN, h.buckle_strength_kN * 2) * 1e3 / SF, "hardware", "hardware"))
    # ---- kinematics --------------------------------------------------------------
    lay = sc.layout
    rc = cfg.vehicle.rear_compartment
    imp = cfg.collision.impact_direction
    axis = {"frontal": np.array([1.0, 0, 0]), "rear": np.array([-1.0, 0, 0]), "left": np.array([0, 1.0, 0]),
            "right": np.array([0, -1.0, 0])}[imp]
    h0 = C["head_pos"][0]
    exc = (C["head_pos"] - h0) @ axis
    head_r = 0.06 * sc.scaling.length_scale
    if A.max_head_excursion_m is not None:
        lim_exc = A.max_head_excursion_m
    elif imp == "frontal" and lay.support != "cargo_floor":
        lim_exc = max(rc.front_seat_back_x_m - h0[0] - head_r, 0.05)
    elif imp in ("left", "right"):
        lim_exc = max(0.5 * rc.cabin_width_m - abs(h0[1]) - head_r, 0.05)
    else:
        lim_exc = 1.1 * sc.scaling.withers_height
    crit.append(_crit("head_excursion", f"Head excursion towards the {('struck face' if imp != 'frontal' else 'front')}",
                      max(exc.max(), 0.0), "m", lim_exc, "kinematic", "kinematics",
                      note="limit = clearance to the nearest interior surface" if A.max_head_excursion_m is None else ""))
    lat = np.abs(C["torso_pos"][:, 1] - C["torso_pos"][0, 1])
    lim_lat = A.max_lateral_excursion_m or 0.6 * sc.scaling.withers_height
    crit.append(_crit("lateral_excursion", "Lateral torso excursion", lat.max(), "m", lim_lat, "kinematic", "kinematics"))
    rot = np.abs(C["torso_rot"])
    crit.append(_crit("torso_rotation", "Torso rotation (max of roll/pitch/yaw)", rot.max(), "deg", ref("torso_rotation", 90.0),
                      "kinematic", "kinematics", note="CPS-style: the dog should not rotate/flip over"))
    if C["harness_q"].size and h.style != "collar_only":
        hq = C["harness_q"]
        b = sc.bushing
        shift = float(np.abs(hq[:, 0]).max())
        lim_shift = max(abs(b["x_range"][0]), b["x_range"][1])
        crit.append(_crit("harness_shift", "Harness shift along the body", shift * 1e3, "mm", lim_shift * 1e3 * 1.05,
                          "kinematic", "harness", note="at the limit the harness rides up to the neck or back to the waist"))
        roll = float(np.degrees(np.abs(hq[:, 3]).max()))
        crit.append(_crit("harness_rotation", "Harness rotation around the chest", roll, "deg",
                          math.degrees(b["roll_range"]) * 1.02, "kinematic", "harness",
                          note="large rotation lets the dog roll out of the harness"))
    # staying on the seat
    tor_end = C["torso_pos"][-1]
    if lay.support == "cushion":
        on = (lay.bight[0] - 0.15 <= tor_end[0] <= lay.cushion_front[0] + 0.10) and \
             (abs(tor_end[1]) <= 0.5 * rc.seat_width_m + 0.05) and (tor_end[2] >= lay.cushion_z(tor_end[0]) - 0.02)
    elif lay.support == "floor":
        on = abs(tor_end[0] - lay.dog_ref[0]) < 0.6 and tor_end[2] > rc.floor_height_m - 0.05
    else:
        on = lay.trunk_rear_x - 0.05 <= tor_end[0] <= lay.trunk_front_x + 0.2
    # ---- impacts ---------------------------------------------------------------------
    head_i = DOG_PARTS.index("head")
    support = [VEH_PARTS.index("rear_cushion")] if lay.support == "cushion" else [VEH_PARTS.index("floor"), VEH_PARTS.index("trunk")]
    contact_peaks = {}
    for di, dp in enumerate(DOG_PARTS):
        for vi, vp in enumerate(VEH_PARTS):
            if vi in support:
                continue
            pk = float(cf[post, di, vi].max())
            if pk > 20.0:
                contact_peaks[f"{dp} - {vp}"] = pk
    head_contacts = {k: v for k, v in contact_peaks.items() if k.startswith("head -")}
    dog_cargo = float(C["dog_cargo_force"][post].max()) if C["dog_cargo_force"].size else 0.0
    # ---- events / failures ---------------------------------------------------------------
    ev = dict(res.events)
    failures = [k for k, v in ev.items() if k.endswith("_failure_s") and v is not None]
    flags = []
    if not on:
        flags.append("dog did not remain on its seat/support")
    if failures:
        flags += [f"{k[:-10].replace('_', ' ')} failed at {ev[k] * 1e3:.0f} ms" for k in failures]
    if ev.get("numerical_instability_s") is not None:
        flags.append("simulation became numerically unstable - results after that time are invalid")
    if dog_cargo > 0.5 * W:
        flags.append(f"cargo struck the dog (peak {dog_cargo:.0f} N)")
    sb_fwd = float(np.degrees(C["seat_back_angle"].max()))
    if sb_fwd > 20.0:
        flags.append(f"seat back folded forward by {sb_fwd:.0f} deg")
    for c in crit:
        if c["status"] == "poor" and c["basis"] == "hardware":
            flags.append(f"{c['label']} exceeds the rated strength / safety factor")
    # ---- risk estimates (scaled human curves) ---------------------------------------------------
    risk = {
        "head_AIS3+": risk_ais3_hic(hic15 * lam_h ** 1.5),
        "neck_AIS3+": risk_ais3_nij(next(c["value"] for c in crit if c["key"] == "nij")),
        "chest_AIS3+": risk_ais3_chest_deflection(float(Cc[post].max()) * 230.0),
    }
    # ---- summary -----------------------------------------------------------------------------
    worst = max(crit, key=lambda c: (STATUS_ORDER.index(c["status"]), c["ratio"]))
    statuses = [c["status"] for c in crit]
    if flags and any(("failed" in f) or ("remain" in f) or ("unstable" in f) or ("exceeds" in f) for f in flags):
        overall = "FAIL"
    elif "poor" in statuses:
        overall = "FAIL"
    elif "marginal" in statuses or flags:
        overall = "MARGINAL"
    else:
        overall = "PASS"
    region_status = {}
    for c in crit:
        r = c["region"]
        if r not in region_status or STATUS_ORDER.index(c["status"]) > STATUS_ORDER.index(region_status[r]):
            region_status[r] = c["status"]
    peaks = {
        "head_resultant_g": float(ah[post].max()), "chest_resultant_g": float(ac[post].max()),
        "tether_or_belt_N": float(TT[post].max()) if TT.size else 0.0,
        "harness_load_N": float(harness_load[post].max()), "throat_force_N": float(throat[post].max()),
        "head_excursion_m": float(max(exc.max(), 0.0)), "torso_rotation_deg": float(rot.max()),
        "seat_back_rotation_deg": sb_fwd, "dog_cargo_contact_N": dog_cargo,
        "vehicle_peak_g": float(res.motion.info.get("peak_accel_g", 0.0)),
        "vehicle_delta_v_kph": float(res.motion.info.get("delta_v_kph", 0.0)),
    }
    return {
        "criteria": crit, "risk": risk, "flags": flags, "events": ev, "remained_on_seat": bool(on),
        "contact_peaks_N": dict(sorted(contact_peaks.items(), key=lambda kv: -kv[1])),
        "head_contacts_N": head_contacts, "pressures_kPa": {k: float(np.max(v[post]) / 1e3) for k, v in pressures.items()},
        "region_status": region_status, "peaks": peaks,
        "scaling": {"lambda_body": lam, "lambda_head": lam_h, "dog_head_mass_kg": head_mass},
        "summary": {"overall": overall, "worst_criterion": worst["label"], "worst_ratio": worst["ratio"]},
    }
