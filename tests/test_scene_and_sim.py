import mujoco
import numpy as np
import pytest

from dog_crash_sim.config import config_from_dict, dog_mass_and_height
from dog_crash_sim.scene.builder import build_scene

SHORT = {"collision": {"mode": "sled", "duration_after_impact_s": 0.09},
         "environment": {"braking": {"enabled": False}, "pre_impact_duration_s": 0.02},
         "simulation": {"settle_time_s": 0.25}}


def _penetration(sc):
    m = sc.model
    d = mujoco.MjData(m)
    d.qpos[sc.dog_qpos_slice] = sc.dog_init_qpos
    mujoco.mj_forward(m, d)
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        dog1 = sc.geom_dog_part[c.geom1] >= 0
        dog2 = sc.geom_dog_part[c.geom2] >= 0
        if dog1 != dog2 and sc.model.geom(c.geom1).name[:3] != "Ca_" and sc.model.geom(c.geom2).name[:3] != "Ca_":
            worst = min(worst, c.dist)
    return worst


@pytest.mark.parametrize("over", [
    {},
    {"harness": {"style": "y_front"}},
    {"harness": {"style": "norwegian"}},
    {"harness": {"style": "step_in", "attachment_point": "ventral"}},
    {"harness": {"style": "collar_only"}},
    {"restraints": {"attachment": "buckle_tether"}},
    {"restraints": {"attachment": "isofix_tether"}},
    {"restraints": {"attachment": "headrest_tether"}},
    {"rear_passenger": {"seating_position": "rear_center", "posture": "lying_sternal"}},
    {"rear_passenger": {"seating_position": "rear_right", "posture": "standing"}},
    {"rear_passenger": {"seating_position": "rear_footwell", "posture": "lying_sternal", "seat_belt_used": False}},
    {"vehicle": {"preset": "midsize_suv"}, "rear_passenger": {"seating_position": "cargo_area",
                                                              "posture": "lying_sternal"},
     "trunk_cargo": {"items": []}},
    {"rear_passenger": {"breed_size": "large", "seating_position": "rear_center"}},
    {"vehicle": {"preset": "station_wagon"}, "rear_passenger": {"breed_size": "large", "seating_position": "cargo_area",
                                                                "posture": "lying_sternal"}, "trunk_cargo": {"items": []}},
    {"rear_passenger": {"breed_size": "toy", "posture": "lying_lateral"}},
])
def test_scene_builds_without_initial_interpenetration(over):
    cfg = config_from_dict(over)
    sc = build_scene(cfg)
    mass, _ = dog_mass_and_height(cfg.rear_passenger)
    dog_mass = sc.model.body_subtreemass[sc.model.body("torso").id] - (cfg.harness.mass_kg if cfg.harness.style != "collar_only" else 0)
    assert dog_mass == pytest.approx(mass, rel=1e-3)
    assert _penetration(sc) > -0.005


def test_dog_that_cannot_fit_is_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        build_scene(config_from_dict({"rear_passenger": {"breed_size": "giant", "seating_position": "rear_center"}}))


@pytest.mark.parametrize("method", ["tie_down_straps", "cargo_net", "cargo_barrier", "locked_floor_box"])
def test_cargo_restraints_build(method):
    sc = build_scene(config_from_dict({"vehicle": {"preset": "station_wagon"},
                                       "trunk_cargo": {"restraint_method": method, "items": [
                                           {"name": "a", "mass_kg": 10}, {"name": "b", "mass_kg": 5,
                                                                          "size_m": [0.3, 0.3, 0.3]}]}}))
    assert len(sc.cargo) == 2


def test_airbags_start_hidden_and_inactive():
    sc = build_scene(config_from_dict({"restraints": {"airbags": [
        {"type": "curtain", "side": "left"}, {"type": "rear_side_thorax", "side": "left"},
        {"type": "front_seatback", "side": "left", "fire_on": ["frontal"]}, {"type": "rear_center", "side": "center"}]}}))
    assert len(sc.airbags) == 4
    for ab in sc.airbags:
        assert sc.model.geom_contype[ab["gid"]] == 0
        assert sc.model.geom_rgba[ab["gid"], 3] == 0.0


def test_short_sled_simulation_and_roundtrip(tmp_path):
    from dog_crash_sim.results_io import load_results, save_results
    from dog_crash_sim.simulation import run_scenario
    cfg = config_from_dict(dict(SHORT, restraints={"attachment": "isofix_tether", "airbags": []}))
    res = run_scenario(cfg)
    assert "numerical_instability_s" not in res.events
    names = res.channels["tendon_names"]
    T = res.channels["tendon_tension"][:, names.index("tether")]
    mass, _ = dog_mass_and_height(cfg.rear_passenger)
    assert T.max() > 0.5 * mass * 9.81            # the tether takes the crash load
    assert res.channels["harness_q"].shape[1] == 6
    keys = {c["key"] for c in res.metrics["criteria"]}
    assert {"hic15", "nij", "chest_3ms_g", "tether_load", "ring_load", "head_excursion"} <= keys
    assert res.metrics["summary"]["overall"] in ("PASS", "MARGINAL", "FAIL")
    save_results(res, tmp_path / "run")
    back = load_results(tmp_path / "run")
    assert np.allclose(back.t, res.t)
    assert back.metrics["summary"]["overall"] == res.metrics["summary"]["overall"]
    assert back.states_qpos.shape == res.states_qpos.shape


def test_unrestrained_dog_moves_further_than_restrained():
    from dog_crash_sim.simulation import run_scenario
    exc = {}
    for used in (True, False):
        cfg = config_from_dict(dict(SHORT, rear_passenger={"seat_belt_used": used}, restraints={"airbags": []},
                                    collision={"mode": "sled", "duration_after_impact_s": 0.11}))
        res = run_scenario(cfg)
        exc[used] = (res.channels["torso_pos"][:, 0] - res.channels["torso_pos"][0, 0]).max()
    assert exc[False] > exc[True] + 0.05
