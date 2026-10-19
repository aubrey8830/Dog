import numpy as np
import pytest

from dog_crash_sim.config import KPH, config_from_dict
from dog_crash_sim.vehicle.crash_model import ElastoPlastic, SeriesInterface, run_vehicle_model, sled_pulse

NO_BRAKE = {"braking": {"enabled": False}, "pre_impact_duration_s": 0.1}


def test_elastoplastic_unloads_elastically():
    s = ElastoPlastic([(0.1, 100e3), (0.2, 200e3)], 1e7, 1e8, unload_factor=10.0)
    f1 = s.force(0.05)                      # on the first plateau
    s.commit(0.05)
    assert f1 == pytest.approx(100e3 + 2e4 * 0.05, rel=1e-3)
    assert s.force(0.15) == pytest.approx(200e3 + 2e4 * 0.15, rel=1e-3)
    # unloading is steep and reaches zero after a small spring-back
    assert s.force(0.0499) < f1
    assert s.force(0.04) == 0.0


def test_series_interface_force_balance():
    a = ElastoPlastic([(0.3, 200e3)], 2e6, 1e7)
    b = ElastoPlastic([(0.3, 100e3)], 2e6, 1e7)
    it = SeriesInterface(a, b)
    f = it.solve(0.2)
    assert it.ds + it.dp == pytest.approx(0.2, abs=1e-6)
    # the weaker structure crushes first
    assert it.dp > it.ds
    assert a.force(it.ds) == pytest.approx(f, rel=0.02)


def test_full_frontal_rigid_barrier_pulse_is_plausible():
    cfg = config_from_dict({"collision": {"impact_speed_kph": 56.0}, "environment": NO_BRAKE})
    vm = run_vehicle_model(cfg)
    i = vm.info
    assert i["impact_speed_kph"] == pytest.approx(56.0, abs=0.3)
    assert 56.0 <= i["delta_v_kph"] <= 56.0 * 1.3          # some restitution
    assert 20.0 < i["peak_accel_g"] < 60.0
    assert 0.3 < i["max_crush_subject_m"] < 1.0
    assert abs(vm.euler[:, 2]).max() < 1e-6                # symmetric impact, no yaw


def test_braking_hits_target_impact_speed():
    cfg = config_from_dict({"collision": {"impact_speed_kph": 40.0},
                            "environment": {"braking": {"enabled": True, "onset_before_impact_s": 0.8},
                                            "pre_impact_duration_s": 1.0}})
    vm = run_vehicle_model(cfg)
    assert vm.info["impact_speed_kph"] == pytest.approx(40.0, abs=0.3)
    assert vm.info["initial_speed_kph"] > 55.0              # 0.8 g for ~0.7 s removes ~20 km/h


def test_equal_cars_head_on_share_delta_v():
    cfg = config_from_dict({"collision": {"impact_speed_kph": 50.0, "partner": {"type": "vehicle", "speed_kph": 50.0,
                                                                                 "mass_kg": 1600.0}},
                            "environment": NO_BRAKE})
    vm = run_vehicle_model(cfg)
    # momentum: common velocity zero, so the subject loses at least its 50 km/h
    assert 50.0 <= vm.info["delta_v_kph"] < 70.0


def test_offset_impact_yaws_the_vehicle():
    cfg = config_from_dict({"collision": {"impact_speed_kph": 64.0, "overlap": 0.4,
                                          "partner": {"type": "deformable_barrier"}}, "environment": NO_BRAKE})
    vm = run_vehicle_model(cfg)
    assert abs(np.degrees(vm.euler[-1, 2])) > 3.0


def test_side_impact_produces_door_intrusion():
    cfg = config_from_dict({"collision": {"impact_speed_kph": 0.0, "impact_direction": "left",
                                          "partner": {"type": "moving_barrier", "speed_kph": 60.0}},
                            "environment": NO_BRAKE})
    vm = run_vehicle_model(cfg)
    assert vm.intrusion_left.max() > 0.05
    assert vm.intrusion_right.max() == 0.0
    assert vm.info["delta_v_y_kph"] < -15.0                 # pushed to the right


@pytest.mark.parametrize("shape", ["fmvss213", "half_sine", "trapezoid"])
def test_sled_pulse_delta_v(shape):
    cfg = config_from_dict({"collision": {"mode": "sled", "sled": {"shape": shape, "delta_v_kph": 48.0}},
                            "environment": NO_BRAKE})
    f, T = sled_pulse(cfg)
    ts = np.linspace(0, T, 20001)
    dv = np.trapezoid([f(t) for t in ts], ts) / KPH
    assert dv == pytest.approx(48.0, rel=0.01)
    vm = run_vehicle_model(cfg)
    assert vm.info["delta_v_kph"] == pytest.approx(48.0, rel=0.03)
