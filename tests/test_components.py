import math

import numpy as np
import pytest

from dog_crash_sim.analysis.signals import cfc_filter, clip_3ms, hic
from dog_crash_sim.biophysics.dog_model import ANATOMICAL_FRACTIONS, load_dog_spec, scale_dog_spec, segment_of
from dog_crash_sim.restraints.models import EnergyAbsorber, Retractor, SeatBack, Webbing


def test_hic_of_constant_acceleration():
    t = np.arange(0, 0.05, 1e-4)
    a = np.full_like(t, 60.0)
    val, t1, t2 = hic(t, a, 0.015)
    assert val == pytest.approx(0.015 * 60.0 ** 2.5, rel=0.01)
    assert t2 - t1 == pytest.approx(0.015, abs=2e-4)


def test_clip_and_filter():
    t = np.arange(0, 0.1, 1e-4)
    tri = np.interp(t, [0, 0.05, 0.1], [0, 100, 0])
    # a 3 ms window centred on the peak of a 100 g/50 ms ramp stays above 97 g
    assert clip_3ms(t, tri) == pytest.approx(97.0, abs=0.5)
    noisy = 10.0 + np.sin(2 * math.pi * 3000 * t)
    f = cfc_filter(noisy, 1e-4, 60.0)
    assert np.abs(f[100:-100] - 10.0).max() < 0.05


def test_webbing_is_tension_only_with_hysteresis_and_breaks():
    w = Webbing("w", 1.0, 100e3, strength=5e3, damping_ratio=0.0)
    assert w.force(0.99, 0.0, 0.0) == 0.0
    T1 = w.force(1.02, 0.0, 0.0)
    assert T1 == pytest.approx(2e3)
    assert w.force(1.01, 0.0, 0.0) < 1e3          # unloading is steeper than loading
    assert w.force(1.06, 0.0, 0.1) == 0.0         # 6 kN > 5 kN strength
    assert w.broken and w.broken_at == 0.1


def test_energy_absorber_caps_tension():
    w = Webbing("w", 1.0, 100e3, damping_ratio=0.0)
    ea = EnergyAbsorber(force=2e3, stroke=0.1)
    T = ea.limit(w, 1.05, 0.0, 0.0)
    assert T == pytest.approx(2e3, rel=1e-3)
    assert ea.payout > 0.02


def test_retractor_lock_pretensioner_and_load_limiter():
    w = Webbing("belt", 1.0, 125e3, damping_ratio=0.0)
    r = Retractor(lock_g=0.45, film_spool=0.0, pretensioner_stroke=0.05, pretensioner_force=2e3,
                  pretensioner_fire_time=0.01, load_limit=3e3, load_limit_stroke=0.2)
    assert r.update(w, 1.0, 0.0, 0.0, 0.1, 1e-4) == r.rewind_tension   # unlocked
    r.update(w, 1.0, 0.0, 0.0, 1.0, 1e-4)                              # locks
    assert r.locked
    for k in range(200):
        r.update(w, 1.0, 0.0, 0.01 + k * 1e-4, 10.0, 1e-4)
    assert 0.0 < r.retracted <= 0.05
    T = r.update(w, 1.2, 0.0, 0.05, 10.0, 1e-4)
    assert T == pytest.approx(3e3, rel=1e-3)
    assert r.paid_out > 0.0


def test_seat_back_yields_and_latch_breaks():
    sb = SeatBack(stiffness=40e3, forward_yield=1800.0, rearward_yield=2500.0, latch_strength=3200.0, damping=0.0)
    assert sb.update(0.01, 0.0, 0.0) == pytest.approx(-400.0)
    sb.update(0.06, 0.0, 0.0)                      # beyond the forward yield moment
    assert sb.theta_p > 0.0
    sb.update(0.2, 0.0, 0.1)                       # demand beyond the latch strength
    assert sb.latch_broken


@pytest.mark.parametrize("mass,height", [(3.0, 0.23), (20.0, 0.5), (60.0, 0.78)])
def test_dog_scaling_hits_target_mass_and_fractions(mass, height):
    spec = load_dog_spec()
    sc = scale_dog_spec(spec, mass, height)
    m = spec.compile()
    assert m.body_subtreemass[1] == pytest.approx(mass, rel=1e-6)
    seg = {}
    for i in range(1, m.nbody):
        seg[segment_of(m.body(i).name)] = seg.get(segment_of(m.body(i).name), 0.0) + m.body_mass[i]
    for k, frac in ANATOMICAL_FRACTIONS.items():
        assert seg[k] / mass == pytest.approx(frac, rel=1e-6)
    assert sc.length_scale == pytest.approx(height / 0.57)
