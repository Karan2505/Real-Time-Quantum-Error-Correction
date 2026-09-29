import math

import numpy as np
import pytest

from aura.qpu.physics import (CZPulse, Drive, QubitPhysics, Repeat, Twirled, Wait, evolve_1q, evolve_2q)

Q = QubitPhysics(t1_ps=80e6, t2_ps=60e6, detuning_hz=0.0, anharm_hz=-2e8, pi_amp=0.5, ef_pi_amp=0.35)


def test_rabi_rotation_angle():
    for frac in (0.25, 0.5, 1.0):
        p = evolve_1q(QubitPhysics(1e30, 1e30, 0, -2e8, 0.5, 0.35), [Drive(0.5 * frac, 0.0, 2)])
        assert p[1] == pytest.approx(math.sin(math.pi * frac / 2) ** 2, abs=1e-9)


def test_t1_decay():
    p = evolve_1q(Q, [Drive(0.5, 0, 2), Wait(40_000_000)])
    assert p[1] == pytest.approx(math.exp(-40 / 80), rel=1e-3)


def test_ramsey_fringe_frequency_and_decay():
    q = QubitPhysics(t1_ps=1e30, t2_ps=20e6, detuning_hz=1e6, anharm_hz=-2e8, pi_amp=0.5, ef_pi_amp=0.35)
    for t in (0, 250_000, 500_000, 1_000_000):
        p1 = evolve_1q(q, [Drive(0.25, 0, 0), Wait(t), Drive(0.25, 0, 0)])[1]
        expect = 0.5 * (1 + math.exp(-t / 20e6) * math.cos(2 * math.pi * 1e6 * t * 1e-12))
        assert p1 == pytest.approx(expect, abs=1e-6)


def test_ef_transition_populates_level_2():
    p = evolve_1q(QubitPhysics(1e30, 1e30, 0, -2e8, 0.5, 0.35), [Drive(0.5, 0, 2), Drive(0.35, 0, 2, "ef")])
    assert p[2] == pytest.approx(1.0, abs=1e-9)


def test_repeat_equals_unrolled_and_twirl_matches_depolarising_theory():
    d = Drive(0.501, 0.0, 24_000)
    a = evolve_1q(Q, [d] * 50)
    b = evolve_1q(Q, [Repeat((d,), 50)])
    assert np.allclose(a, b, atol=1e-12)
    q = QubitPhysics(1e30, 1e30, 0, -2e8, 0.5, 0.35, sq_depol=1e-3)
    x = ((0j, 1 + 0j), (1 + 0j, 0j))
    p = evolve_1q(q, [Twirled((Drive(0.5, 0, 2),), x, 1000)])
    assert p[1] == pytest.approx((1 - (1 - 4e-3 / 3) ** 1000) / 2, rel=1e-9)


def test_cz_conditional_and_dynamic_phases():
    q = QubitPhysics(1e30, 1e30, 0, -2e8, 0.5, 0.35)
    for ctrl in (0, 1):
        seq = ([Drive(0.5, 0, 2, target=0)] if ctrl else []) + [Drive(0.25, 0, 2, target=1), CZPulse(0.8, 2)]
        # scan the final analysis phase and find the fringe maximum
        phis = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        p1 = [evolve_2q(q, q, seq + [Drive(0.25, float(ph), 2, target=1)], 0.8, (0.3, 0.7))[[1, 3]].sum()
              for ph in phis]
        best = phis[int(np.argmax(p1))]
        expected = (0.7 + (math.pi if ctrl else 0.0)) % (2 * np.pi)
        assert abs(math.remainder(best - expected, 2 * np.pi)) < 0.1
