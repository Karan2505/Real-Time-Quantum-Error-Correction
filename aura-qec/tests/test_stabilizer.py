import numpy as np
import pytest

from aura.qpu import pauli as P
from aura.qpu.stabilizer import FrameSimulator, Tableau

stim = pytest.importorskip("stim")


def test_tableau_matches_stim_on_random_clifford_circuits():
    rng = np.random.default_rng(1)
    for trial in range(150):
        n = int(rng.integers(1, 8))
        T = Tableau(n, np.random.default_rng(trial))
        S = stim.TableauSimulator()
        S.set_num_qubits(n)
        for _ in range(50):
            g = int(rng.integers(0, 9))
            q = int(rng.integers(n))
            if g == 0: T.h([q]); S.h(q)
            elif g == 1: T.s([q]); S.s(q)
            elif g == 2: T.sx([q]); S.sqrt_x(q)
            elif g == 3 and n > 1:
                a, b = (int(x) for x in rng.choice(n, 2, replace=False)); T.cx([a], [b]); S.cx(a, b)
            elif g == 4 and n > 1:
                a, b = (int(x) for x in rng.choice(n, 2, replace=False)); T.cz([a], [b]); S.cz(a, b)
            elif g == 5: T.pauli_x([q]); S.x(q)
            elif g == 6: T.pauli_y([q]); S.y(q)
            elif g == 7: T.s_dag([q]); S.s_dag(q)
            else:
                assert T.peek_z(q) == S.peek_z(q)
                m = S.measure(q)
                b, _ = T.measure_z(q, forced=int(m) if T.peek_z(q) == 0 else None)
                assert b == int(m)
        for q in range(n):
            assert T.peek_z(q) == S.peek_z(q)


def test_forcing_a_deterministic_measurement_wrongly_raises():
    t = Tableau(1)
    with pytest.raises(ValueError):
        t.measure_z(0, forced=1)


def test_frame_simulator_deterministic_measurements_never_flip(rng):
    # computational-basis circuits: every measurement is deterministic, so the
    # random Z kicks of the frame formalism must never reach the X frame
    f = FrameSimulator(4, 64, rng)
    for _ in range(30):
        f.cx([0], [1]); f.cz([1], [2]); f.s([2]); f.cx([3], [0])
        assert not f.measure_z(int(rng.integers(4))).any()


def test_frame_simulator_random_measurements_are_unbiased(rng):
    f = FrameSimulator(1, 20000, rng)
    f.h([0])
    assert abs(f.measure_z(0).mean() - 0.5) < 0.02


@pytest.mark.parametrize("k", [1, 2])
def test_independent_decomposition_is_exact(k, rng):
    size = 4 ** k
    p = rng.random(size) * 0.02
    p[0] = 0
    p = P._check(p)
    q = P.to_independent(p, strict=True)
    # compose independent mechanisms and compare with the original channel
    acc = np.zeros(size); acc[0] = 1.0
    for idx in range(1, size):
        ch = np.zeros(size); ch[idx] = q[idx]; ch[0] = 1 - q[idx]
        acc = P.compose(acc, ch)
    assert np.allclose(acc, p, atol=1e-12)


def test_pta_idle_limits():
    ch = P.pta_idle(50e6, 30e6, 1e6)
    assert ch[1] == ch[2] and ch[3] > 0
    assert np.isclose(ch.sum(), 1)
    assert P.pta_idle(50e6, 30e6, 0)[0] == 1.0
    with pytest.raises(ValueError):
        P.pauli_channel_1(0.6, 0.6, 0)
