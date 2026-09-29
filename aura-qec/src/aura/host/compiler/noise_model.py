"""
Host-side noise model: turns the calibration store into a NoiseSnapshot
for DEM generation (the decoder's priors).

The host only knows what it has measured.  Parameters that calibration
drives to zero (over-rotation, detuning, conditional-phase error, dynamic
phases) are *assumed* zero; their real residuals and subsequent drift are
exactly the model mismatch a production decoder must live with.
"""

from __future__ import annotations


import numpy as np

from ...core.errors import CompileError
from ...qpu import noise as N
from ...qpu.device import DeviceSpec
from ..calibration.store import CalibrationStore

# Conservative fallbacks when a quantity was never measured.
DEFAULT_T1_PS = 50e6
DEFAULT_SQ_ERROR = 1e-3
DEFAULT_CZ_ERROR = 1e-2
DEFAULT_READOUT_ERROR = 2e-2
DEFAULT_RESET_ERROR = 2e-3
DEFAULT_SPECTATOR = 0.0


def estimated_snapshot(spec: DeviceSpec, cal: CalibrationStore, strict: bool = False) -> N.NoiseSnapshot:
    qs = []
    for q, qc in enumerate(cal.qubits):
        if strict and (qc.t1_ps is None or qc.p01 is None):
            raise CompileError(f"qubit {q} lacks calibration data (T1 / readout)")
        t1 = qc.t1_ps or DEFAULT_T1_PS
        t2s = min(qc.t2_star_ps or t1, 2 * t1)
        t2e = min(qc.t2_echo_ps or t2s, 2 * t1)
        ro = DEFAULT_READOUT_ERROR
        qs.append(N.QubitNoise(
            t1_ps=t1, t2_star_ps=t2s, t2_echo_ps=max(t2e, t2s),
            sq_depol=qc.sq_error if qc.sq_error is not None else DEFAULT_SQ_ERROR,
            meas_p01=qc.p01 if qc.p01 is not None else ro,
            meas_p10=qc.p10 if qc.p10 is not None else ro,
            reset_error=DEFAULT_RESET_ERROR if qc.p01 is None else max(qc.p01, 1e-4)))
    cs = {}
    for c, (ci, cc) in enumerate(zip(spec.couplers, cal.couplers)):
        cs[(ci.q0, ci.q1)] = N.CouplerNoise(cz_depol=cc.cz_error if cc.cz_error is not None else DEFAULT_CZ_ERROR,
                                            spectator_z=DEFAULT_SPECTATOR)
    d = spec.durations
    return N.build_snapshot(qs, cs, spec.neighbors(), d.sx_ps, d.x180_ps, d.cz_ps)


def uniform_snapshot(spec: DeviceSpec, p: float, t1_ps: float = float("inf")) -> N.NoiseSnapshot:
    """Simple benchmark model: depolarising p on every gate, readout flip p,
    reset error p, optional finite T1 (idle)."""
    t1 = t1_ps if np.isfinite(t1_ps) else 1e30
    qs = [N.QubitNoise(t1_ps=t1, t2_star_ps=min(2 * t1, t1), t2_echo_ps=min(2 * t1, t1), sq_depol=p,
                       meas_p01=p, meas_p10=p, reset_error=p) for _ in spec.qubits]
    cs = {(ci.q0, ci.q1): N.CouplerNoise(cz_depol=p) for ci in spec.couplers}
    d = spec.durations
    return N.build_snapshot(qs, cs, spec.neighbors(), d.sx_ps, d.x180_ps, d.cz_ps)
