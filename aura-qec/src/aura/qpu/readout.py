"""
Dispersive readout physics.

A measurement drives the readout resonator; the demodulated, integrated
signal is a point in the IQ plane whose mean depends on the transmon state
(|0>, |1>, |2>).  Physical effects:

* Gaussian amplifier noise (``sigma`` per quadrature);
* T1 decay *during* the integration window: a |1> that decays at time t_d
  produces the time-weighted average of the |1> and |0> responses and leaves
  the qubit in |0> (this is the dominant real source of asymmetric
  assignment error);
* measurement-induced excitation |0> -> |1>;
* slow rotation of the IQ plane (readout-phase drift) that degrades a stale
  discriminator.
"""

from __future__ import annotations

import numpy as np

from .device import ReadoutTruth


def sample_iq(states: np.ndarray, ro: ReadoutTruth, t1_ps: float, duration_ps: float,
              rotation_rad: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Integrated IQ points for true pre-measurement states (0, 1 or 2).

    Returns (iq, post_state).  ``post_state`` is the transmon state after the
    measurement (differs from the projected state if it decayed / was excited).
    """
    s = np.asarray(states, dtype=np.int64)
    b = s.shape[0]
    rot = np.exp(1j * rotation_rad)
    c0, c1, c2 = ro.c0 * rot, ro.c1 * rot, ro.c2 * rot
    iq = np.where(s == 0, c0, np.where(s == 1, c1, c2)).astype(complex)
    post = s.copy()
    # |1> -> |0> decay during integration
    ones = s == 1
    if ones.any() and np.isfinite(t1_ps):
        td = rng.exponential(t1_ps, size=b)
        dec = ones & (td < duration_ps)
        if dec.any():
            f = td[dec] / duration_ps
            iq[dec] = f * c1 + (1 - f) * c0
            post[dec] = 0
    # |2> -> |1> decay (rate ~ 2/T1)
    twos = s == 2
    if twos.any() and np.isfinite(t1_ps):
        td = rng.exponential(t1_ps / 2.0, size=b)
        dec = twos & (td < duration_ps)
        if dec.any():
            f = td[dec] / duration_ps
            iq[dec] = f * c2 + (1 - f) * c1
            post[dec] = 1
    # measurement-induced excitation
    zeros = s == 0
    if ro.excite_prob > 0 and zeros.any():
        ex = zeros & (rng.random(b) < ro.excite_prob)
        if ex.any():
            u = rng.random(int(ex.sum()))
            iq[ex] = u * c0 + (1 - u) * c1
            post[ex] = 1
    noise = rng.normal(0.0, ro.sigma, size=(b, 2))
    iq = iq + noise[:, 0] + 1j * noise[:, 1]
    return iq, post
