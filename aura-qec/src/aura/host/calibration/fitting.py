"""Robust curve fits for calibration experiments (scipy least squares with
data-driven initial guesses and sanity checks)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from ...core.errors import CalibrationError


@dataclass
class FitResult:
    params: dict
    stderr: dict
    residual_rms: float
    ok: bool
    note: str = ""


def _finish(names, popt, pcov, x, y, f) -> FitResult:
    err = np.sqrt(np.clip(np.diag(pcov), 0, None)) if pcov is not None and np.all(np.isfinite(pcov)) \
        else np.full(len(popt), np.nan)
    res = float(np.sqrt(np.mean((f(x, *popt) - y) ** 2)))
    return FitResult(dict(zip(names, map(float, popt))), dict(zip(names, map(float, err))), res,
                     bool(np.all(np.isfinite(popt))))


def fit_exp_decay(t: np.ndarray, y: np.ndarray) -> FitResult:
    """y = a exp(-t/tau) + c"""
    t = np.asarray(t, float); y = np.asarray(y, float)
    c0 = float(np.mean(y[-max(2, len(y) // 8):]))
    a0 = float(y[0] - c0) or 1e-3
    target = c0 + a0 / math.e
    idx = int(np.argmin(np.abs(y - target)))
    tau0 = float(max(t[idx], (t[-1] - t[0]) / 20 or 1.0))

    def f(t, a, tau, c):
        return a * np.exp(-t / tau) + c
    try:
        popt, pcov = curve_fit(f, t, y, p0=[a0, tau0, c0], maxfev=20000,
                               bounds=([-2, (t[1] - t[0]) / 10 if len(t) > 1 else 1e-12, -1],
                                       [2, 1e3 * max(t[-1], 1e-12), 2]))
    except (RuntimeError, ValueError) as e:
        raise CalibrationError(f"exponential fit failed: {e}") from e
    return _finish(["a", "tau", "c"], popt, pcov, t, y, f)


def _dominant_freq(t: np.ndarray, y: np.ndarray) -> float:
    yy = y - np.mean(y)
    n = len(t)
    dt = float(np.mean(np.diff(t)))
    freqs = np.fft.rfftfreq(8 * n, dt)
    spec = np.abs(np.fft.rfft(yy, 8 * n))
    spec[0] = 0
    return float(freqs[int(np.argmax(spec))])


def fit_damped_cosine(t: np.ndarray, y: np.ndarray, f_hint: float | None = None,
                      f_window: float = 0.5) -> FitResult:
    """y = a exp(-t/tau) cos(2 pi f t + phi) + c

    The frequency is searched in [f0 (1 - f_window), f0 (1 + f_window)] and
    always strictly below the Nyquist frequency of the sampling grid, so the
    fit can never lock onto a sampling alias."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    nyq = 0.5 / float(np.min(np.diff(t))) if len(t) > 1 else np.inf
    f_fft = _dominant_freq(t, y)
    f0 = f_hint if f_hint is not None else f_fft
    f_lo = max(0.0, f0 * (1 - f_window))
    f_hi = min(f0 * (1 + f_window), 0.98 * nyq)
    if f_hint is not None and f_lo <= f_fft <= f_hi:
        f0 = f_fft                          # data-driven start inside the allowed window
    if not f_lo < f0 < f_hi:
        f0 = 0.5 * (f_lo + f_hi)
    c0 = float(np.mean(y))
    a0 = float((np.max(y) - np.min(y)) / 2) or 0.1
    tau0 = float(t[-1] - t[0]) or 1.0

    def f(t, a, fr, phi, tau, c):
        return a * np.exp(-t / tau) * np.cos(2 * np.pi * fr * t + phi) + c
    best = None
    for phi0 in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        try:
            popt, pcov = curve_fit(f, t, y, p0=[a0, f0, phi0, tau0, c0], maxfev=20000,
                                   bounds=([0, f_lo, -2 * np.pi, (t[1] - t[0]) if len(t) > 1 else 1e-12, -1],
                                           [2, f_hi, 2 * np.pi, 1e4 * t[-1], 2]))
        except (RuntimeError, ValueError):
            continue
        r = float(np.mean((f(t, *popt) - y) ** 2))
        if best is None or r < best[0]:
            best = (r, popt, pcov)
    if best is None:
        raise CalibrationError("damped-cosine fit failed")
    return _finish(["a", "f", "phi", "tau", "c"], best[1], best[2], t, y, f)


def fit_cosine(x: np.ndarray, y: np.ndarray, period_hint: float) -> FitResult:
    """y = a cos(2 pi x / period + phi) + c  (period free)"""
    x = np.asarray(x, float); y = np.asarray(y, float)

    def f(x, a, period, phi, c):
        return a * np.cos(2 * np.pi * x / period + phi) + c
    best = None
    for phi0 in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        try:
            popt, pcov = curve_fit(f, x, y, p0=[0.5, period_hint, phi0, 0.5], maxfev=20000,
                                   bounds=([0, period_hint / 4, -2 * np.pi, -1], [2, period_hint * 4, 2 * np.pi, 2]))
        except (RuntimeError, ValueError):
            continue
        r = float(np.mean((f(x, *popt) - y) ** 2))
        if best is None or r < best[0]:
            best = (r, popt, pcov)
    if best is None:
        raise CalibrationError("cosine fit failed")
    return _finish(["a", "period", "phi", "c"], best[1], best[2], x, y, f)


def fit_phase(phi: np.ndarray, y: np.ndarray) -> FitResult:
    """y = c + a cos(phi - phi0) with known unit period in phi (linear least squares)."""
    phi = np.asarray(phi, float); y = np.asarray(y, float)
    A = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi)])
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, u, v = coef
    amp = float(np.hypot(u, v))
    phi0 = float(np.arctan2(v, u))
    resid = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
    return FitResult({"c": float(c), "a": amp, "phi0": phi0}, {}, resid, amp > 0.05)


def fit_line(x: np.ndarray, y: np.ndarray) -> FitResult:
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
    return FitResult({"slope": float(coef[0]), "intercept": float(coef[1])}, {}, resid, True)
