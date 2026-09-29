"""
Readout electronics: state discrimination and soft information.

Each readout card receives integrated IQ points from multiplexed feedlines
and classifies them with a calibrated Gaussian discriminator (equal-variance
blobs => linear decision boundary, as implemented on FPGA readout cards).
It also produces

* a log-likelihood ratio  LLR = log P(iq | 0) / P(iq | 1)  (soft information
  for soft-decision decoders), quantised to a small signed integer, and
* a leakage flag when the point is most likely the |2> blob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Discriminator:
    c0: complex
    c1: complex
    c2: Optional[complex]
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("discriminator sigma must be positive")
        if self.c0 == self.c1:
            raise ValueError("discriminator blobs coincide")

    def llr(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=complex)
        return (np.abs(iq - self.c1) ** 2 - np.abs(iq - self.c0) ** 2) / (2.0 * self.sigma ** 2)

    def classify(self, iq) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (bit, llr, leaked).  Leaked points report bit 1."""
        iq = np.atleast_1d(np.asarray(iq, dtype=complex))
        llr = self.llr(iq)
        bit = llr < 0
        if self.c2 is not None:
            d0 = np.abs(iq - self.c0)
            d1 = np.abs(iq - self.c1)
            d2 = np.abs(iq - self.c2)
            leaked = (d2 < d0) & (d2 < d1)
        else:
            leaked = np.zeros(iq.shape, dtype=bool)
        bit = bit | leaked
        return bit, llr, leaked


def quantize_llr(llr: np.ndarray, bits: int, step: float) -> np.ndarray:
    """Symmetric signed quantisation of LLRs to ``bits`` bits (0 -> none)."""
    if bits == 0:
        return np.zeros(np.shape(llr), dtype=np.int8)
    if bits not in (4, 8):
        raise ValueError("soft information width must be 0, 4 or 8 bits")
    lim = 2 ** (bits - 1) - 1
    return np.clip(np.round(np.asarray(llr) / step), -lim, lim).astype(np.int8)


def fit_discriminator(iq0: np.ndarray, iq1: np.ndarray, iq2: Optional[np.ndarray] = None) -> Discriminator:
    """Fit blob centres by robust (median) estimation; sigma from the |0> blob."""
    c0 = complex(np.median(iq0.real), np.median(iq0.imag))
    c1 = complex(np.median(iq1.real), np.median(iq1.imag))
    dev = np.concatenate([(iq0 - c0).real, (iq0 - c0).imag])
    sigma = float(1.4826 * np.median(np.abs(dev - np.median(dev))))
    c2 = None
    if iq2 is not None:
        c2 = complex(np.median(iq2.real), np.median(iq2.imag))
    return Discriminator(c0=c0, c1=c1, c2=c2, sigma=max(sigma, 1e-9))
