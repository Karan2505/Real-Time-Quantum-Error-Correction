"""
Records exchanged between the control system, AURA-Link and the decoding
side (Router & Preprocessing Unit / decoder).  These are the in-memory
counterparts of the SYN / COR wire frames plus delivery metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# SYN flags
FLAG_SOFT = 0x01          # payload carries soft information (quantised LLRs)
FLAG_LEAK = 0x02          # payload carries leakage flags
FLAG_FIRST = 0x04         # first round of a shot
FLAG_LAST = 0x08          # last frame of a shot (final data-qubit measurement)
FLAG_ABORT = 0x10         # shot aborted by the control system
FLAG_RETX = 0x20          # retransmission (management traffic)


@dataclass
class SyndromeRecord:
    """One QEC round of one patch (payload of a SYN frame)."""
    patch: int
    shot: int
    round: int
    bits: np.ndarray                     # (n_meas,) uint8 raw measurement outcomes
    soft: Optional[np.ndarray] = None    # (n_meas,) int8 quantised LLR (log P0/P1)
    leak: Optional[np.ndarray] = None    # (n_meas,) bool
    flags: int = 0
    origin_ts: int = 0                   # control-system clock: end of last measurement of the round
    seq: int = -1                        # stream sequence number (filled by transport)
    # ingress metadata (filled by the gateway)
    sof_ts: int = -1                     # arrival timestamp of start of frame (gateway clock)
    ready_ts: int = -1                   # time the record left the ingress FIFO
    late: bool = False                   # arrived after being declared erased
    link_id: int = 0

    @property
    def is_last(self) -> bool:
        return bool(self.flags & FLAG_LAST)


@dataclass
class ErasureRecord:
    """A syndrome round that did not arrive intact (loss, CRC failure, overflow)."""
    patch: int
    seq: int
    reason: str                          # 'gap' | 'payload_crc' | 'overflow' | 'epoch'
    shot: int = -1
    round: int = -1
    inferred: bool = True                # shot/round inferred from stream position
    detect_ts: int = -1


@dataclass
class CorrectionRecord:
    """Decoder output for a patch (payload of a COR frame)."""
    patch: int
    shot: int
    round: int                           # last round covered by this decode
    obs_flips: np.ndarray                # (k,) uint8 logical observable flips
    status: int = 0                      # decoder status bits (converged, wcet_forced, fallback, ...)
    decoder_latency_ns: int = 0          # decoder-reported processing time
    ref_seq: int = -1                    # SYN sequence number that completed the window
    origin_ts: int = 0                   # gateway clock when the correction was emitted
    seq: int = -1
    arrival_ts: int = -1                 # control-system clock at arrival


# decoder status bits carried in COR frames
STATUS_CONVERGED = 0x01
STATUS_WCET_FORCED = 0x02
STATUS_FALLBACK = 0x04
STATUS_ERASURE_SEEN = 0x08
STATUS_ORACLE = 0x80
