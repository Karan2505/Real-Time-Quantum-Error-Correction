"""
Far-end stand-ins for Layers 2-3 (Router & Preprocessing Unit + decoder).

They implement the ``IngressSink`` interface of the AURA-Link gateway and
send corrections back through the gateway's egress path, so the whole
Layer 0 + Layer 1 loop can be exercised before the real decoder exists.

* ``NullDecoderStub``      -- answers "no correction" (measures the raw loop)
* ``OracleDecoderStub``    -- perfect answer using the noiseless reference
                              parities (validates every piece of plumbing:
                              logical error rate after correction must be 0)
* ``MatchingDecoderStub``  -- a real software decoder (PyMatching MWPM on the
                              compiler's detector error model), optional
                              dependency; gives meaningful logical error rates

Decoder latency is modelled in virtual time: ``base_ps + per_event_ps *
(#detection events)`` (Python compute time is irrelevant to the twin).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.telemetry import MetricsRegistry
from ..core.units import NS
from ..link.records import (STATUS_CONVERGED, STATUS_ERASURE_SEEN, STATUS_ORACLE, CorrectionRecord,
                            ErasureRecord, SyndromeRecord)


@dataclass
class _ShotBuf:
    rounds: dict = field(default_factory=dict)      # round -> bits
    erased: set = field(default_factory=set)
    last_seq: int = -1
    done: bool = False


class _BaseStub:
    name = "stub"

    def __init__(self, sched, gateway=None, base_latency_ps: int = 400 * NS, per_event_ps: int = 0,
                 metrics: Optional[MetricsRegistry] = None):
        self.s = sched
        self.gw = gateway
        self.base = base_latency_ps
        self.per_event = per_event_ps
        self.m = metrics or MetricsRegistry(self.name)
        self.manifest = None
        self._bufs: dict[tuple[int, int], _ShotBuf] = {}
        self._patch = {}
        self.decoded = 0
        self.records: list[SyndromeRecord] = []
        self.erasures: list[ErasureRecord] = []
        self.keep_records = False

    def attach(self, gateway) -> None:
        self.gw = gateway

    # ---------------------------------------------------------- IngressSink
    def on_session(self, event: str, info: dict) -> None:
        if event == "config":
            self.manifest = info["manifest"]
            self._patch = {pm.patch_id: pm for pm in self.manifest.patches}
            self._bufs.clear()
            self._configured()

    def _configured(self) -> None:
        pass

    def on_syndrome(self, rec: SyndromeRecord) -> None:
        if self.keep_records:
            self.records.append(rec)
        pm = self._patch.get(rec.patch)
        if pm is None:
            self.m.counter("unknown_patch").inc()
            return
        b = self._bufs.setdefault((rec.patch, rec.shot), _ShotBuf())
        if b.done:
            self.m.counter("after_decode").inc()
            return
        b.rounds[rec.round] = rec.bits
        b.last_seq = max(b.last_seq, rec.seq)
        if rec.round == pm.rounds or len(b.rounds) + len(b.erased) == pm.rounds + 1:
            if len(b.rounds) + len(b.erased) >= pm.rounds + 1 or rec.round == pm.rounds:
                self._decode(rec.patch, rec.shot, b)

    def on_erasure(self, e: ErasureRecord) -> None:
        self.erasures.append(e)
        self.m.counter(f"erasure.{e.reason}").inc()
        pm = self._patch.get(e.patch)
        if pm is None or e.shot < 0:
            return
        b = self._bufs.setdefault((e.patch, e.shot), _ShotBuf())
        if b.done:
            return
        b.erased.add(e.round)
        if len(b.rounds) + len(b.erased) >= pm.rounds + 1:
            self._decode(e.patch, e.shot, b)

    # ---------------------------------------------------------- decoding
    def _assemble(self, pm, b: _ShotBuf) -> np.ndarray:
        """Bits indexed by acquisition id (missing rounds -> 0)."""
        bits = np.zeros(self.manifest.num_acq, dtype=np.uint8)
        for r, accs in enumerate(pm.round_acqs):
            if r in b.rounds:
                bits[accs] = b.rounds[r]
        return bits

    def _decode(self, patch: int, shot: int, b: _ShotBuf) -> None:
        b.done = True
        pm = self._patch[patch]
        bits = self._assemble(pm, b)
        obs, status, nevents = self.decode(pm, bits)
        if b.erased:
            status |= STATUS_ERASURE_SEEN
        lat = self.base + self.per_event * nevents
        cor = CorrectionRecord(patch=patch, shot=shot, round=pm.rounds, obs_flips=np.asarray(obs, np.uint8),
                               status=status, decoder_latency_ns=lat // 1000, ref_seq=b.last_seq)
        self.decoded += 1
        self.m.counter("decoded").inc()
        self.s.call_later(lat, self._emit, cor)
        # bounded memory
        if len(self._bufs) > 4096:
            for k in list(self._bufs)[:1024]:
                if self._bufs[k].done:
                    del self._bufs[k]

    def _emit(self, cor: CorrectionRecord) -> None:
        if self.gw is not None:
            self.gw.send_correction(cor)

    def decode(self, pm, bits: np.ndarray) -> tuple[np.ndarray, int, int]:
        raise NotImplementedError


class NullDecoderStub(_BaseStub):
    name = "null-decoder"

    def decode(self, pm, bits):
        return np.zeros(len(pm.observables), np.uint8), 0, 0


class OracleDecoderStub(_BaseStub):
    """Returns the exact observable flip (measured parity XOR noiseless reference)."""
    name = "oracle-decoder"

    def decode(self, pm, bits):
        return pm.observable_flips(bits), STATUS_ORACLE | STATUS_CONVERGED, 0


class MatchingDecoderStub(_BaseStub):
    """Minimum-weight perfect matching on the DEM restricted to the detectors
    that see the memory basis' logical errors (graph-like for surface /
    repetition codes).  Requires ``pymatching``."""
    name = "matching-decoder"

    def __init__(self, *a, per_event_ps: int = 20 * NS, **kw):
        super().__init__(*a, per_event_ps=per_event_ps, **kw)
        self._matchers: dict[int, tuple] = {}

    def _configured(self) -> None:
        try:
            import pymatching
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("MatchingDecoderStub needs the optional 'pymatching' package "
                               "(pip install 'aura-qec[decoders]')") from e
        self._matchers = {}
        for pm in self.manifest.patches:
            if pm.dem is None:
                raise RuntimeError("manifest has no DEM; compile with build_dem=True")
            keep = [i for i, b in enumerate(pm.det_basis) if b == pm.basis]
            dem = pm.dem.restrict(keep)
            h_cols, weights, l_cols = [], [], []
            split = 0
            for p, ds, os_ in zip(dem.probs, dem.dets, dem.obs):
                if p <= 0 or p >= 0.5:
                    continue
                w = float(np.log((1 - p) / p))
                if len(ds) <= 2:
                    h_cols.append(ds); l_cols.append(os_); weights.append(w)
                else:                                   # rare: split a hyperedge into pairs
                    split += 1
                    ds = list(ds)
                    for i in range(0, len(ds), 2):
                        part = tuple(ds[i:i + 2])
                        h_cols.append(part); l_cols.append(os_ if i == 0 else ()); weights.append(w)
            nd, nm = len(keep), len(h_cols)
            H = np.zeros((nd, nm), np.uint8)
            L = np.zeros((pm.dem.num_observables, nm), np.uint8)
            for j, (ds, os_) in enumerate(zip(h_cols, l_cols)):
                H[list(ds), j] = 1
                if os_:
                    L[list(os_), j] = 1
            mt = pymatching.Matching.from_check_matrix(H, weights=np.array(weights), faults_matrix=L,
                                                       use_virtual_boundary_node=True)
            self._matchers[pm.patch_id] = (mt, np.array(keep))
            self.m.gauge(f"hyperedges_split.p{pm.patch_id}").set(split)

    def decode(self, pm, bits):
        mt, keep = self._matchers[pm.patch_id]
        ev = pm.detection_events(bits)[keep]
        pred = mt.decode(ev)
        return np.asarray(pred, np.uint8), STATUS_CONVERGED, int(ev.sum())
