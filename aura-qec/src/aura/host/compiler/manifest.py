"""
Decoder manifest: everything the decoding side needs to interpret the
syndrome stream of a compiled experiment.

Per patch:
* ``round_acqs[r]``  -- acquisition ids carried by the SYN frame of round r,
  in payload bit order (X-check ancillas, then Z-check ancillas; the final
  frame r = rounds carries the data-qubit measurements);
* detectors / observables as lists of acquisition ids (parity checks);
* reference parities of a noiseless run (detection event = parity XOR ref);
* the detector error model (priors) derived from calibration.

The manifest is serialised canonically; its hash is the ``config epoch``
that both ends of AURA-Link must agree on before syndrome data is accepted.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .dem import DetectorErrorModel


@dataclass
class PatchManifest:
    patch_id: int
    code_name: str
    code_fingerprint: str
    n: int
    k: int
    basis: str
    rounds: int
    hx: list[list[int]]
    hz: list[list[int]]
    round_acqs: list[list[int]]
    round_kinds: list[list[str]]              # per slot: 'x', 'z' or 'd'
    detectors: list[list[int]]
    det_coords: list[list[float]]
    det_basis: list[str]
    observables: list[list[int]]
    det_ref: list[int]
    obs_ref: list[int]
    flag_acq: int = -1
    dem: Optional[DetectorErrorModel] = None

    @property
    def num_detectors(self) -> int:
        return len(self.detectors)

    @property
    def payload_bits(self) -> list[int]:
        return [len(a) for a in self.round_acqs]

    def detection_events(self, bits_by_acq: np.ndarray) -> np.ndarray:
        """bits_by_acq: (..., num_acq) -> detection events (..., num_detectors)."""
        b = np.asarray(bits_by_acq, dtype=np.uint8)
        out = np.zeros(b.shape[:-1] + (len(self.detectors),), dtype=np.uint8)
        for i, accs in enumerate(self.detectors):
            out[..., i] = np.bitwise_xor.reduce(b[..., accs], axis=-1) ^ self.det_ref[i]
        return out

    def observable_flips(self, bits_by_acq: np.ndarray) -> np.ndarray:
        b = np.asarray(bits_by_acq, dtype=np.uint8)
        out = np.zeros(b.shape[:-1] + (len(self.observables),), dtype=np.uint8)
        for i, accs in enumerate(self.observables):
            out[..., i] = np.bitwise_xor.reduce(b[..., accs], axis=-1) ^ self.obs_ref[i]
        return out

    def to_dict(self, include_dem: bool = True) -> dict:
        d = {k: getattr(self, k) for k in (
            "patch_id", "code_name", "code_fingerprint", "n", "k", "basis", "rounds", "hx", "hz",
            "round_acqs", "round_kinds", "detectors", "det_coords", "det_basis", "observables",
            "det_ref", "obs_ref", "flag_acq")}
        if include_dem and self.dem is not None:
            d["dem"] = self.dem.to_json_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PatchManifest":
        d = dict(d)
        dem = d.pop("dem", None)
        m = cls(**d)
        m.dem = DetectorErrorModel.from_json_dict(dem) if dem else None
        return m


@dataclass
class DecoderManifest:
    experiment: str
    program_fingerprint: str
    num_acq: int
    patches: list[PatchManifest]
    acq_to_patch_round_slot: list[list[int]] = field(default_factory=list)   # acq -> [patch, round, slot]

    def patch(self, pid: int) -> PatchManifest:
        for p in self.patches:
            if p.patch_id == pid:
                return p
        raise KeyError(pid)

    def canonical_bytes(self, include_dem: bool = True) -> bytes:
        body = {"experiment": self.experiment, "program": self.program_fingerprint, "num_acq": self.num_acq,
                "patches": [p.to_dict(include_dem) for p in self.patches],
                "acq_map": self.acq_to_patch_round_slot}
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()

    def config_epoch(self) -> int:
        """32-bit identifier of this exact decoder configuration (never 0)."""
        h = hashlib.blake2b(self.canonical_bytes(True), digest_size=4).digest()
        v = int.from_bytes(h, "little")
        return v or 1

    def to_blob(self) -> bytes:
        return zlib.compress(self.canonical_bytes(True), 6)

    @classmethod
    def from_blob(cls, blob: bytes) -> "DecoderManifest":
        d = json.loads(zlib.decompress(blob))
        return cls(experiment=d["experiment"], program_fingerprint=d["program"], num_acq=d["num_acq"],
                   patches=[PatchManifest.from_dict(p) for p in d["patches"]],
                   acq_to_patch_round_slot=d["acq_map"])
