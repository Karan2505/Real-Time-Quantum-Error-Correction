"""
AURA-Link v1.1 wire format.

All multi-byte integers are little-endian.  Every frame:

  off  size  field
  0    2     PREAMBLE      bytes A5 C3
  2    1     VERSION       0x11  (major 1, minor 1)
  3    1     TYPE          see FrameType
  4    1     FLAGS         type specific (SYN: SOFT, LEAK, FIRST, LAST, ABORT)
  5    1     CHANNEL       traffic class: 0 real-time, 1 management, 2 timing, 3 control
  6    2     PATCH_ID      logical patch / stream id
  8    4     SEQ           per-(type, patch) sequence number (serial arithmetic, wraps)
  12   4     SHOT          shot number
  16   2     ROUND         QEC round within the shot
  18   2     PAYLOAD_LEN   N
  20   8     ORIGIN_TS     sender clock (ps) -- SYN: end of the round's last measurement
  28   4     CONFIG_EPOCH  hash of the decoder configuration both ends must share
  32   4     HDR_CRC       CRC-32C over bytes 0..31
  36   N     PAYLOAD
  36+N 4     PAYLOAD_CRC   CRC-32C over the payload (only when N > 0)

A separate header CRC lets the gateway validate and route a frame as soon as
the first 36 bytes arrive (cut-through), and lets it attribute a payload
corruption to a known (patch, shot, round) -> precise erasure signalling.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.errors import ProtocolError
from .crc import crc32c

PREAMBLE = b"\xA5\xC3"
VERSION = 0x11
VERSION_MAJOR = VERSION >> 4
HEADER_LEN = 36
CRC_LEN = 4
MAX_PAYLOAD = 65535 - 64
_HDR = struct.Struct("<2sBBBBHIIHHQI")      # 32 bytes before HDR_CRC
assert _HDR.size == 32


class FrameType(enum.IntEnum):
    SYN = 0x01            # syndrome round (control -> gateway)
    COR = 0x02            # correction (gateway -> control)
    HEARTBEAT = 0x03
    HELLO = 0x04
    HELLO_ACK = 0x05
    CLOSE = 0x06
    CREDIT = 0x07
    TS_SYNC = 0x10
    TS_FOLLOW_UP = 0x11
    TS_DELAY_REQ = 0x12
    TS_DELAY_RESP = 0x13
    MGMT_DATA = 0x20
    MGMT_ACK = 0x21


class Channel(enum.IntEnum):
    REALTIME = 0
    MGMT = 1
    TIMING = 2
    CONTROL = 3


CHANNEL_OF = {
    FrameType.SYN: Channel.REALTIME, FrameType.COR: Channel.REALTIME,
    FrameType.HEARTBEAT: Channel.CONTROL, FrameType.HELLO: Channel.CONTROL,
    FrameType.HELLO_ACK: Channel.CONTROL, FrameType.CLOSE: Channel.CONTROL, FrameType.CREDIT: Channel.CONTROL,
    FrameType.TS_SYNC: Channel.TIMING, FrameType.TS_FOLLOW_UP: Channel.TIMING,
    FrameType.TS_DELAY_REQ: Channel.TIMING, FrameType.TS_DELAY_RESP: Channel.TIMING,
    FrameType.MGMT_DATA: Channel.MGMT, FrameType.MGMT_ACK: Channel.MGMT,
}


@dataclass
class Header:
    type: int
    flags: int = 0
    channel: int = 0
    patch: int = 0
    seq: int = 0
    shot: int = 0
    round: int = 0
    payload_len: int = 0
    origin_ts: int = 0
    epoch: int = 0
    version: int = VERSION

    def pack(self) -> bytes:
        for name, v, bits in (("flags", self.flags, 8), ("channel", self.channel, 8), ("patch", self.patch, 16),
                              ("seq", self.seq, 32), ("shot", self.shot, 32), ("round", self.round, 16),
                              ("payload_len", self.payload_len, 16), ("epoch", self.epoch, 32)):
            if not 0 <= v < (1 << bits):
                raise ProtocolError(f"header field {name}={v} does not fit in {bits} bits")
        if not 0 <= self.origin_ts < (1 << 64):
            raise ProtocolError("origin_ts out of range")
        body = _HDR.pack(PREAMBLE, self.version, int(self.type), self.flags, self.channel, self.patch,
                         self.seq, self.shot, self.round, self.payload_len, self.origin_ts, self.epoch)
        return body + struct.pack("<I", crc32c(body))


def encode_frame(h: Header, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload of {len(payload)} bytes exceeds maximum {MAX_PAYLOAD}")
    h.payload_len = len(payload)
    out = h.pack()
    if payload:
        out += payload + struct.pack("<I", crc32c(payload))
    return out


class HeaderError(ProtocolError):
    def __init__(self, reason: str, msg: str = ""):
        super().__init__(msg or reason)
        self.reason = reason


def decode_header(buf: bytes) -> Header:
    """Validate and decode the 36-byte header.  Raises HeaderError(reason)."""
    if len(buf) < HEADER_LEN:
        raise HeaderError("runt", f"frame shorter than header ({len(buf)} bytes)")
    if buf[0:2] != PREAMBLE:
        raise HeaderError("preamble", "bad preamble")
    (crc_rx,) = struct.unpack_from("<I", buf, 32)
    if crc32c(bytes(buf[:32])) != crc_rx:
        raise HeaderError("hdr_crc", "header CRC mismatch")
    (_, ver, typ, flags, chan, patch, seq, shot, rnd, plen, ts, epoch) = _HDR.unpack_from(buf, 0)
    if ver >> 4 != VERSION_MAJOR:
        raise HeaderError("version", f"unsupported protocol version 0x{ver:02x}")
    return Header(typ, flags, chan, patch, seq, shot, rnd, plen, ts, epoch, ver)


def frame_length(h: Header) -> int:
    return HEADER_LEN + (h.payload_len + CRC_LEN if h.payload_len else 0)


def check_payload(buf: bytes, h: Header) -> bytes:
    """Return the payload after length and CRC checks.  Raises HeaderError(reason)."""
    need = frame_length(h)
    if len(buf) < need:
        raise HeaderError("truncated", f"frame truncated: {len(buf)} < {need}")
    if len(buf) > need:
        raise HeaderError("oversize", f"trailing bytes: {len(buf)} > {need}")
    if h.payload_len == 0:
        return b""
    payload = bytes(buf[HEADER_LEN:HEADER_LEN + h.payload_len])
    (crc_rx,) = struct.unpack_from("<I", buf, HEADER_LEN + h.payload_len)
    if crc32c(payload) != crc_rx:
        raise HeaderError("payload_crc", "payload CRC mismatch")
    return payload


# =========================================================================== payload codecs
def _pack_bits(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8) & 1, bitorder="little").tobytes()


def _unpack_bits(data: bytes, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), count=n, bitorder="little").astype(np.uint8)


def _nbytes(n: int) -> int:
    return (n + 7) // 8


def encode_syn_payload(bits: np.ndarray, soft: Optional[np.ndarray], soft_bits: int,
                       leak: Optional[np.ndarray]) -> bytes:
    n = int(len(bits))
    if n > 65535:
        raise ProtocolError("too many measurements in one frame")
    out = struct.pack("<HBB", n, soft_bits if soft is not None else 0, 0) + _pack_bits(bits)
    if soft is not None:
        s = np.asarray(soft, dtype=np.int64)
        if soft_bits == 8:
            if np.any((s < -127) | (s > 127)):
                raise ProtocolError("8-bit soft value out of range")
            out += s.astype(np.int8).tobytes()
        elif soft_bits == 4:
            if np.any((s < -7) | (s > 7)):
                raise ProtocolError("4-bit soft value out of range")
            nib = (s & 0xF).astype(np.uint8)
            if n % 2:
                nib = np.append(nib, 0)
            out += (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8).tobytes()
        else:
            raise ProtocolError("soft_bits must be 4 or 8 when soft data is present")
    if leak is not None:
        out += _pack_bits(leak)
    return out


def decode_syn_payload(p: bytes, has_leak: bool) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if len(p) < 4:
        raise HeaderError("syn_payload", "SYN payload too short")
    n, sb, _ = struct.unpack_from("<HBB", p, 0)
    off = 4
    nb = _nbytes(n)
    need = 4 + nb + (n if sb == 8 else _nbytes(4 * n) if sb == 4 else 0) + (nb if has_leak else 0)
    if sb not in (0, 4, 8):
        raise HeaderError("syn_payload", f"bad soft width {sb}")
    if len(p) != need:
        raise HeaderError("syn_payload", f"SYN payload length {len(p)} != expected {need}")
    bits = _unpack_bits(p[off:off + nb], n)
    off += nb
    soft = None
    if sb == 8:
        soft = np.frombuffer(p[off:off + n], dtype=np.int8).copy()
        off += n
    elif sb == 4:
        raw = np.frombuffer(p[off:off + _nbytes(4 * n)], dtype=np.uint8)
        nib = np.empty(2 * len(raw), dtype=np.int16)
        nib[0::2] = raw & 0xF
        nib[1::2] = raw >> 4
        nib = nib[:n]
        soft = np.where(nib >= 8, nib - 16, nib).astype(np.int8)
        off += _nbytes(4 * n)
    leak = None
    if has_leak:
        leak = _unpack_bits(p[off:off + nb], n).astype(bool)
    return bits, soft, leak


_COR = struct.Struct("<HBBII")


def encode_cor_payload(obs_flips: np.ndarray, status: int, latency_ns: int, ref_seq: int) -> bytes:
    k = int(len(obs_flips))
    return _COR.pack(k, status & 0xFF, 0, min(max(int(latency_ns), 0), 0xFFFFFFFF), ref_seq & 0xFFFFFFFF) \
        + _pack_bits(obs_flips)


def decode_cor_payload(p: bytes) -> tuple[np.ndarray, int, int, int]:
    if len(p) < _COR.size:
        raise HeaderError("cor_payload", "COR payload too short")
    k, status, _, lat, ref = _COR.unpack_from(p, 0)
    if len(p) != _COR.size + _nbytes(k):
        raise HeaderError("cor_payload", "COR payload length mismatch")
    return _unpack_bits(p[_COR.size:], k), status, lat, ref


# --------------------------------------------------------------------- TLV (HELLO)
def encode_tlv(items: dict[int, bytes]) -> bytes:
    out = b""
    for t, v in sorted(items.items()):
        if len(v) > 65535:
            raise ProtocolError("TLV value too long")
        out += struct.pack("<BH", t, len(v)) + v
    return out


def decode_tlv(p: bytes) -> dict[int, bytes]:
    out = {}
    off = 0
    while off < len(p):
        if off + 3 > len(p):
            raise HeaderError("tlv", "truncated TLV")
        t, ln = struct.unpack_from("<BH", p, off)
        off += 3
        if off + ln > len(p):
            raise HeaderError("tlv", "TLV overruns payload")
        out[t] = bytes(p[off:off + ln])
        off += ln
    return out


# HELLO TLV types
TLV_ROLE = 1            # u8: 0 control, 1 gateway
TLV_NODE = 2            # utf8 name
TLV_CAPS = 3            # u32 capability bits
TLV_PATCHES = 4         # repeated <H patch, H rounds, H max_meas_per_frame>
TLV_MAX_FRAME = 5       # u16
TLV_STATUS = 6          # u8 (HELLO_ACK)
TLV_CREDIT = 7          # u32 initial credit limit (HELLO_ACK)
TLV_REASON = 8          # utf8

CAP_SOFT4 = 1 << 0
CAP_SOFT8 = 1 << 1
CAP_LEAK = 1 << 2
CAP_CUT_THROUGH = 1 << 3
CAP_MGMT_ARQ = 1 << 4
CAP_TIMESYNC = 1 << 5
CAP_ERASURE = 1 << 6
CAP_CREDITS = 1 << 7

HELLO_OK = 0
HELLO_NEED_CONFIG = 1
HELLO_VERSION = 2
HELLO_EPOCH_MISMATCH = 3
HELLO_BUSY = 4


# --------------------------------------------------------------------- misc payloads
_HB = struct.Struct("<BBHIQQI")           # state, rsv, rsv, credit_limit, tx_frames, rx_frames, rx_errors
_CREDIT = struct.Struct("<IHH")           # credit_limit, fifo occupancy, flags
_TS = struct.Struct("<HHQ")               # ts seq, rsv, timestamp (ps)
_MGMT = struct.Struct("<HBBIII")          # xfer id, kind, rsv, total, offset, digest
_MGMT_ACK = struct.Struct("<HBBI")        # xfer id, status, rsv, next expected offset


def encode_heartbeat(state: int, credit_limit: int, tx: int, rx: int, errors: int) -> bytes:
    return _HB.pack(state & 0xFF, 0, 0, credit_limit & 0xFFFFFFFF, tx, rx, errors & 0xFFFFFFFF)


def decode_heartbeat(p: bytes) -> tuple[int, int, int, int, int]:
    if len(p) != _HB.size:
        raise HeaderError("heartbeat", "bad heartbeat payload")
    st, _, _, cl, tx, rx, er = _HB.unpack(p)
    return st, cl, tx, rx, er


def encode_credit(limit: int, occupancy: int, flags: int = 0) -> bytes:
    return _CREDIT.pack(limit & 0xFFFFFFFF, min(occupancy, 0xFFFF), flags)


def decode_credit(p: bytes) -> tuple[int, int, int]:
    if len(p) != _CREDIT.size:
        raise HeaderError("credit", "bad credit payload")
    return _CREDIT.unpack(p)


def encode_ts(seq: int, ts: int = 0) -> bytes:
    return _TS.pack(seq & 0xFFFF, 0, ts & ((1 << 64) - 1))


def decode_ts(p: bytes) -> tuple[int, int]:
    if len(p) != _TS.size:
        raise HeaderError("timesync", "bad timesync payload")
    s, _, t = _TS.unpack(p)
    return s, t


def encode_mgmt(xfer: int, kind: int, total: int, offset: int, digest: int, data: bytes) -> bytes:
    return _MGMT.pack(xfer & 0xFFFF, kind, 0, total, offset, digest & 0xFFFFFFFF) + data


def decode_mgmt(p: bytes) -> tuple[int, int, int, int, int, bytes]:
    if len(p) < _MGMT.size:
        raise HeaderError("mgmt", "short MGMT payload")
    x, k, _, tot, off, dg = _MGMT.unpack_from(p, 0)
    return x, k, tot, off, dg, bytes(p[_MGMT.size:])


def encode_mgmt_ack(xfer: int, status: int, next_off: int) -> bytes:
    return _MGMT_ACK.pack(xfer & 0xFFFF, status, 0, next_off)


def decode_mgmt_ack(p: bytes) -> tuple[int, int, int]:
    if len(p) != _MGMT_ACK.size:
        raise HeaderError("mgmt_ack", "bad MGMT ACK payload")
    x, st, _, nx = _MGMT_ACK.unpack(p)
    return x, st, nx


# --------------------------------------------------------------------- serial arithmetic
def seq_diff(a: int, b: int) -> int:
    """a - b under RFC 1982 serial arithmetic on 32-bit sequence numbers."""
    d = (a - b) & 0xFFFFFFFF
    return d - (1 << 32) if d >= (1 << 31) else d
