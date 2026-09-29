import os

import numpy as np
import pytest

from aura.core import ProtocolError
from aura.link import frames as F
from aura.link.crc import _crc32c_py, crc32c


def test_crc32c_known_answer_and_backend_agreement():
    assert crc32c(b"123456789") == 0xE3069283
    for n in (0, 1, 7, 8, 9, 63, 64, 65, 1000):
        d = os.urandom(n)
        assert crc32c(d) == _crc32c_py(d)


def _frame(**kw):
    h = F.Header(type=F.FrameType.SYN, flags=0x05, channel=0, patch=3, seq=0xFFFFFFFE, shot=77, round=4,
                 origin_ts=123456789012, epoch=0xDEADBEEF, **kw)
    payload = F.encode_syn_payload(np.array([1, 0, 1, 1, 0, 0, 1], np.uint8), np.array([3, -7, 0, 7, -1, 2, -3]),
                                   4, np.array([0, 0, 1, 0, 0, 0, 0], bool))
    return h, payload, F.encode_frame(h, payload)


def test_header_and_syn_payload_roundtrip():
    h, payload, frame = _frame()
    assert len(frame) == F.HEADER_LEN + len(payload) + 4
    d = F.decode_header(frame[:36])
    assert (d.type, d.flags, d.patch, d.seq, d.shot, d.round, d.origin_ts, d.epoch) == \
        (h.type, h.flags, h.patch, h.seq, h.shot, h.round, h.origin_ts, h.epoch)
    p = F.check_payload(frame, d)
    bits, soft, leak = F.decode_syn_payload(p, True)
    assert bits.tolist() == [1, 0, 1, 1, 0, 0, 1]
    assert soft.tolist() == [3, -7, 0, 7, -1, 2, -3]
    assert leak.tolist() == [False, False, True, False, False, False, False]


@pytest.mark.parametrize("sb", [0, 8])
def test_syn_payload_widths(sb, rng):
    n = 133
    bits = rng.integers(0, 2, n).astype(np.uint8)
    soft = rng.integers(-127, 128, n) if sb else None
    p = F.encode_syn_payload(bits, soft, sb, None)
    b2, s2, l2 = F.decode_syn_payload(p, False)
    assert np.array_equal(bits, b2) and l2 is None
    if sb:
        assert np.array_equal(soft, s2)


def test_every_single_bit_flip_is_detected():
    _, _, frame = _frame()
    buf = bytearray(frame)
    for bit in range(len(buf) * 8):
        buf[bit // 8] ^= 1 << (bit % 8)
        try:
            h = F.decode_header(bytes(buf[:36]))
            F.check_payload(bytes(buf), h)
            detected = False
        except F.HeaderError:
            detected = True
        assert detected, f"undetected flip at bit {bit}"
        buf[bit // 8] ^= 1 << (bit % 8)


def test_malformed_frames_rejected():
    _, _, frame = _frame()
    with pytest.raises(F.HeaderError):
        F.decode_header(frame[:20])
    h = F.decode_header(frame[:36])
    with pytest.raises(F.HeaderError):
        F.check_payload(frame[:-1], h)
    with pytest.raises(F.HeaderError):
        F.check_payload(frame + b"\x00", h)
    bad = bytearray(frame)
    bad[2] = 0x21                      # major version 2
    bad[32:36] = F.crc32c(bytes(bad[:32])).to_bytes(4, "little")
    with pytest.raises(F.HeaderError) as ei:
        F.decode_header(bytes(bad[:36]))
    assert ei.value.reason == "version"
    with pytest.raises(ProtocolError):
        F.Header(type=1, patch=70000).pack()
    with pytest.raises(ProtocolError):
        F.encode_syn_payload(np.zeros(3, np.uint8), np.array([8, 0, 0]), 4, None)


def test_cor_tlv_and_serial_arithmetic():
    p = F.encode_cor_payload(np.array([1, 0, 1], np.uint8), 0x81, 1234, 99)
    obs, st, lat, ref = F.decode_cor_payload(p)
    assert obs.tolist() == [1, 0, 1] and (st, lat, ref) == (0x81, 1234, 99)
    t = F.encode_tlv({1: b"\x00", 4: b"abc"})
    assert F.decode_tlv(t) == {1: b"\x00", 4: b"abc"}
    with pytest.raises(F.HeaderError):
        F.decode_tlv(t[:-1])
    assert F.seq_diff(0, 0xFFFFFFFF) == 1
    assert F.seq_diff(0xFFFFFFFF, 0) == -1
    assert F.seq_diff(5, 3) == 2
