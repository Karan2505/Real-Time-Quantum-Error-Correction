# AURA-Link v1.1 — Wire Protocol and Ingress Gateway Specification

Status: implemented and tested in `src/aura/link/`. Reference (golden) model for
the FPGA implementation. AURA-Link is an independent protocol; it does not reuse
any Riverlane QECi specification text, register map or wire format.

---

## 1. Layering

| Layer | Implementation | Purpose |
|---|---|---|
| PHY (serial) | `link/phy.py` | 64b/66b PCS, x^58+x^39+1 scrambler, SerDes, block lock, CDC into the FPGA core clock |
| PHY (IP)     | `link/udp.py` | one frame per UDP datagram, OS timestamps (lab / cross-machine use) |
| Framing      | `link/frames.py`, `link/crc.py` | 36-byte header with its own CRC-32C, payload + CRC-32C |
| Ingress      | `link/ingress.py` | Frame Parser, Arrival Timestamp Tagger, Sequence Tracker, Ingress Elastic FIFO |
| Session      | `link/endpoint.py` | handshake, config distribution, liveness, credits, CLOSE / auto-reconnect |
| Timing       | `link/timesync.py` | PTP-style two-step time transfer, PI clock servo |

## 2. Physical layer (AURA-PHY, serial)

* Line coding: 66-bit blocks, 2-bit sync header (`01` data, `10` control),
  64-bit payload transmitted LSB first.
* Control block types (IEEE 802.3 clause 49 codes): `0x78` START (+7 data
  bytes), `0x87 0x99 0xAA 0xB4 0xCC 0xD2 0xE1 0xFF` = TERMINATE with 0..7
  data bytes, `0x1E` IDLE. A frame is START, *n* data blocks, TERMINATE_k.
* Scrambler: self-synchronising x^58 + x^39 + 1 over the 64-bit payload (sync
  header unscrambled). One line bit error becomes three data bit errors
  (verified in `tests/test_link_phy.py`).
* Block lock: lost after 16 invalid sync headers within a 64-block window, or
  on loss of signal; re-acquired after 64 consecutive valid blocks. Frames
  that arrive while unlocked are lost. (Idle blocks are not simulated one by one;
  their effect on lock and descrambler resynchronisation is applied analytically.)
* Default rates: SFP+ 10GBASE-R, 10.3125 GBd (6.4 ns per block). SFP28
  25.78125 GBd is supported (the FPGA core clock must then be at least 391 MHz
  for a 64-bit datapath; configuration validation enforces this).
* Latency model (defaults): TX PCS+gearbox+serialiser 32 ns, propagation
  5 ns/m (2 m), RX CDR+deserialiser+block sync+descramble 48 ns, two-flop CDC
  synchroniser into a 300 MHz core clock with one 64-bit word per core cycle.
* Transmit QoS: strict priority real-time (SYN/COR) > timing > control > management.
  A frame already on the wire is never pre-empted; at least one idle block separates frames.

## 3. Frame format

All integers little-endian.

| Off | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | PREAMBLE | bytes `A5 C3` |
| 2 | 1 | VERSION | `0x11` (major 1, minor 1). A different major version is rejected |
| 3 | 1 | TYPE | see §4 |
| 4 | 1 | FLAGS | SYN: `0x01` SOFT, `0x02` LEAK, `0x04` FIRST, `0x08` LAST, `0x10` ABORT |
| 5 | 1 | CHANNEL | 0 real-time, 1 management, 2 timing, 3 control |
| 6 | 2 | PATCH_ID | logical patch / stream |
| 8 | 4 | SEQ | per (TYPE, PATCH) sequence number, RFC 1982 serial arithmetic, restarts at 0 each session |
| 12 | 4 | SHOT | shot number |
| 16 | 2 | ROUND | QEC round within the shot (the last frame of a shot carries the data-qubit measurements) |
| 18 | 2 | PAYLOAD_LEN | N |
| 20 | 8 | ORIGIN_TS | sender clock, picoseconds. SYN: end of the round's last measurement |
| 28 | 4 | CONFIG_EPOCH | 32-bit hash of the decoder configuration shared by both ends |
| 32 | 4 | HDR_CRC | CRC-32C over bytes 0..31 |
| 36 | N | PAYLOAD | |
| 36+N | 4 | PAYLOAD_CRC | CRC-32C over the payload (present only when N > 0) |

CRC-32C (Castagnoli) check value: `crc32c("123456789") = 0xE3069283`.
All 1-bit errors in a frame are detected (exhaustively tested).

## 4. Frame types and payloads

| Type | Code | Dir | Payload |
|---|---|---|---|
| SYN | 0x01 | C→G | `u16 n_meas, u8 soft_bits (0/4/8), u8 0`, measurement bits (LSB-first), soft LLRs (4-bit two's complement nibbles low-first, or int8), leak bitmap (if LEAK) |
| COR | 0x02 | G→C | `u16 k, u8 status, u8 0, u32 decoder_latency_ns, u32 ref_seq`, k observable-flip bits |
| HEARTBEAT | 0x03 | both | `u8 state, u8, u16, u32 credit_limit, u64 tx_frames, u64 rx_frames, u32 rx_errors` |
| HELLO | 0x04 | C→G | TLVs: role, node, capability bits, per patch (id, frames/shot, max meas/frame), max frame |
| HELLO_ACK | 0x05 | G→C | TLVs: role, node, status (0 OK, 1 NEED_CONFIG, 2 VERSION, 3 EPOCH/LAYOUT MISMATCH, 4 BUSY), initial credit limit, reason |
| CLOSE | 0x06 | both | reason text (`no session` = "you are talking to a gateway without a session") |
| CREDIT | 0x07 | G→C | `u32 credit_limit, u16 fifo_occupancy, u16 flags (bit0 = above high watermark)` |
| TS_SYNC | 0x10 | C→G | `u16 seq, u16, u64 0` |
| TS_FOLLOW_UP | 0x11 | C→G | `u16 seq, u16, u64 t1` |
| TS_DELAY_REQ | 0x12 | G→C | `u16 seq, u16, u64 0` |
| TS_DELAY_RESP | 0x13 | C→G | `u16 seq, u16, u64 t4` |
| MGMT_DATA | 0x20 | both | `u16 xfer, u8 kind, u8, u32 total, u32 offset, u32 crc32c(blob)`, data |
| MGMT_ACK | 0x21 | both | `u16 xfer, u8 status (0 progress, 1 complete, 2 digest failure), u8, u32 next_expected_offset` |

COR status bits: `0x01` converged, `0x02` forced by WCET guard, `0x04` fallback
used, `0x08` erasure seen, `0x80` oracle (test decoder).

## 5. Session

```
control                                   gateway
HELLO(epoch E, layout) ------------------->  E unknown?  -> HELLO_ACK(NEED_CONFIG)
MGMT_DATA(decoder manifest, go-back-N) --->  verify CRC-32C and hash == E
HELLO(E) --------------------------------->  layout matches manifest -> HELLO_ACK(OK, credits)
TS_SYNC / FOLLOW_UP / DELAY_REQ / RESP <-->  PI servo locks the gateway clock
<--------------------- HEARTBEAT(state=UP)   gateway UP once |offset| < 2 ns for 4 samples
SYN ... ---------------------------------->  ingress pipeline -> Layer 2
<------------------------------------ COR
```

* Configuration epoch: `blake2b-32` of the canonical decoder manifest (codes,
  round layouts, detectors, observables, reference parities, DEM). Frames with
  a different epoch are dropped and counted.
* Liveness: heartbeats every 50 µs; a session is declared DOWN after 250 µs
  without any frame from the peer.
* Half-open protection: a gateway without a session answers session traffic
  with a rate-limited `CLOSE("no session")`. The control endpoint tears down and
  re-handshakes automatically (`auto_reconnect`). The control system pauses at
  the next shot boundary until the link is UP again.
* Duplicate HELLOs (lost or crossed ACK) are re-acknowledged without resetting the session.

## 6. Ingress gateway pipeline (FPGA side)

| Block | Behaviour | Timing (defaults) |
|---|---|---|
| **AURA-Link PHY Rx** | SerDes, block lock, descrambler, 64b/66b decode, CDC FIFO into the core clock | 48 ns + 1 block + 2 core cycles |
| **Frame Parser** | cut-through header check when bytes 0..35 are in the core domain (preamble, version, header CRC, length); end-of-frame check (PCS status, exact length, payload CRC) | 2 cycles after word 4; 2 cycles after the last word |
| **Arrival Timestamp Tagger** | latches the disciplined clock at start of frame, subtracts the constant RX latency (IEEE 1588 reference plane), independent of frame length and queueing | at SOF |
| **Sequence Tracker** | per patch: in-order / re-ordered / duplicate / late; a missing sequence number becomes an **erasure** after a 2 µs re-order window, with (shot, round) inferred from stream position | at header time |
| **Ingress Elastic FIFO** | bounded (512), watermarks 75% / 25%, optional consumer service rate, overflow turns the record into an erasure, cumulative pop count drives credits | 1 core cycle |

Erasure reasons delivered to Layer 2: `gap` (lost frame, inferred position),
`payload_crc` / `pcs:*` / `truncated` (header was valid: exact position),
`overflow`, `layout`.

**Guarantee (tested):** every SYN frame sent is either delivered bit-exact
exactly once, or reported exactly once as an erasure carrying its correct
(shot, round). Nothing corrupted is ever delivered.

## 7. Flow control

Cumulative credits (PCIe-style, robust to lost CREDIT frames):
`credit_limit = fifo_capacity + records_popped` (mod 2^32), advertised in
HELLO_ACK, CREDIT and HEARTBEAT frames. The control side computes
`available = credit_limit − syn_sent`. Real-time data is never held back
mid-shot (qubits cannot wait). Instead a shot starts only when a whole shot's
worth of frames is available ("shot-boundary backpressure"); overruns are
counted.

## 8. Time synchronisation

A two-step PTP-style exchange runs every 20 µs. Egress timestamps are taken
when the first bit leaves the wire (two-step, via FOLLOW_UP); ingress
timestamps come from the Arrival Timestamp Tagger.

    offset = ((t2 − t1) − (t4 − t3) − asymmetry) / 2
    delay  = ((t2 − t1) + (t4 − t3)) / 2

The servo steps the clock the first time, or whenever |offset| > 1 µs. Otherwise it applies PI frequency steering (kp 0.7, ki 0.3, linuxptp structure). Measured in simulation: a 37 µs / 12.5 ppm initial error converges to ≤ 0.5 ns, the timestamp resolution. Path asymmetry appears as asymmetry/2 unless calibrated via `asymmetry_ps`. Over UDP (OS timestamps) accuracy is ~50–100 µs.

## 9. Changes relative to the v1.0 sketch in the design document (Section 5)

| v1.0 | v1.1 | Why |
|---|---|---|
| 2-byte preamble, TYPE, PATCH, SEQ, ARRIVAL_TS, LEN, PAYLOAD, CRC32 | added VERSION, FLAGS, CHANNEL, SHOT, ROUND, ORIGIN_TS, CONFIG_EPOCH, separate HDR_CRC | version negotiation; explicit shot/round removes ambiguity for the RPU; the epoch prevents decoding with the wrong configuration; the header CRC enables cut-through routing and exact erasure attribution |
| ARRIVAL_TS written into the frame on receipt | arrival timestamp is side-band metadata from the TSU | no need to rewrite the frame and recompute its CRC in the FPGA |
| timestamp when the last byte is validated | timestamp at start of frame, referred to the wire | frame-length independent, IEEE 1588 practice; "ready" time is recorded separately |
| CRC-32 (IEEE) | CRC-32C | better Hamming distance at these lengths; hardware support |
| malformed frames dropped | dropped **and** converted into erasures delivered to the decoder | the decoder can treat a missing round as erased detectors instead of silently mis-decoding |
| — | session, config push, liveness, credits, time sync, auto-reconnect | needed for an unattended production link |
