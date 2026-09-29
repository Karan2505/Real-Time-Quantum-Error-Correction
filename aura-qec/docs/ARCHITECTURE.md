# AURA-QEC Layer 0 + Layer 1 — Architecture

```
 ┌──────────────────────────── Layer 0: quantum hardware & control ─────────────────────────────┐
 │                                                                                             │
 │  CONTROL-SYSTEM HOST                                                                        │
 │   ├─ QEC compiler ── circuit builder → native lowering → timed schedule → QPU program        │
 │   │                  → sequencer programs → detectors/observables → DEM → decoder manifest   │
 │   ├─ calibration ─── calibration graph (bring-up / maintenance) → store (versioned)         │
 │   ├─ experiment runner (VirtualLab) ── stream / batch runs, results, statistics             │
 │   └─ operations console ── REST API + web dashboard, CLI                                    │
 │                                                                                             │
 │  VIRTUAL CONTROL SYSTEM                                                                     │
 │   timing ─► per-channel sequencers (AURA-SEQ ISA) ─pulse bus─► VIRTUAL QPU                  │
 │   readout cards (discriminate, LLR, leakage flag, timestamp) ◄─IQ─┘                         │
 │   results crossbar ─► syndrome packetizer ─► AURA-Link TX                                   │
 │   Pauli-frame / decision registers ◄─ corrections ◄─ AURA-Link RX (feed-forward)             │
 └─────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                               │ SFP+ serial lane (or UDP)
 ┌───────────────────────── Layer 1: AURA-Link ingress transport gateway ─────────────────────┐
 │  PHY Rx ─► Frame Parser ─► Arrival Timestamp Tagger ─► Sequence Tracker ─► Elastic FIFO ─► │──► Layer 2 (RPU)
 │  PHY Tx ◄─ corrections (egress) ◄───────────────────────────────────────────────────────── │◄── decoder
 │  session · time sync (slave) · credits · management ARQ · telemetry                        │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
```

Everything runs on one deterministic discrete-event clock (`core/sim.py`,
integer picoseconds). A run with the same seed is bit-for-bit reproducible.
The same transport code also runs on the wall clock over real sockets
(`RealtimeScheduler` + `link/udp.py`).

## 1. Virtual QPU (`src/aura/qpu/`)

**Device model (`device.py`).** Flux-tunable transmons with f01,
anharmonicity, T1, T2*, T2-echo, π-pulse amplitude, |1>–|2> π amplitude,
residual gate error, leakage, reset infidelity, and dispersive readout (IQ blobs for
|0>, |1>, |2>). Tunable couplers carry: CZ flux amplitude (the conditional
phase is proportional to it), CZ error, CZ leakage, spectator crosstalk, and
unknown dynamic single-qubit phases.

The chip is *code-native*: one coupler per (ancilla, data) pair of the
Tanner graph, as in planar surface-code chips or long-range-coupler BB designs.

Profiles `sota`, `nisq` and `ideal` are representative, not vendor data.

**Hidden truth vs. knowledge.** `DeviceTruth` is visible only to the QPU. The host
sees `DeviceSpec` (topology, design targets, durations) and must *measure*
everything else.

**Drift (`drift.py`).** The following fluctuate over time:

- Ornstein–Uhlenbeck fluctuations of T1, frequency, π amplitude, CZ amplitude and readout phase;
- two-level-system T1 dips;
- cosmic-ray bursts (Poisson, ~0.1/s): T1 is suppressed on every qubit within 3 coupler hops and recovers over ~25 ms (McEwen et al., Nat. Phys. 2022).

A `speedup` factor compresses lab time for demos.

**Execution engine (`qpu.py`, `stabilizer.py`).**

- An exact CHP tableau (validated against Stim on thousands of random circuits) produces the noiseless reference record.
- Shots are Pauli frames relative to it, run either one at a time (streaming, driven by sequencer events) or B at a time (batched).
- Noise is computed from the true, drifting parameters *combined with the control settings actually uploaded*:
  - drive amplitude error → over-rotation;
  - drive frequency error → detuning;
  - CZ flux amplitude → conditional-phase error;
  - virtual-Z compensation → residual dynamic phase.

  Mis-calibration therefore degrades QEC exactly as in a lab.
- Channels:
  - idle decoherence: Pauli-twirl, or *exact amplitude damping* for qubits in a computational-basis state;
  - gate depolarising error;
  - twirled coherent errors;
  - spectator Z;
  - reset error;
  - leakage: leaked partners randomise their CZ neighbour, seepage decays at 2/T1, reset removes leakage with 95% probability, and a leaked qubit reads out as the |2> blob.

**Readout (`readout.py`).** Gaussian IQ blobs, T1 decay *during* integration
(time-weighted signal plus a post-measurement state flip), measurement-induced
excitation, and IQ-plane rotation drift.

**Calibration physics (`physics.py`).** Lindblad density-matrix simulation
(qutrit for one qubit, 4-level for a coupler) with decoherence, detuning,
depolarising gate error, exact `Repeat` blocks (superoperator powers) and
Pauli-`Twirled` blocks (randomised-benchmarking-style averages).

## 2. Virtual control system (`src/aura/control/`)

* **AURA-SEQ ISA** (`isa.py`): WAIT, VZ, PLAY, ACQUIRE, RESET, WAIT_DECISION,
  COND_PLAY, SYNC, MARK and END on a 4 ns grid. Each channel (one per qubit,
  one per coupler) runs its own program. Sequencers track the frame phase and
  cross-check it against the compiler (a *self-check that caught a real lowering bug*).
* **Pulse bus**: ops starting at the same instant are delivered to the QPU as one
  bundle in canonical order.
* **Readout cards** (`readout.py`): Gaussian ML discriminator, quantised LLR
  soft information (4 or 8 bit), leakage flag. The readout-chain latency is
  400 ns, then 120 ns to classify.
* **Results crossbar** 140 ns, then the **syndrome packetizer** (40 ns) builds one SYN frame per
  (patch, round) as soon as all its ancilla results are in.
* **Feed-forward**: corrections set decision registers (160 ns broadcast);
  WAIT_DECISION/COND_PLAY implement logical branching with a timeout. Waiting
  qubits accumulate real idle noise.
* **Shot loop**: back-to-back shots with a repetition delay and shot-boundary
  backpressure from AURA-Link credits.

## 3. Control-system host (`src/aura/host/`)

**Compiler (`compiler/`).**

* Builder: syndrome extraction for any CSS code.
  - Interleaved hook-safe CNOT order for rotated surface codes. Circuit distance equals d, verified with Stim for d = 3, 5, 7.
  - Generic *separated* schedule for any CSS code (e.g. BB codes), using optimal König edge colouring.
  - Multi-patch programs run in parallel.
* Lowering:
  - CX → H·CZ·H, with H·H pairs cancelled.
  - H → VZ·SX·VZ with virtual-Z frame tracking.
  - Per-qubit ASAP timing on the 4 ns grid, with a readout barrier per round.
  - Dynamical-decoupling X180 pairs in long data idles.
  - Output: canonical QPU program + channel programs.
* Detectors/observables for memory experiments in the Z or X basis, with reference parities from a noiseless run.
* **Detector error model**: backward sensitivity propagation over packed bit-vectors, with exact disjoint→independent channel conversion and a built-in *determinism check*. Its structure matches Stim's DEM mechanism-for-mechanism.
* Decoder manifest: round layouts, detectors, observables, DEM, code matrices, plus the config epoch.
* Stim export for interoperability.

**Calibration (`calibration/`).**

- Experiments: readout (|0>/|1>/|2> blobs, assignment errors), Rabi, error-amplified Rabi, two-stage Ramsey (±artificial detuning, alias-proof fits), T1, Hahn echo, Pauli-twirled single-qubit benchmark, error-amplified conditional-phase and dynamic-phase calibration, twirled CZ benchmark.
- A dependency graph provides `calibrate_all` (bring-up) and `maintain`, which only recalibrates stale or out-of-spec targets; cheap checks run first and only coupled edges cascade.
- Coupler experiments run in parallel colour groups.
- Virtual time advances by the experiment's lab duration, so the device drifts while being calibrated.

**Runner (`lab.py`, `runner/`).** `VirtualLab` composes everything:

- **stream** mode runs the full stack in virtual time;
- **batch** mode runs the QPU → discriminator → decoder path vectorised, for statistics.

Results include:

- LER with Wilson intervals;
- LER per round;
- measured vs model-predicted detection rate (a live calibration-quality indicator);
- latency distributions;
- feed-forward response time and link/decoder telemetry.

**Console (`console/`, `cli.py`).** Standard-library HTTP server with a background job worker, token auth, and a live dashboard (device map, calibration, link, jobs, runs, event log, BER fault injection). The `aura` CLI provides `info`, `compile`, `calibrate`, `run`, `console`, `link-loopback` and `selftest`.

## 4. AURA-Link (`src/aura/link/`)

See `AURA_LINK_SPEC.md`.

## 5. Measured timing (surface-3, `sota` defaults, virtual time)

| Segment | Value |
|---|---|
| QEC round (4 CZ layers + readout + reset) | ≈ 1.0 µs |
| last measurement of a round → SYN frame at gateway wire | 745 ns (control readout chain 700 ns + link 45 ns) |
| gateway wire SOF → record ready for Layer 2 (ingress pipeline) | 112 ns (150 ns for the 72-check BB frame) |
| last measurement → record ready for Layer 2 | 860 ns |
| last measurement → correction back at control (stand-in decoder 400 ns + 20 ns/event) | 1.45 µs |
| last measurement → conditional pulse (feed-forward response) | 1.61 µs |
| gateway clock error after lock | ≤ 0.5 ns |

The control-side latencies default to the same order of magnitude as the
component breakdown published by Caune et al., Nature Communications 17, 7383
(2026). They are configurable and should be replaced by measured values for a
specific control stack.

## 6. Scale and performance (single CPU core, Python)

| Workload | Wall-clock |
|---|---|
| surface-3, stream mode (full stack) | ~13 ms per shot |
| surface-3, batch mode | ~22k shots/s (sampling), ~7k shots/s with software MWPM decoding |
| BB [[72,12,6]], stream mode | ~150 ms per shot |
| BB [[72,12,6]], batch mode | ~1.6k shots/s |
| full calibration of surface-3 (17 qubits, 24 couplers) | ~15 s wall, 22 s virtual lab time |

For large devices (BB144/288), calibration takes minutes. Use `calibrate("oracle")` for fast demos.
