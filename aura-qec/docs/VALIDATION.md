# Validation evidence

Reproduce everything with `python -m pytest tests` (106 tests, ~45 s on one
core) and the scripts in `examples/`. Stim and PyMatching are used **only**
as independent oracles in the tests; they are not runtime dependencies.

## Quantum simulation correctness

| Claim | Evidence | Test |
|---|---|---|
| Stabilizer tableau is exact | peek/measure agreement with Stim's TableauSimulator on 150 random Clifford circuits (≥3,000 checks, 0 mismatches) | `test_stabilizer.py::test_tableau_matches_stim_on_random_clifford_circuits` |
| Disjoint→independent channel conversion is exact | recomposition reproduces the channel to 1e-12 (1- and 2-qubit) | `test_stabilizer.py::test_independent_decomposition_is_exact` |
| Compiled circuits are valid | all detectors deterministic and all reference parities 0 for rep-5, surface-3/5 (Z, X), BB-72 (Z, X) | `test_compiler.py::test_all_detectors_deterministic...` |
| DEM generator is correct | identical mechanism set to Stim's DEM (surface-3, surface-5, BB-72); probabilities within 0.5 % (the residual is Stim's disjoint-channel approximation; in matched approximation mode the differences are ~1e-6 absolute) | `test_compiler.py::test_dem_matches_stim_exactly_in_structure` |
| Surface schedule is hook-safe | circuit-level distance d = 3 and 5, both bases (Stim `shortest_graphlike_error`); d = 7 checked manually | `test_compiler.py::test_surface_schedule_preserves_circuit_distance` |
| The virtual QPU samples its own error model | per-detector rates within 4.5σ of DEM marginals over 20,000 shots; the mean within 4 standard errors | `test_qpu.py::test_qpu_sampling_matches_its_detector_error_model` |
| Noise-free device is noise-free | zero detection events and zero logical flips | `test_qpu.py::test_noiseless_device_produces_no_detection_events` |
| Mis-calibration hurts | 5 % amplitude + 200 kHz frequency error raises the detection rate by more than 1.5× | `test_qpu.py::test_miscalibration_increases_errors` |
| Readout physics | decay-induced P(0|1) matches the analytic 1 − exp(−τ/2T1) within 1.5e-3; the |2> blob is identified > 99 % | `test_qpu.py::test_readout_physics_assignment_errors` |
| Lindblad engine | Rabi angles, T1 decay, Ramsey fringes and ef transitions match analytic formulas; CZ conditional and dynamic phases; Repeat and Twirled blocks exact | `test_physics.py` |
| Below threshold | LER per round 0.43 % → 0.24 % → 0.12 % for d = 3, 5, 7 (`sota`, MWPM) | `examples/02_threshold_scan.py` |

## Calibration (recovering hidden truth from measurements only)

On a surface-3 chip (17 qubits, 24 couplers), full bring-up gives the
following residual errors vs. hidden truth:

| Parameter | Residual error |
|---|---|
| π amplitude | < 0.2 % |
| Drive frequency | ≈ 10 kHz (dominated by real frequency drift during the 22 s calibration) |
| CZ flux amplitude | ≈ 0.13 % (conditional-phase error ≈ 0.004 rad) |
| CZ dynamic-phase compensation | ≈ 0.004 rad |
| T1 | ~5 % median |
| Single-qubit error estimate (twirled benchmark) | within ~20 % of truth |
| CZ error estimate (twirled benchmark) | within ~20 % of truth |

Maintenance after bring-up costs ~1 s of lab time instead of 22 s. The tests are in `test_calibration.py` (marked `slow`).

## Control system and feed-forward

| Claim | Evidence |
|---|---|
| Frame-phase self-check | a corrupted VZ instruction raises `HardwareFault` (`test_control.py`) |
| Decision timeout path | a slow decoder produces timeouts, and the conditional pulse happens at the timeout (`test_control.py`) |
| Closed loop is exact | oracle decoder → 0 logical errors, 0 missing corrections over every shot (`test_lab.py`) |
| Feed-forward | flag qubit agrees with the decoder decision in 100 % of 40 shots; response ≈ 1.6 µs (`test_lab.py`, `examples/04_feed_forward.py`) |
| Stream vs batch equivalence | detection rates agree within statistics (`test_lab.py`) |

## AURA-Link

| Claim | Evidence |
|---|---|
| CRC-32C | known-answer value; the table and extension backends agree |
| Every single-bit error in a frame is detected | exhaustive over all bits of a SYN frame (`test_link_frames.py`) |
| PCS + scrambler | round trip for every frame length 7..89 bytes; exactly 3 data errors per line error |
| Latency model | exact SOF/EOF/header timing formulas (`test_link_phy.py::test_phy_latency_model`) |
| QoS | real-time frames overtake queued management frames, never pre-empt the frame on the wire |
| Loss of signal | lock lost once, relock after 64 blocks, frames during the outage lost |
| Frame accounting under errors | at BER 3e-5 and 2e-4, *every* SYN frame is delivered bit-exact or erased with its exact (shot, round); nothing corrupted is delivered (`test_link_session.py`) |
| Session | config push, time lock ≤ 2 ns, provisioning, liveness, automatic recovery, half-open repair, epoch filtering, corrections back |
| Credits | a slow consumer is protected: no FIFO overflow, shots throttled at shot boundaries |
| Management ARQ | the decoder configuration transfers over a bursty lossy link |
| Real network | UDP loopback: 400/400 frames, 100/100 corrections, 0 erasures (`test_lab.py::test_udp_loopback_over_real_sockets`) |
| Console | token auth, validation, job execution, run retrieval, dashboard served (`test_lab.py::test_console_api`) |

## Known model limitations (measured, not hidden)

* **Pauli-twirled coherent errors.** Over-rotation, detuning and phase errors are
  twirled per operation; coherent accumulation across a whole circuit is not
  modelled. The calibration experiments use the full Lindblad model.
* **The DEM excludes non-Pauli / uncharacterised physics.** It leaves out
  leakage and spectator crosstalk (the host has no crosstalk characterisation
  node yet). On BB-72 (`sota`) this shows up as a measured detection rate ≈ 1.23× the model:
  - ≈ 15 % of the excess is leakage;
  - ≈ 7 % is crosstalk;
  - with both disabled the ratio is 0.98.

  On surface-3 the measured/model ratio is typically 1.00–1.15, depending on the device instance and calibration. Leakage flags are carried in every SYN frame for leakage-aware decoding.
* **Idle blocks are not simulated one by one.** Block-lock loss from errors is evaluated per frame window.
* **Python cannot run in real time.** All nanosecond figures are virtual-time
  (modelled) results. Real-socket mode demonstrates protocol behaviour, not
  nanosecond latency.
