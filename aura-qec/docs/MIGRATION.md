# From the original Deltaflow test harness to AURA-QEC Layers 0–1

The original harness (`qeci_interface.py`, `stimulus_generator.py`,
`impairment_emulator.py`, `timestamp_capture.py`, `test_config.yaml`) was a
sound black-box test design. This package keeps its test philosophy:

- boundary timestamps;
- correlation by ID;
- separation of transport from algorithm;
- fault injection;
- throughput and latency sweeps.

It replaces the synthetic parts with physically generated data and a real protocol.

| Original | Limitation | Replaced by |
|---|---|---|
| `StimulusGenerator`: random bits with an error rate | no physics, no time correlation, no detectors | `qpu/` virtual QPU + `control/` control system: syndromes come from simulated qubits under circuit-level noise, drift and readout physics, with valid detectors and a DEM |
| `Chunk` / `Measurement` (chunk of random measurements) | chunking done at the source | one SYN frame per (patch, round); windowing and chunking move to the RPU (Layer 2), fed by `IngressSink` |
| `QECiStub` (thread pool + `sleep`, random corrections) | wall-clock sleeps, fake corrections | `IngressGateway` + decoder stand-ins (`testbed/decoder_stubs.py`: null, oracle, PyMatching) over the modelled PHY, in deterministic virtual time |
| `QECiAdapter` (TCP, placeholder framing) | no real framing or CRC; the TCP stream hides loss | AURA-Link v1.1 frames with CRC-32C over the serial PHY model or UDP (`link/udp.py`); loss surfaces as erasures |
| `ImpairmentEmulator` (sleep delay, drop, reorder, zeroed payload) | application-level, non-deterministic | `link/phy.py` impairments at the bit level (uniform BER, Gilbert–Elliott bursts, loss of signal), deterministic and seeded |
| `TimestampCapture` (`time.time_ns`, skew warning) | host clock only, skew not corrected | Arrival Timestamp Tagger at start of frame + PTP-style time sync (≤ 0.5 ns in the model) + origin timestamps in every frame |
| `test_config.yaml` | untyped, placeholder limits | strict TOML/JSON configs (`configs/*.toml`, unknown keys rejected, cross-field validation) |

The measurement formulas of the original test plan map directly:

| Test-plan quantity | New metric |
|---|---|
| L_chunk = t_egress − t_final_input | `measurement_to_correction_ps` (per shot) |
| per-round link latency | `meas_to_arrival_ps`, `ingress_pipeline_ps`, `meas_to_ready_ps` (gateway histograms) |
| T = N_processed / Δt | shots and frames per virtual second from `RunResult` |
| A = M_match / N_total | oracle decoder: every correction must match; logical error rate with Wilson interval |
