# Decisions taken, and decisions that need you

## Decisions I took (reversible, flagged for review)

1. **Virtual-time digital twin in Python.** Every hardware block runs on a
   deterministic nanosecond discrete-event clock. Python cannot meet real-time
   deadlines, so wall-clock nanoseconds would be fiction. The transport also
   runs over real UDP sockets for cross-process and cross-machine use. The link
   code is written as a *bit-exact golden model* for the future HLS/RTL gateway
   (same frame format, same state machines, same timing structure).
2. **Two physics tiers from one hidden truth.** An exact stabilizer + Pauli-frame
   engine runs QEC circuits (fast, validated against Stim). A Lindblad
   density-matrix engine runs calibration experiments, which Pauli models cannot
   represent.
3. **The control system ships raw measurements + soft information + leakage
   flags, not detection events.** Detection events, windowing and chunking belong
   to the Router & Preprocessing Unit (Layer 2), as in the design document.
4. **AURA-Link v1.1 header** (see `AURA_LINK_SPEC.md §9`): adds version,
   flags, channel, shot, round, origin timestamp, config epoch and a separate
   header CRC, and switches to CRC-32C. Arrival timestamps become side-band
   metadata at start of frame. Lost or corrupted rounds become **erasures**
   instead of silent drops.
5. **Code-native chips.** The device generator builds the Tanner-graph
   connectivity each code needs (planar for surface codes, long-range couplers
   for BB codes) instead of mapping onto a fixed lattice.
6. **Naming.** No Riverlane trademarks (Deltaflow, QECi) or specification text
   are used. The transport is AURA-Link.
7. **Runtime dependencies are numpy and scipy only.** PyMatching is optional (the
   reference decoder stand-in). Stim is used only in tests.

## Decisions that need you

| # | Question | Options / my recommendation |
|---|---|---|
| D1 | **Update the design document (Section 5) to AURA-Link v1.1?** | Recommended: yes, the implementation is the reference now. I can regenerate the Word document. |
| D2 | **BB-code syndrome schedule.** The generic "separated" schedule is valid for any CSS code but is not depth-optimal, and its circuit-level distance is not yet verified. | Next step: implement the published depth-8 interleaved BB schedule (Bravyi et al. 2024) and verify its distance. Needs your go-ahead because it changes the BB decoder manifest (new epoch). |
| D3 | **Crosstalk characterisation node.** The DEM currently omits spectator crosstalk, a measured ~7 % detection-rate bias on BB-72. | Add a spectator-Ramsey calibration node (a few hours of work) so decoder priors include it. |
| D4 | **Leakage in the decoder.** Leakage cannot be expressed in a Pauli DEM. | Leakage flags are already in every SYN frame. Decide whether Layer 2/3 consumes them (erasure-style BP priors) — relevant to the RL-BP design. |
| D5 | **Control-latency defaults.** 400/120/140/40/160 ns are order-of-magnitude values from the literature. | Replace with measurements from the control hardware you target, if you have one in mind. |
| D6 | **Link rate / FPGA clock.** Defaults: SFP+ 10.3125 GBd, 300 MHz core. | Choose the real target (e.g. Alveo U50 GTY at 25G with a ≥ 391 MHz 64-bit datapath, or a 128-bit datapath at lower clock). |
| D7 | **Licence and repository hosting.** | The package metadata says "Proprietary". Choose a licence before sharing with collaborators. |
| D8 | **Console exposure.** The console binds to localhost with optional token auth and no TLS. | Keep it local, or put it behind a reverse proxy with TLS if it must be reachable remotely. |
