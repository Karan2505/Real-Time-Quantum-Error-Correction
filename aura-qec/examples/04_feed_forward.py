"""Real-time feed-forward (logical branching): after the last QEC round the
control system waits for the decoder's decision and conditionally flips a
flag qubit.  Measures the full decoding response time (last measurement ->
conditional pulse), the quantity reported by Caune et al. (2026)."""

from aura.lab import LabConfig, PatchConfig, VirtualLab

lab = VirtualLab(LabConfig(seed=4, patches=[PatchConfig(0, "surface:3")], decoder_latency_ns=400.0))
lab.calibrate("oracle")
r = lab.run(lab.experiment(rounds=3, branching=True), shots=100, decoder="matching", mode="stream")
b = r.branching
print(f"response time p50 {b['response_ps']['p50'] / 1e3:.0f} ns, p99 {b['response_ps']['p99'] / 1e3:.0f} ns, "
      f"timeouts {b['timeouts']}, flag agrees with decision {b['flag_matches_decision']:.3f}")
