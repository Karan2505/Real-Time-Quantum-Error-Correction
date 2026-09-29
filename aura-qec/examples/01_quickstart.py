"""Quickstart: build a virtual lab, calibrate it from scratch, run a surface-code
memory experiment through the full stack (QPU -> control system -> AURA-Link
-> decoder stand-in -> back), and print logical error rate and latencies."""

from aura.lab import LabConfig, PatchConfig, VirtualLab

lab = VirtualLab(LabConfig(seed=1, profile="sota", patches=[PatchConfig(0, "surface:3")]))
print("calibration:", lab.calibrate("full")["actions"])

exp = lab.experiment(rounds=3, basis="Z")
res = lab.run(exp, shots=300, decoder="matching", mode="stream")
print(res.summary())
