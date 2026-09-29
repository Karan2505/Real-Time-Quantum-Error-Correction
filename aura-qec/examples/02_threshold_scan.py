"""Fast batched statistics: logical error rate vs code distance on the same
device technology (below threshold, the LER falls with distance)."""

from aura.lab import LabConfig, PatchConfig, VirtualLab

for d in (3, 5, 7):
    lab = VirtualLab(LabConfig(seed=2, profile="sota", patches=[PatchConfig(0, f"surface:{d}")], cosmic_rays=False))
    lab.calibrate("oracle")            # fast: truth-based calibration (test mode)
    r = lab.run(lab.experiment(rounds=d), shots=20000, decoder="matching", mode="batch")
    p = r.patches[0]
    lo, hi = p.ler_ci
    print(f"d={d}: LER {p.ler:.5f} [{lo:.5f}, {hi:.5f}]  per round {p.ler_per_round}")
