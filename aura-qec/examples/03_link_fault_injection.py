"""AURA-Link under stress: bit errors, burst errors and a fibre pull.
Every syndrome round is either delivered intact or reported to the decoder
as an erasure with its (shot, round) -- never silently lost or corrupted."""

from aura.core import US
from aura.lab import LabConfig, PatchConfig, VirtualLab
from aura.link import GilbertElliott, Impairments

lab = VirtualLab(LabConfig(seed=3, patches=[PatchConfig(0, "surface:3")], cosmic_rays=False))
lab.calibrate("oracle")
exp = lab.experiment(rounds=3)

for label, imp in [("clean", Impairments()),
                   ("BER 1e-4", Impairments(ber=1e-4)),
                   ("bursts", Impairments(burst=GilbertElliott(p_good_to_bad=1e-3, p_bad_to_good=0.1, ber_bad=0.05)))]:
    lab.link.set_impairments(down=imp)
    r = lab.run(exp, shots=200, decoder="oracle", mode="stream")
    er = {k: v for k, v in r.decoder_stats["counters"].items() if k.startswith("erasure.")}
    print(f"{label:9s}: logical errors {r.patches[0].logical_errors}, missing corrections "
          f"{r.patches[0].missing_corrections}, erasures {er}")
lab.link.set_impairments(down=Impairments())

# a 300 us fibre pull in the middle of a run: session drops, then the next run re-opens it
t0 = lab.sim.now
lab.link.set_impairments(down=Impairments(los_intervals=[(t0 + 200 * US, t0 + 500 * US)]))
r = lab.run(exp, shots=200, decoder="oracle", mode="stream")
print("fibre pull:", r.patches[0].shots, "shots,", r.notes or "no notes")
print("gateway state log:", lab.link.gateway.state_log[-4:])
