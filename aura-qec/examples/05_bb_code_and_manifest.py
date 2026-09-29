"""Bivariate-bicycle [[72,12,6]] code: compile, inspect the decoder manifest
(the configuration the RL-BP decoder will consume), stream a few shots."""

from aura.lab import LabConfig, PatchConfig, VirtualLab

lab = VirtualLab(LabConfig(seed=5, patches=[PatchConfig(0, "bb:72")], cosmic_rays=False))
lab.calibrate("oracle")
ce = lab.compile(lab.experiment(rounds=3))
pm = ce.manifest.patches[0]
H = pm.dem.check_matrix()
print(f"{pm.code_name}: {pm.num_detectors} detectors, {pm.dem.num_mechanisms} error mechanisms, "
      f"H is {H.shape}, max column weight {H.sum(0).max()}, config epoch 0x{ce.config_epoch:08x}")
print("manifest blob:", len(ce.manifest.to_blob()), "bytes (shipped to the gateway over AURA-Link MGMT)")
r = lab.run(lab.experiment(rounds=3), shots=20, decoder="oracle", mode="stream")
print(r.summary())
