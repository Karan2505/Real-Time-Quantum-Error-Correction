import dataclasses

import pytest

from aura.control.isa import ChannelProgram, Instr
from aura.core import HardwareFault, US
from aura.lab import LabConfig, VirtualLab


def test_sequencer_frame_phase_self_check_catches_bad_programs():
    lab = VirtualLab(LabConfig(seed=2, cosmic_rays=False))
    lab.calibrate("oracle")
    ce = lab.compile(lab.experiment(rounds=1))
    chans = list(ce.sequencers.channels)
    for i, ch in enumerate(chans):
        instrs = list(ch.instrs)
        for j, ins in enumerate(instrs):
            if ins.op == "VZ":
                instrs[j] = Instr("VZ", quarter=(ins.quarter + 1) % 4)     # corrupt a frame update
                chans[i] = ChannelProgram(ch.channel, ch.kind, ch.target, tuple(instrs))
                break
        else:
            continue
        break
    bad = dataclasses.replace(ce, sequencers=dataclasses.replace(ce.sequencers, channels=tuple(chans)))
    lab.control.load(bad)
    lab.link.control.on_ready = lambda: lab.control.run_shots(1)
    lab.link.control.open(bad.manifest)
    with pytest.raises(HardwareFault):
        lab.sim.run(until_ps=lab.sim.now + 2000 * US)


def test_decision_timeout_when_decoder_is_too_slow():
    lab = VirtualLab(LabConfig(seed=3, cosmic_rays=False, decoder_latency_ns=80_000.0))
    lab.calibrate("oracle")
    exp = lab.experiment(rounds=2, branching=True, decision_timeout_ps=20 * US)
    r = lab.run(exp, shots=5, decoder="oracle", mode="stream")
    assert r.branching["timeouts"] == 5
    assert r.branching["response_ps"]["min"] >= 20 * US
