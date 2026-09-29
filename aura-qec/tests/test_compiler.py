
import pytest

from aura.codes import get_code
from aura.core import CompileError
from aura.host.calibration import CalibrationStore
from aura.host.compiler import (DecoderManifest, QecCompiler, generate_dem, memory_experiment, to_stim_text,
                                uniform_snapshot)
from aura.host.compiler.dem import DeterminismError
from aura.qpu import build_device
from aura.qpu.program import StaticOp

from conftest import make_compiled

stim = pytest.importorskip("stim")


def _stim_dem(text):
    dem = stim.Circuit(text).detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    out = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ds = tuple(sorted(t.val for t in inst.targets_copy() if t.is_relative_detector_id()))
        os_ = tuple(sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id()))
        q = out.get((ds, os_), 0.0)
        p = inst.args_copy()[0]
        out[(ds, os_)] = q + p - 2 * q * p
    return out


@pytest.mark.parametrize("spec,basis", [("rep:5", "Z"), ("surface:3", "Z"), ("surface:3", "X"),
                                        ("surface:5", "X"), ("bb:72", "Z"), ("bb:72", "X")])
def test_all_detectors_deterministic_and_reference_parities_zero(spec, basis):
    _, _, ce, _ = make_compiled(spec, 3, basis)
    pm = ce.manifest.patches[0]
    assert sum(pm.det_ref) == 0 and sum(pm.obs_ref) == 0
    assert pm.num_detectors == (len(pm.round_acqs[0]) * (pm.rounds - 1)
                                + 2 * (get_code(spec).num_z_checks if basis == "Z" else get_code(spec).num_x_checks))


@pytest.mark.parametrize("spec,basis", [("surface:3", "Z"), ("surface:5", "X"), ("bb:72", "Z")])
def test_dem_matches_stim_exactly_in_structure(spec, basis):
    _, _, ce, snap = make_compiled(spec, 3, basis, p=2e-3)
    pm = ce.manifest.patches[0]
    text = to_stim_text(ce.qpu_program, pm.detectors, pm.observables, snap, z_states=ce.z_states)
    sd = _stim_dem(text)
    mine = generate_dem(ce.qpu_program, pm.detectors, pm.observables, snap, approximate_disjoint=True,
                        z_states=ce.z_states).as_dict()
    assert set(sd) == set(mine)
    worst = max(abs(sd[k] - mine[k]) / sd[k] for k in sd)
    assert worst < 5e-3


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_surface_schedule_preserves_circuit_distance(d, basis):
    _, _, ce, snap = make_compiled(f"surface:{d}", d, basis)
    pm = ce.manifest.patches[0]
    c = stim.Circuit(to_stim_text(ce.qpu_program, pm.detectors, pm.observables, snap))
    assert len(c.shortest_graphlike_error()) == d


def test_broken_detector_is_caught_by_determinism_check():
    _, _, ce, snap = make_compiled("surface:3", 2, "Z")
    pm = ce.manifest.patches[0]
    # an X-check in round 0 of a Z-memory is random: using it alone as a detector must fail
    bad = [pm.round_acqs[0][0]]
    with pytest.raises(DeterminismError):
        generate_dem(ce.qpu_program, [bad], [], snap)


def test_program_is_valid_and_sequencers_cover_every_op():
    _, dev, ce, _ = make_compiled("surface:5", 3, "Z")
    ce.qpu_program.validate()
    covered = set()
    for ch in ce.sequencers.channels:
        for ins in ch.instrs:
            if ins.op_index >= 0:
                covered.add(ins.op_index)
        assert ch.instrs[-1].op == "END"
    assert covered == set(range(len(ce.qpu_program.ops)))
    for o in ce.qpu_program.ops:
        if o.kind == "CZ":
            assert (o.qubits[0], o.qubits[1]) in dev.spec.coupler_index()


def test_dynamical_decoupling_can_be_disabled_and_changes_only_x180_count():
    _, _, a, _ = make_compiled("surface:3", 3, "Z", dynamical_decoupling=True)
    _, _, b, _ = make_compiled("surface:3", 3, "Z", dynamical_decoupling=False)
    ka, kb = a.stats["ops"], b.stats["ops"]
    assert ka.get("X180", 0) > 0 and kb.get("X180", 0) == 0
    assert {k: v for k, v in ka.items() if k != "X180"} == {k: v for k, v in kb.items() if k != "X180"}


def test_manifest_roundtrip_and_epoch_stability():
    _, _, ce, _ = make_compiled("surface:3", 3, "Z")
    blob = ce.manifest.to_blob()
    back = DecoderManifest.from_blob(blob)
    assert back.config_epoch() == ce.config_epoch
    assert back.patches[0].dem.num_mechanisms == ce.manifest.patches[0].dem.num_mechanisms
    _, _, ce2, _ = make_compiled("surface:3", 3, "Z")
    assert ce2.config_epoch == ce.config_epoch            # deterministic compilation
    _, _, ce3, _ = make_compiled("surface:3", 4, "Z")
    assert ce3.config_epoch != ce.config_epoch


def test_uncalibrated_compile_is_refused_by_default():
    code = get_code("surface:3")
    dev = build_device([(0, code)], "sota", seed=1)
    with pytest.raises(CompileError):
        QecCompiler(dev.spec, CalibrationStore(dev.spec)).compile(memory_experiment(code, 3))


def test_multi_patch_programs_are_disjoint_and_parallel():
    c1, c2 = get_code("surface:3"), get_code("rep:5")
    dev = build_device([(0, c1), (7, c2)], "sota", seed=1)
    from aura.host.compiler import ExperimentSpec, PatchSpec, CompileOptions
    exp = ExperimentSpec("two", (PatchSpec(0, c1, "Z", 3), PatchSpec(7, c2, "Z", 5)),
                         CompileOptions(allow_uncalibrated=True))
    ce = QecCompiler(dev.spec, CalibrationStore(dev.spec)).compile(exp, snapshot=uniform_snapshot(dev.spec, 1e-3))
    ids = [p.patch_id for p in ce.manifest.patches]
    assert ids == [0, 7]
    all_acqs = [a for p in ce.manifest.patches for r in p.round_acqs for a in r]
    assert len(all_acqs) == len(set(all_acqs)) == ce.num_acq


def test_static_op_validation():
    with pytest.raises(ValueError):
        StaticOp(0, "CZ", (1,), 48000, 0)
    with pytest.raises(ValueError):
        StaticOp(0, "MEASURE", (1,), 500000, 0)
    with pytest.raises(ValueError):
        StaticOp(0, "FOO", (1,), 1, 0)
