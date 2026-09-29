import numpy as np
import pytest

from aura.codes import get_code
from aura.host.calibration import CalibrationStore
from aura.host.compiler import QecCompiler, memory_experiment, uniform_snapshot
from aura.qpu import build_device


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def make_compiled(code_spec="surface:3", rounds=3, basis="Z", profile="sota", seed=1, p=1e-3, **opts):
    code = get_code(code_spec)
    dev = build_device([(0, code)], profile, seed=seed)
    comp = QecCompiler(dev.spec, CalibrationStore(dev.spec))
    snap = uniform_snapshot(dev.spec, p)
    ce = comp.compile(memory_experiment(code, rounds, basis, allow_uncalibrated=True, **opts), snapshot=snap)
    return code, dev, ce, snap
