import numpy as np
import pytest

from aura.codes import get_code, gf2, make_css_code
from aura.core import CodeError
from aura.host.compiler.builder import bipartite_edge_coloring


@pytest.mark.parametrize("spec,n,k", [("rep:5", 5, 1), ("surface:3", 9, 1), ("surface:5", 25, 1),
                                      ("surface:7", 49, 1), ("bb:72", 72, 12), ("bb:90", 90, 8),
                                      ("bb:108", 108, 8), ("bb:144", 144, 12), ("bb:288", 288, 12)])
def test_code_parameters_match_literature(spec, n, k):
    c = get_code(spec)
    assert (c.n, c.k) == (n, k)
    c.validate()


def test_logicals_are_canonical_and_commute():
    for spec in ("surface:5", "bb:72"):
        c = get_code(spec)
        assert not gf2.matmul(c.logical_x, c.hz.T).any()
        assert not gf2.matmul(c.logical_z, c.hx.T).any()
        assert np.array_equal(gf2.matmul(c.logical_x, c.logical_z.T), np.eye(c.k, dtype=np.uint8))


def test_surface_logicals_have_weight_d():
    for d in (3, 5, 7):
        c = get_code(f"surface:{d}")
        assert c.logical_x.sum() == d and c.logical_z.sum() == d


def test_invalid_codes_rejected():
    with pytest.raises(CodeError):
        make_css_code("bad", [[1, 0]], [[1, 1]])                # anticommuting checks
    with pytest.raises(CodeError):
        get_code("surface:4")
    with pytest.raises(CodeError):
        get_code("bb:999")


def test_gf2_inverse_and_nullspace(rng):
    for _ in range(20):
        m = rng.integers(0, 2, (6, 6)).astype(np.uint8)
        if gf2.rank(m) == 6:
            assert np.array_equal(gf2.matmul(m, gf2.inverse(m)), np.eye(6, dtype=np.uint8))
        a = rng.integers(0, 2, (4, 9)).astype(np.uint8)
        ns = gf2.nullspace(a)
        assert not gf2.matmul(a, ns.T).any()
        assert ns.shape[0] == 9 - gf2.rank(a)


def test_bipartite_edge_coloring_is_proper_and_optimal(rng):
    for spec in ("bb:72", "bb:144", "surface:5"):
        c = get_code(spec)
        for h in (c.hx, c.hz):
            edges = [(int(a), int(b)) for a, b in zip(*np.nonzero(h))]
            cols = bipartite_edge_coloring(edges)
            delta = max(h.sum(0).max(), h.sum(1).max())
            assert max(cols) + 1 == delta
            seen = set()
            for (u, v), col in zip(edges, cols):
                assert ("L", u, col) not in seen and ("R", v, col) not in seen
                seen.add(("L", u, col)); seen.add(("R", v, col))
