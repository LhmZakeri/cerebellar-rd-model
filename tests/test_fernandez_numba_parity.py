"""Trace-parity tests: FernandezBatch (Numba) vs Fernandez2007CellModel (Python).

Numba replaced the earlier planned C++ port (see DESIGN.md, "Performance
strategy: Numba"). Max absolute pointwise voltage deviation over the run
must stay under 0.1 mV, same bar the C++ plan used.

The two-CF-pulse toggle protocol below is the canonical stimulus for
"bistable pause after climbing-fiber pulse" (see CONTEXT.md, "Bistable
toggle (Fernandez Purkinje)") -- taken directly from
scripts/run_fernandez2007_prototype.py's
test_fig4c_two_intermediate_pulses_toggle_on_then_off.
"""
import numpy as np

from src.models.fernandez_cell import Fernandez2007CellModel
from src.models.fernandez_numba import FernandezBatch

_DT = 0.01
_TOLERANCE_MV = 0.1

_TOGGLE_SEGMENTS = [
    (100.0, 0.0), (15.0, 4200.0), (500.0, 0.0), (15.0, 4200.0), (100.0, 0.0),
]


def _run_reference(segments) -> np.ndarray:
    cell = Fernandez2007CellModel()
    n_total = sum(round(d / _DT) for d, _ in segments)
    V = np.empty(n_total)
    k = 0
    for duration_ms, current_nA in segments:
        for _ in range(round(duration_ms / _DT)):
            cell.step(_DT, current_nA)
            V[k] = cell.get_voltage()
            k += 1
    return V


def _run_batch(n_nodes: int, segments) -> np.ndarray:
    batch = FernandezBatch(n_nodes)
    n_total = sum(round(d / _DT) for d, _ in segments)
    V = np.empty((n_total, n_nodes))
    k = 0
    for duration_ms, current_nA in segments:
        for _ in range(round(duration_ms / _DT)):
            batch.step(_DT, current_nA)
            V[k, :] = batch.get_voltage()
            k += 1
    return V


class TestFernandezBatchParity:

    def test_subthreshold_matches_reference(self):
        ref = _run_reference([(500.0, 0.0)])
        batch = _run_batch(1, [(500.0, 0.0)])[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_MV, f"max deviation {max_dev:.4f} mV >= {_TOLERANCE_MV} mV"

    def test_bistable_toggle_protocol_matches_reference(self):
        """The exact two-CF-pulse toggle stimulus (CONTEXT.md's canonical protocol)."""
        ref = _run_reference(_TOGGLE_SEGMENTS)
        batch = _run_batch(1, _TOGGLE_SEGMENTS)[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_MV, f"max deviation {max_dev:.4f} mV >= {_TOLERANCE_MV} mV"

    def test_all_nodes_in_batch_evolve_identically(self):
        V = _run_batch(8, [(200.0, 4200.0)])
        for j in range(1, V.shape[1]):
            assert np.array_equal(V[:, 0], V[:, j]), f"node {j} diverged from node 0"

    def test_reset_restores_initial_conditions(self):
        batch = FernandezBatch(4)
        V_init = batch.get_voltage().copy()
        batch.step(_DT, 4200.0)
        batch.reset()
        assert np.array_equal(batch.get_voltage(), V_init)


class TestFernandezBatchHeterogeneity:
    """Sou11-style per-node heterogeneity (DESIGN.md): g_leak and
    initial V_s/V_d each independently drawn uniform +/-20% around their
    base value when heterogeneity_seed is set. NO membrane-area
    heterogeneity -- this model has no area parameter at all (current
    density units, no defined cell area, per this module's own docstring)."""

    def test_no_seed_matches_prior_homogeneous_behavior(self):
        batch = FernandezBatch(10)
        assert len(set(batch._g_leak)) == 1
        assert len(set(batch.V_s)) == 1
        assert len(set(batch.V_d)) == 1
        assert not hasattr(batch, "_area_cm2")

    def test_seed_produces_real_per_node_spread_within_20_percent(self):
        batch = FernandezBatch(200, heterogeneity_seed=1)
        assert batch._g_leak.std() > 0
        assert batch._g_leak.min() >= 0.8 * batch._p.g_leak - 1e-12
        assert batch._g_leak.max() <= 1.2 * batch._p.g_leak + 1e-12
        assert batch.V_s.std() > 0
        assert batch.V_d.std() > 0

    def test_same_seed_reproducible(self):
        b1 = FernandezBatch(20, heterogeneity_seed=42)
        b2 = FernandezBatch(20, heterogeneity_seed=42)
        assert np.array_equal(b1._g_leak, b2._g_leak)
        assert np.array_equal(b1.V_s, b2.V_s)
        assert np.array_equal(b1.V_d, b2.V_d)

    def test_reset_reproduces_identical_heterogeneous_initial_v(self):
        batch = FernandezBatch(20, heterogeneity_seed=7)
        Vs_first, Vd_first = batch.V_s.copy(), batch.V_d.copy()
        batch.step(_DT, 4200.0)
        batch.reset()
        assert np.array_equal(Vs_first, batch.V_s)
        assert np.array_equal(Vd_first, batch.V_d)

    def test_heterogeneous_batch_steps_without_crashing(self):
        batch = FernandezBatch(50, heterogeneity_seed=3)
        for _ in range(100):
            batch.step(_DT, 4200.0)
        assert np.all(np.isfinite(batch.get_voltage()))
