"""Trace-parity tests: DAngeloBatch (Numba) vs DAngelo2001CellModel (Python).

Numba replaced the earlier planned C++ port (see DESIGN.md, "Performance
strategy: Numba"). These tests are this project's equivalent of the
originally-planned C++-vs-Python parity harness: max absolute pointwise
voltage deviation over the run must stay under 0.1 mV.
"""
import numpy as np

from src.models.dangelo_cell import DAngelo2001CellModel
from src.models.dangelo_numba import DAngeloBatch

_DT = 0.025
_TOLERANCE_MV = 0.1


def _run_reference(duration_ms: float, I_nA: float) -> np.ndarray:
    cell = DAngelo2001CellModel()
    n_steps = round(duration_ms / _DT)
    V = np.empty(n_steps)
    for i in range(n_steps):
        cell.step(_DT, I_nA)
        V[i] = cell.get_voltage()
    return V


def _run_batch(n_nodes: int, duration_ms: float, I_nA: float) -> np.ndarray:
    batch = DAngeloBatch(n_nodes)
    n_steps = round(duration_ms / _DT)
    V = np.empty((n_steps, n_nodes))
    for i in range(n_steps):
        batch.step(_DT, I_nA)
        V[i, :] = batch.get_voltage()
    return V


class TestDAngeloBatchParity:

    def test_subthreshold_matches_reference(self):
        """Below spike threshold: batch trace must track the Python trace tightly."""
        ref = _run_reference(500.0, 0.005)
        batch = _run_batch(1, 500.0, 0.005)[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_MV, f"max deviation {max_dev:.4f} mV >= {_TOLERANCE_MV} mV"

    def test_repetitive_firing_matches_reference(self):
        """Suprathreshold, repetitive-firing regime: same <0.1 mV bar."""
        ref = _run_reference(500.0, 0.05)
        batch = _run_batch(1, 500.0, 0.05)[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_MV, f"max deviation {max_dev:.4f} mV >= {_TOLERANCE_MV} mV"

    def test_all_nodes_in_batch_evolve_identically(self):
        """Same input on every node -> every node's trace must be identical
        (guards against cross-node contamination in the parallel kernel)."""
        V = _run_batch(8, 200.0, 0.03)
        for j in range(1, V.shape[1]):
            assert np.array_equal(V[:, 0], V[:, j]), f"node {j} diverged from node 0"

    def test_reset_restores_initial_conditions(self):
        batch = DAngeloBatch(4)
        V_init = batch.get_voltage().copy()
        Ca_init = batch.get_calcium().copy()
        batch.step(_DT, 0.05)
        batch.reset()
        assert np.array_equal(batch.get_voltage(), V_init)
        assert np.array_equal(batch.get_calcium(), Ca_init)


class TestDAngeloBatchHeterogeneity:
    """Sou11-style per-node heterogeneity (DESIGN.md): gl_Lkg1,
    ggaba_Lkg2, membrane area, and initial V each independently drawn
    uniform +/-20% around their base value when heterogeneity_seed is set."""

    def test_no_seed_matches_prior_homogeneous_behavior(self):
        batch = DAngeloBatch(10)
        assert len(set(batch._area_cm2)) == 1
        assert len(set(batch._gl_Lkg1)) == 1
        assert len(set(batch._ggaba_Lkg2)) == 1
        assert len(set(batch.V)) == 1

    def test_seed_produces_real_per_node_spread_within_20_percent(self):
        batch = DAngeloBatch(200, heterogeneity_seed=1)
        base_area = np.pi * batch._p.diam * 1e-4 * batch._p.L * 1e-4
        assert batch._area_cm2.std() > 0
        assert batch._area_cm2.min() >= 0.8 * base_area - 1e-12
        assert batch._area_cm2.max() <= 1.2 * base_area + 1e-12
        assert batch._gl_Lkg1.min() >= 0.8 * batch._p.gl_Lkg1 - 1e-12
        assert batch._gl_Lkg1.max() <= 1.2 * batch._p.gl_Lkg1 + 1e-12
        assert batch._ggaba_Lkg2.min() >= 0.8 * batch._p.ggaba_Lkg2 - 1e-12
        assert batch._ggaba_Lkg2.max() <= 1.2 * batch._p.ggaba_Lkg2 + 1e-12
        assert batch.V.std() > 0

    def test_same_seed_reproducible(self):
        b1 = DAngeloBatch(20, heterogeneity_seed=42)
        b2 = DAngeloBatch(20, heterogeneity_seed=42)
        assert np.array_equal(b1._area_cm2, b2._area_cm2)
        assert np.array_equal(b1._gl_Lkg1, b2._gl_Lkg1)
        assert np.array_equal(b1._ggaba_Lkg2, b2._ggaba_Lkg2)
        assert np.array_equal(b1.V, b2.V)

    def test_different_seeds_differ(self):
        b1 = DAngeloBatch(20, heterogeneity_seed=1)
        b2 = DAngeloBatch(20, heterogeneity_seed=2)
        assert not np.array_equal(b1._area_cm2, b2._area_cm2)

    def test_reset_reproduces_identical_heterogeneous_initial_v(self):
        """reset() must be idempotent -- calling it twice must give the
        exact same (heterogeneous) initial V both times, not a fresh draw
        each call."""
        batch = DAngeloBatch(20, heterogeneity_seed=7)
        V_first = batch.V.copy()
        batch.step(_DT, 0.05)
        batch.reset()
        V_second = batch.V.copy()
        assert np.array_equal(V_first, V_second)

    def test_heterogeneous_batch_steps_without_crashing(self):
        batch = DAngeloBatch(50, heterogeneity_seed=3)
        for _ in range(100):
            batch.step(_DT, 0.05)
        assert np.all(np.isfinite(batch.get_voltage()))
