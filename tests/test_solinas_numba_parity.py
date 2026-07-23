"""Trace-parity tests: SolinasBatch (Numba) vs Solinas2007CellModel (Python).

Numba replaced the earlier planned C++ port (see DESIGN.md, "Performance
strategy: Numba"). These tests are this project's equivalent of the
originally-planned C++-vs-Python parity harness: max absolute pointwise
voltage deviation over the run must stay under 0.1 mV -- same bar as the
other three cell models.

The spontaneous pacemaker rate asserted here (1-8 Hz) matches
tests/test_solinas_cell.py's own validated band for the Python reference,
not the 5-15 Hz commonly cited in the literature -- see DESIGN.md's ionic
model roadmap table. SolinasBatch is a numerics-preserving port, not an
independent re-calibration, so it inherits whatever rate the reference
prototype actually produces.
"""
import numpy as np

from src.models.solinas_cell import Solinas2007CellModel
from src.models.solinas_numba import SolinasBatch

_DT = 0.025
_TOLERANCE_MV = 0.1


def _run_reference(duration_ms: float, I_nA: float) -> np.ndarray:
    cell = Solinas2007CellModel()
    n_steps = round(duration_ms / _DT)
    V = np.empty(n_steps)
    for i in range(n_steps):
        cell.step(_DT, I_nA)
        V[i] = cell.get_voltage()
    return V


def _run_batch(n_golgi: int, duration_ms: float, I_nA) -> np.ndarray:
    batch = SolinasBatch(n_golgi)
    n_steps = round(duration_ms / _DT)
    V = np.empty((n_steps, n_golgi))
    for i in range(n_steps):
        batch.step(_DT, I_nA)
        V[i, :] = batch.get_voltage()
    return V


def _count_spikes(V: np.ndarray, threshold: float = 0.0) -> int:
    above = V > threshold
    return int(np.sum(above[1:] & ~above[:-1]))


class TestSolinasBatchParity:

    def test_spontaneous_pacemaker_matches_reference(self):
        """Zero input, spontaneous firing: batch trace must track the
        Python trace tightly."""
        ref = _run_reference(1000.0, 0.0)
        batch = _run_batch(1, 1000.0, 0.0)[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_MV, f"max deviation {max_dev:.4f} mV >= {_TOLERANCE_MV} mV"

    def test_depolarising_input_matches_reference(self):
        """Suprathreshold depolarising drive: same <0.1 mV bar."""
        ref = _run_reference(500.0, 0.05)
        batch = _run_batch(1, 500.0, 0.05)[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_MV, f"max deviation {max_dev:.4f} mV >= {_TOLERANCE_MV} mV"

    def test_all_nodes_in_batch_evolve_identically(self):
        """Same input on every node -> every node's trace must be identical
        (guards against cross-node contamination in the parallel kernel)."""
        V = _run_batch(8, 500.0, 0.0)
        for j in range(1, V.shape[1]):
            assert np.array_equal(V[:, 0], V[:, j]), f"node {j} diverged from node 0"

    def test_reset_restores_initial_conditions(self):
        batch = SolinasBatch(4)
        V_init = batch.get_voltage().copy()
        Ca_init = batch.get_calcium().copy()
        batch.step(_DT, 0.05)
        batch.reset()
        assert np.array_equal(batch.get_voltage(), V_init)
        assert np.array_equal(batch.get_calcium(), Ca_init)


class TestSolinasBatchFiringRate:
    """Sanity check on SolinasBatch directly (not just parity) -- the same
    1-8 Hz spontaneous band tests/test_solinas_cell.py validates for the
    Python reference (see module docstring for why not 5-15 Hz)."""

    def test_spontaneous_pacemaker_firing_rate(self):
        batch = SolinasBatch(1)
        n_prestim = round(1000.0 / _DT)
        for _ in range(n_prestim):
            batch.step(_DT, 0.0)

        n_measure = round(2000.0 / _DT)
        V = np.empty(n_measure)
        for i in range(n_measure):
            batch.step(_DT, 0.0)
            V[i] = batch.get_voltage()[0]

        freq = _count_spikes(V) / 2.0  # spikes per second over a 2s window
        assert 1 <= freq <= 8, f"Pacemaker freq = {freq:.1f} Hz, expected 1-8 Hz"


class TestSolinasBatchHeterogeneity:
    """Sou11-style per-cell heterogeneity (DESIGN.md): glbar_lkg,
    membrane area, and initial V each independently drawn uniform +/-20%
    around their base value when heterogeneity_seed is set."""

    def test_no_seed_matches_prior_homogeneous_behavior(self):
        batch = SolinasBatch(10)
        assert len(set(batch._area_cm2)) == 1
        assert len(set(batch._glbar_lkg)) == 1
        assert len(set(batch.V)) == 1

    def test_seed_produces_real_per_cell_spread_within_20_percent(self):
        batch = SolinasBatch(200, heterogeneity_seed=1)
        base_area = np.pi * batch._p.diam * 1e-4 * batch._p.L * 1e-4
        assert batch._area_cm2.std() > 0
        assert batch._area_cm2.min() >= 0.8 * base_area - 1e-12
        assert batch._area_cm2.max() <= 1.2 * base_area + 1e-12
        assert batch._glbar_lkg.min() >= 0.8 * batch._p.glbar_lkg - 1e-12
        assert batch._glbar_lkg.max() <= 1.2 * batch._p.glbar_lkg + 1e-12
        assert batch.V.std() > 0

    def test_same_seed_reproducible(self):
        b1 = SolinasBatch(20, heterogeneity_seed=42)
        b2 = SolinasBatch(20, heterogeneity_seed=42)
        assert np.array_equal(b1._area_cm2, b2._area_cm2)
        assert np.array_equal(b1._glbar_lkg, b2._glbar_lkg)
        assert np.array_equal(b1.V, b2.V)

    def test_heterogeneous_batch_steps_without_crashing(self):
        batch = SolinasBatch(50, heterogeneity_seed=3)
        for _ in range(100):
            batch.step(_DT, 0.0)
        assert np.all(np.isfinite(batch.get_voltage()))
