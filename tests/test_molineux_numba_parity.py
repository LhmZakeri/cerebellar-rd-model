"""Trace-parity tests: MolineuxBatch (Numba) vs MolineuxStellateCellModel (Python).

Numba replaced the earlier planned C++ port (see DESIGN.md, "Performance
strategy: Numba"). Max absolute pointwise voltage deviation over the run
must stay under 0.1 mV, same bar the C++ plan used.
"""
import numpy as np

from src.models.molineux_cell import MolineuxStellateCellModel
from src.models.molineux_numba import MolineuxBatch

_DT = 0.025
_TOLERANCE_MV = 0.1
_SUBTHRESHOLD_NA = 0.0003
_SUPRATHRESHOLD_NA = 0.01


def _run_reference(duration_ms: float, I_nA: float) -> np.ndarray:
    cell = MolineuxStellateCellModel()
    n_steps = round(duration_ms / _DT)
    V = np.empty(n_steps)
    for i in range(n_steps):
        cell.step(_DT, I_nA)
        V[i] = cell.get_voltage()
    return V


def _run_batch(n_nodes: int, duration_ms: float, I_nA: float) -> np.ndarray:
    batch = MolineuxBatch(n_nodes)
    n_steps = round(duration_ms / _DT)
    V = np.empty((n_steps, n_nodes))
    for i in range(n_steps):
        batch.step(_DT, I_nA)
        V[i, :] = batch.get_voltage()
    return V


class TestMolineuxBatchParity:

    def test_subthreshold_matches_reference(self):
        ref = _run_reference(200.0, _SUBTHRESHOLD_NA)
        batch = _run_batch(1, 200.0, _SUBTHRESHOLD_NA)[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_MV, f"max deviation {max_dev:.4f} mV >= {_TOLERANCE_MV} mV"

    def test_suprathreshold_firing_matches_reference(self):
        ref = _run_reference(200.0, _SUPRATHRESHOLD_NA)
        batch = _run_batch(1, 200.0, _SUPRATHRESHOLD_NA)[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_MV, f"max deviation {max_dev:.4f} mV >= {_TOLERANCE_MV} mV"

    def test_all_nodes_in_batch_evolve_identically(self):
        V = _run_batch(8, 100.0, _SUPRATHRESHOLD_NA)
        for j in range(1, V.shape[1]):
            assert np.array_equal(V[:, 0], V[:, j]), f"node {j} diverged from node 0"

    def test_reset_restores_initial_conditions(self):
        batch = MolineuxBatch(4)
        V_init = batch.get_voltage().copy()
        batch.step(_DT, _SUPRATHRESHOLD_NA)
        batch.reset()
        assert np.array_equal(batch.get_voltage(), V_init)
