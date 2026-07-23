"""Trace-parity tests: TwoStateDestexheBatch (Numba) vs TwoStateDestexhe (Python).

Numba replaced the earlier planned C++ port (see DESIGN.md, "Performance
strategy: Numba"). Unlike the ODE cell models, this synapse steps via a
closed-form analytic solution (no integrator), so both implementations
should match to near machine precision, not just <0.1 mV -- a much
tighter tolerance is used here accordingly.
"""
import numpy as np

from src.models.destexhe_synapse import TwoStateDestexhe
from src.models.destexhe_numba import TwoStateDestexheBatch

_DT = 0.025
_TOLERANCE_R = 1e-9

REST_V = -70.0
SPIKE_V = 30.0


def _run_reference(factory, pulse_ms=1.0, post_ms=60.0) -> np.ndarray:
    syn = factory()
    n_pulse = round(pulse_ms / _DT)
    n_post = round(post_ms / _DT)
    R = np.empty(n_pulse + n_post)
    k = 0
    for _ in range(n_pulse):
        syn.step(_DT, SPIKE_V)
        R[k] = syn.R
        k += 1
    for _ in range(n_post):
        syn.step(_DT, REST_V)
        R[k] = syn.R
        k += 1
    return R


def _run_batch(batch_factory, n_nodes, pulse_ms=1.0, post_ms=60.0) -> np.ndarray:
    batch = batch_factory(n_nodes)
    n_pulse = round(pulse_ms / _DT)
    n_post = round(post_ms / _DT)
    R = np.empty((n_pulse + n_post, n_nodes))
    k = 0
    for _ in range(n_pulse):
        batch.step(_DT, SPIKE_V)
        R[k, :] = batch.R
        k += 1
    for _ in range(n_post):
        batch.step(_DT, REST_V)
        R[k, :] = batch.R
        k += 1
    return R


class TestTwoStateDestexheBatchParity:

    def test_excitatory_pulse_matches_reference(self):
        ref = _run_reference(lambda: TwoStateDestexhe.excitatory(gmax=1.0))
        batch = _run_batch(lambda n: TwoStateDestexheBatch.excitatory(n, gmax=1.0), 1)[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_R, f"max deviation {max_dev:.2e} >= {_TOLERANCE_R}"

    def test_inhibitory_pulse_matches_reference(self):
        ref = _run_reference(lambda: TwoStateDestexhe.inhibitory(gmax=1.0))
        batch = _run_batch(lambda n: TwoStateDestexheBatch.inhibitory(n, gmax=1.0), 1)[:, 0]
        max_dev = np.max(np.abs(ref - batch))
        assert max_dev < _TOLERANCE_R, f"max deviation {max_dev:.2e} >= {_TOLERANCE_R}"

    def test_all_nodes_in_batch_evolve_identically(self):
        R = _run_batch(lambda n: TwoStateDestexheBatch.excitatory(n, gmax=1.0), 8)
        for j in range(1, R.shape[1]):
            assert np.array_equal(R[:, 0], R[:, j]), f"node {j} diverged from node 0"

    def test_reset_restores_initial_conditions(self):
        batch = TwoStateDestexheBatch.excitatory(4, gmax=1.0)
        batch.step(_DT, SPIKE_V)
        batch.reset()
        assert np.all(batch.R == 0.0)
        assert np.all(batch.C == 0.0)
        assert np.all(batch.TimeCount == -1.0)
        assert np.all(batch.lastrelease == -1000.0)

    def test_current_direction_matches_reversal_potential(self):
        batch = TwoStateDestexheBatch.excitatory(4, gmax=1.0)
        for _ in range(round(1.0 / _DT)):
            batch.step(_DT, SPIKE_V)
        assert np.all(batch.R > 0.0)
        p = batch._p
        assert np.all(batch.get_current(p.Erev + 10.0) > 0.0)
        assert np.all(batch.get_current(p.Erev - 10.0) < 0.0)
