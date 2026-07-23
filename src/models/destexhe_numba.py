"""
Numba-accelerated batch kernel for the Destexhe, Mainen & Sejnowski (1994)
two-state kinetic synapse.

Operates on arrays of N synapses at once, sharing one set of kinetic
parameters (TwoStateDestexheParams) across the whole batch -- matching the
single-node TwoStateDestexhe class's own closed-form solution exactly, so
there is no duplicated numerics. See DESIGN.md ("Performance strategy:
Numba") and src/models/dangelo_numba.py for the same pattern applied to
the ionic cell models.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from src.models.destexhe_synapse import TwoStateDestexheParams


@njit(parallel=True, cache=True, fastmath=False)
def _step_all(R, C, R0, R1, lastrelease, TimeCount, t, V_pre, dt,
              Cmax, Cdur, Alpha, Beta, Erev, Prethresh, Deadtime, Rinf, Rtau):
    """Advance every synapse by one dt. Mutates all state arrays in place."""
    n = R.shape[0]
    for i in prange(n):
        t[i] += dt
        TimeCount[i] -= dt

        if TimeCount[i] < -Deadtime:
            if V_pre[i] > Prethresh:
                C[i] = Cmax
                R0[i] = R[i]
                lastrelease[i] = t[i]
                TimeCount[i] = Cdur
        elif TimeCount[i] > 0:
            pass  # still releasing
        elif C[i] == Cmax:
            R1[i] = R[i]
            C[i] = 0.0

        if C[i] > 0:
            R[i] = Rinf + (R0[i] - Rinf) * np.exp(-(t[i] - lastrelease[i]) / Rtau)
        else:
            R[i] = R1[i] * np.exp(-Beta * (t[i] - (lastrelease[i] + Cdur)))


class TwoStateDestexheBatch:
    """N independent two-state kinetic synapses, stepped via one Numba kernel.

    Same math as TwoStateDestexhe (destexhe_synapse.py) -- this is not a
    reimplementation, it shares that module's TwoStateDestexheParams
    dataclass and closed-form solution exactly. Use this when N is large
    enough that the per-node Numba speedup matters; use TwoStateDestexhe
    for single-synapse work.
    """

    def __init__(self, n_nodes: int, params: TwoStateDestexheParams) -> None:
        self.n_nodes = n_nodes
        self._p = params
        self.reset()

    def reset(self) -> None:
        p = self._p
        self.R = np.zeros(self.n_nodes, dtype=np.float64)
        self.C = np.zeros(self.n_nodes, dtype=np.float64)
        self.R0 = np.zeros(self.n_nodes, dtype=np.float64)
        self.R1 = np.zeros(self.n_nodes, dtype=np.float64)
        self.lastrelease = np.full(self.n_nodes, -1000.0, dtype=np.float64)
        self.TimeCount = np.full(self.n_nodes, -1.0, dtype=np.float64)
        self.t = np.zeros(self.n_nodes, dtype=np.float64)
        self.Rinf = p.Cmax * p.Alpha / (p.Cmax * p.Alpha + p.Beta)
        self.Rtau = 1.0 / (p.Alpha * p.Cmax + p.Beta)

    def step(self, dt: float, V_pre) -> None:
        """Advance all synapses by dt [ms]. V_pre: scalar or per-node array [mV]."""
        if np.isscalar(V_pre):
            V_pre_arr = np.full(self.n_nodes, V_pre, dtype=np.float64)
        else:
            V_pre_arr = np.asarray(V_pre, dtype=np.float64)

        p = self._p
        _step_all(
            self.R, self.C, self.R0, self.R1, self.lastrelease, self.TimeCount,
            self.t, V_pre_arr, dt,
            p.Cmax, p.Cdur, p.Alpha, p.Beta, p.Erev, p.Prethresh, p.Deadtime,
            self.Rinf, self.Rtau,
        )

    def get_current(self, V_post) -> np.ndarray:
        """I = gmax * R * (V_post - Erev)  [pA]."""
        return self._p.gmax * self.R * (np.asarray(V_post, dtype=np.float64) - self._p.Erev)

    def get_conductance(self) -> np.ndarray:
        """g = gmax * R  [nS]."""
        return self._p.gmax * self.R

    # --- Presets (published parameter values, unchanged) -----------------------

    @classmethod
    def excitatory(cls, n_nodes: int, gmax: float) -> "TwoStateDestexheBatch":
        """AMPA-like preset (ampa.mod defaults): granular -> Purkinje, Erev = 0 mV."""
        return cls(n_nodes, TwoStateDestexheParams(gmax=gmax))

    @classmethod
    def inhibitory(cls, n_nodes: int, gmax: float, Erev: float = -80.0) -> "TwoStateDestexheBatch":
        """GABA-A-like preset (gabaa.mod defaults): molecular -> Purkinje, Erev = -80 mV
        by default -- pass Erev to override for a different GABA-A pathway
        (e.g. Purkinje -> stellate, Erev ~-82 mV, node_batch.py)."""
        return cls(n_nodes, TwoStateDestexheParams(gmax=gmax, Alpha=5.0, Beta=0.18, Erev=Erev))
