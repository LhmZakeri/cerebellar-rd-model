"""
NodeBatch: co-locates one granule (D'Angelo), one Purkinje (Fernandez), and
one stellate (Molineux) cell per node, connected vertically by three
synapses -- granule->Purkinje (excitatory), stellate->Purkinje (inhibitory),
and Purkinje->stellate (also inhibitory -- a Purkinje-axon-collateral-style
GABAergic feedback onto the molecular-layer interneuron, Erev ~-82mV, not
excitatory as originally coded here; see DESIGN.md's Node design for the
resulting reciprocal-inhibition loop: Purkinje inhibits stellate, stellate
inhibits Purkinje back -- mutual inhibition, not excitation/rebound-
inhibition). Operates on arrays of N nodes at once, same as the underlying
*Batch classes.

Scope: three cell models + three synapse models, no Golgi (deferred, see
DESIGN.md), no spatial coupling between nodes here -- Golgi
and the remaining Node connectivity (Golgi<->granule synapses, Golgi<->Golgi
diffusion) live on GridNodeBatch (DESIGN.md), which folds
its Golgi->granule inhibitory current into step()'s extra_granule_I_nA hook
rather than duplicating this method's orchestration.

Units: I_mossy_nA and I_climbing_nA are nA, matching the project-wide units
contract. Synapse get_current() returns pA (destexhe_numba.py docstring);
_PA_TO_NA converts that back to the nA every CellModel.step() expects.
"""
from __future__ import annotations

import numpy as np

from src.models.dangelo_numba import DAngeloBatch
from src.models.fernandez_numba import FernandezBatch
from src.models.molineux_numba import MolineuxBatch
from src.models.destexhe_numba import TwoStateDestexheBatch
from src.models.dangelo_cell import DAngelo2001Params
from src.models.fernandez_cell import Fernandez2007Params
from src.models.molineux_cell import MolineuxStellateParams

_PA_TO_NA = 1e-3


class NodeBatch:
    """N independent Nodes: granule + Purkinje + stellate + three vertical synapses.

    granule -> excToPurkinje -> Purkinje        (excitatory, AMPA-like)
    stellate -> inhToPurkinje -> Purkinje       (inhibitory, GABA-A-like)
    Purkinje -> inhPurkinjeToStellate -> stellate (inhibitory, GABA-A-like,
        Erev ~-82mV -- Purkinje axon collaterals inhibit molecular-layer
        interneurons; not excitatory, corrected from this class's original
        miscoded version)

    The last synapse closes a reciprocal-inhibition loop: whenever Purkinje
    is driven (by climbing fiber and/or granule excitation), it inhibits
    stellate in turn, while stellate's own inhibitory synapse onto Purkinje
    feeds back the other way -- both directions are inhibitory, not a
    classic excitation/rebound-inhibition motif.

    Mossy-fiber drive reaches only `granule`; climbing-fiber drive reaches
    only `purkinje` -- separate named methods, not one generic target-
    parameterized input, per DESIGN.md's Node design (matters for testing
    whether climbing-fiber synchrony loss reproduces tremor/ataxia-like
    dynamics).
    """

    def __init__(
        self,
        n_nodes: int,
        granule_params: DAngelo2001Params | None = None,
        purkinje_params: Fernandez2007Params | None = None,
        stellate_params: MolineuxStellateParams | None = None,
        gmax_exc: float = 1.0,
        gmax_inh: float = 1.0,
        gmax_ps: float = 1.0,
        heterogeneity_seed: int | None = None,
    ) -> None:
        """heterogeneity_seed: None (default) -> every cell identical, exact
        prior behavior. An int -> Sou11-style per-cell heterogeneity
        (DESIGN.md) for granule and Purkinje (stellate is out of scope --
        Sou11's model doesn't include it, and it wasn't asked for). Granule
        and Purkinje draw from distinct offset seeds, not the same seed
        reused, so their randomness doesn't correlate -- same rationale as
        every other multi-source-of-randomness seed derivation in this repo
        (e.g. GridNodeBatch's connectivity_seed + 1_000_003)."""
        self.n_nodes = n_nodes
        granule_seed = heterogeneity_seed
        purkinje_seed = None if heterogeneity_seed is None else heterogeneity_seed + 500_000_007
        self.granule = DAngeloBatch(n_nodes, granule_params, heterogeneity_seed=granule_seed)
        self.purkinje = FernandezBatch(n_nodes, purkinje_params, heterogeneity_seed=purkinje_seed)
        self.stellate = MolineuxBatch(n_nodes, stellate_params)
        self.exc_to_purkinje = TwoStateDestexheBatch.excitatory(n_nodes, gmax=gmax_exc)
        self.inh_to_purkinje = TwoStateDestexheBatch.inhibitory(n_nodes, gmax=gmax_inh)
        self.inh_purkinje_to_stellate = TwoStateDestexheBatch.inhibitory(
            n_nodes, gmax=gmax_ps, Erev=-82.0
        )

        self._I_mossy = np.zeros(n_nodes, dtype=np.float64)
        self._I_climbing = np.zeros(n_nodes, dtype=np.float64)

    def reset(self) -> None:
        self.granule.reset()
        self.purkinje.reset()
        self.stellate.reset()
        self.exc_to_purkinje.reset()
        self.inh_to_purkinje.reset()
        self.inh_purkinje_to_stellate.reset()
        self._I_mossy = np.zeros(self.n_nodes, dtype=np.float64)
        self._I_climbing = np.zeros(self.n_nodes, dtype=np.float64)

    def inject_mossy_fiber_input(self, I_nA) -> None:
        """Set the tonic mossy-fiber drive -> granule. Persists across step() until changed."""
        self._I_mossy = self._as_array(I_nA)

    def inject_climbing_fiber_input(self, I_nA) -> None:
        """Set the climbing-fiber drive -> Purkinje. Persists across step() until changed."""
        self._I_climbing = self._as_array(I_nA)

    def step(
        self, dt: float,
        extra_granule_I_nA=0.0, extra_purkinje_I_nA=0.0, extra_stellate_I_nA=0.0,
    ) -> None:
        """Advance every sub-model by one dt [ms].

        extra_granule_I_nA / extra_purkinje_I_nA / extra_stellate_I_nA:
        additional current [nA] added to each population's own drive,
        default 0.0 (a true no-op for every caller that doesn't pass them).
        These are the hooks GridNodeBatch uses to fold in Golgi<->granule
        synaptic current and the convergent granule/Purkinje/stellate
        synapses (DESIGN.md) without duplicating this
        method's vertical-synapse orchestration -- the three existing
        synapses below are otherwise untouched.
        """
        purkinje_V = self.purkinje.get_voltage()
        stellate_V = self.stellate.get_voltage()

        # Synapse kinetics respond to their presynaptic cell's current voltage.
        self.exc_to_purkinje.step(dt, self.granule.get_voltage())
        self.inh_to_purkinje.step(dt, stellate_V)
        self.inh_purkinje_to_stellate.step(dt, purkinje_V)

        # get_current() returns an ohmic ionic-style current (I = g*(V-Erev)),
        # same sign convention as the ionic currents inside FernandezBatch's
        # and MolineuxBatch's own RHS -- so it's subtracted from the injected
        # drive, not added.
        I_exc_nA = self.exc_to_purkinje.get_current(purkinje_V) * _PA_TO_NA
        I_inh_nA = self.inh_to_purkinje.get_current(purkinje_V) * _PA_TO_NA
        I_inh_stellate_nA = self.inh_purkinje_to_stellate.get_current(stellate_V) * _PA_TO_NA

        self.granule.step(dt, self._I_mossy + extra_granule_I_nA)
        self.stellate.step(dt, -I_inh_stellate_nA + extra_stellate_I_nA)
        self.purkinje.step(dt, self._I_climbing - I_exc_nA - I_inh_nA + extra_purkinje_I_nA)

    def _as_array(self, I_nA) -> np.ndarray:
        if np.isscalar(I_nA):
            return np.full(self.n_nodes, I_nA, dtype=np.float64)
        return np.asarray(I_nA, dtype=np.float64)
