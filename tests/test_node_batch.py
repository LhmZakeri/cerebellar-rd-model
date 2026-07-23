"""NodeBatch: instantiation + injectInput routing tests.

Direct port of issue #6's original acceptance criteria ("Single Node with
3 cell models + 3 synapse models instantiates correctly", "injectInput()
routes I_mossy to granularCell, I_climbing to purkinjeCell") onto the
Numba/NodeBatch architecture -- see DESIGN.md, "Planned Node design". Also
covers the Purkinje->stellate feedback synapse added after the original
acceptance criteria (Purkinje inhibits stellate via GABAergic axon
collaterals, Erev ~-82mV; stellate's existing inhibitory synapse feeds back
onto Purkinje the other way -- reciprocal inhibition, not
excitation/rebound-inhibition).
"""
import numpy as np

from src.simulation.node_batch import NodeBatch
from src.models.dangelo_numba import DAngeloBatch
from src.models.fernandez_numba import FernandezBatch
from src.models.molineux_numba import MolineuxBatch

_DT = 0.01
_N_NODES = 8


class TestNodeBatchInstantiation:

    def test_instantiates_with_correct_submodel_types_and_sizes(self):
        node = NodeBatch(_N_NODES)
        assert isinstance(node.granule, DAngeloBatch)
        assert isinstance(node.purkinje, FernandezBatch)
        assert isinstance(node.stellate, MolineuxBatch)
        assert node.granule.n_nodes == _N_NODES
        assert node.purkinje.n_nodes == _N_NODES
        assert node.stellate.n_nodes == _N_NODES
        assert node.exc_to_purkinje.n_nodes == _N_NODES
        assert node.inh_to_purkinje.n_nodes == _N_NODES
        assert node.inh_purkinje_to_stellate.n_nodes == _N_NODES

    def test_synapse_presets_are_excitatory_and_inhibitory(self):
        node = NodeBatch(_N_NODES)
        assert node.exc_to_purkinje._p.Erev == 0.0
        assert node.inh_to_purkinje._p.Erev == -80.0
        assert node.inh_purkinje_to_stellate._p.Erev == -82.0

    def test_step_runs_without_error(self):
        node = NodeBatch(_N_NODES)
        node.inject_mossy_fiber_input(0.02)
        node.inject_climbing_fiber_input(0.0)
        for _ in range(100):
            node.step(_DT)
        assert node.granule.get_voltage().shape == (_N_NODES,)

    def test_reset_restores_initial_conditions(self):
        node = NodeBatch(_N_NODES)
        V_g0 = node.granule.get_voltage().copy()
        V_p0 = node.purkinje.get_voltage().copy()
        node.inject_mossy_fiber_input(0.05)
        for _ in range(50):
            node.step(_DT)
        node.reset()
        assert np.array_equal(node.granule.get_voltage(), V_g0)
        assert np.array_equal(node.purkinje.get_voltage(), V_p0)
        assert np.all(node._I_mossy == 0.0)
        assert np.all(node._I_climbing == 0.0)


class TestInjectInputRouting:

    def test_mossy_fiber_reaches_only_granule(self):
        """granule inside NodeBatch must evolve identically to a bare
        DAngeloBatch fed the same current -- granule gets no synaptic
        feedback in this Node design, so there's nothing to make it
        diverge unless the wiring is wrong."""
        node = NodeBatch(_N_NODES)
        node.inject_mossy_fiber_input(0.05)
        node.inject_climbing_fiber_input(0.0)

        reference = DAngeloBatch(_N_NODES)

        for _ in range(200):
            node.step(_DT)
            reference.step(_DT, 0.05)

        assert np.array_equal(node.granule.get_voltage(), reference.get_voltage())

    def test_stellate_unreached_when_purkinje_silent(self):
        """stellate's only input pathway is via Purkinje's inhibitory
        feedback synapse -- with Purkinje's synapses disabled (gmax=0) and
        no climbing-fiber drive, Purkinje never fires, so stellate must
        match a bare, undriven MolineuxBatch even under strong mossy-fiber
        drive to granule (which has no direct route to stellate)."""
        node = NodeBatch(_N_NODES, gmax_exc=0.0, gmax_inh=0.0, gmax_ps=0.0)
        node.inject_mossy_fiber_input(0.05)  # strong enough to make granule fire
        node.inject_climbing_fiber_input(0.0)

        reference = MolineuxBatch(_N_NODES)

        for _ in range(200):
            node.step(_DT)
            reference.step(_DT, 0.0)

        assert np.array_equal(node.stellate.get_voltage(), reference.get_voltage())

    def test_purkinje_stimulation_drives_stellate_via_inhibitory_synapse(self):
        """Sanity check that the feedback pathway is live: with default
        (nonzero) inh_purkinje_to_stellate gmax and strong climbing-fiber
        drive (Purkinje fires), stellate's trace must diverge from an
        undriven baseline."""
        node = NodeBatch(_N_NODES)
        node.inject_mossy_fiber_input(0.0)
        node.inject_climbing_fiber_input(4200.0)

        undriven = MolineuxBatch(_N_NODES)

        for _ in range(2_000):  # 20 ms
            node.step(_DT)
            undriven.step(_DT, 0.0)

        assert not np.array_equal(node.stellate.get_voltage(), undriven.get_voltage())

    def test_purkinje_to_stellate_synapse_is_hyperpolarizing_not_depolarizing(self):
        """Regression test for the Purkinje->stellate sign fix: this pathway
        is GABAergic (inhibitory, Erev ~-82mV), not excitatory as originally
        coded -- driven stellate voltage must end up BELOW an undriven
        baseline, not merely different from it."""
        node = NodeBatch(_N_NODES)
        node.inject_mossy_fiber_input(0.0)
        node.inject_climbing_fiber_input(4200.0)

        undriven = MolineuxBatch(_N_NODES)

        for _ in range(2_000):  # 20 ms
            node.step(_DT)
            undriven.step(_DT, 0.0)

        assert np.all(node.stellate.get_voltage() < undriven.get_voltage())

    def test_climbing_fiber_reaches_purkinje_with_synapses_disabled(self):
        """With synaptic coupling turned off (gmax=0), purkinje must
        evolve identically to a bare FernandezBatch fed the same
        climbing-fiber current -- isolates the injectClimbingFiberInput
        routing from the (separately tested) synaptic coupling."""
        node = NodeBatch(_N_NODES, gmax_exc=0.0, gmax_inh=0.0, gmax_ps=0.0)
        node.inject_mossy_fiber_input(0.0)
        node.inject_climbing_fiber_input(4200.0)

        reference = FernandezBatch(_N_NODES)

        for _ in range(200):
            node.step(_DT)
            reference.step(_DT, 4200.0)

        assert np.array_equal(node.purkinje.get_voltage(), reference.get_voltage())

    def test_granule_firing_drives_purkinje_via_excitatory_synapse(self):
        """Sanity check that the coupling pathway is actually live: with
        default (nonzero) synapse gmax and strong mossy-fiber drive
        (granule fires repetitively -- takes ~100+ ms from rest to reach
        threshold, see D'Angelo's own settling convention in
        test_dangelo_cell.py), purkinje's trace must diverge from an
        undriven baseline -- proving synaptic current is really reaching
        it, not just silently wired to zero."""
        node = NodeBatch(_N_NODES)
        node.inject_mossy_fiber_input(0.05)
        node.inject_climbing_fiber_input(0.0)

        undriven = FernandezBatch(_N_NODES)

        for _ in range(20_000):  # 200 ms
            node.step(_DT)
            undriven.step(_DT, 0.0)

        assert not np.array_equal(node.purkinje.get_voltage(), undriven.get_voltage())

    def test_extra_granule_current_defaults_to_zero_no_behavior_change(self):
        """extra_granule_I_nA (the hook GridNodeBatch uses to fold in
        Golgi->granule inhibitory current, DESIGN.md) must be a true
        no-op when omitted -- every existing caller of step(dt) is
        unaffected by its addition."""
        with_default = NodeBatch(_N_NODES)
        without_arg = NodeBatch(_N_NODES)
        with_default.inject_mossy_fiber_input(0.05)
        without_arg.inject_mossy_fiber_input(0.05)

        for _ in range(200):
            with_default.step(_DT, extra_granule_I_nA=0.0)
            without_arg.step(_DT)

        assert np.array_equal(with_default.granule.get_voltage(), without_arg.granule.get_voltage())

    def test_extra_granule_current_is_additive_and_live(self):
        """A nonzero extra_granule_I_nA must change granule's trace relative
        to omitting it -- proving the hook actually reaches granule.step(),
        not just existing as an unused parameter."""
        driven = NodeBatch(_N_NODES)
        undriven = NodeBatch(_N_NODES)
        driven.inject_mossy_fiber_input(0.0)
        undriven.inject_mossy_fiber_input(0.0)

        for _ in range(200):
            driven.step(_DT, extra_granule_I_nA=0.05)
            undriven.step(_DT)

        assert not np.array_equal(driven.granule.get_voltage(), undriven.granule.get_voltage())

    def test_extra_purkinje_and_stellate_current_default_to_zero_no_behavior_change(self):
        """extra_purkinje_I_nA / extra_stellate_I_nA (the hooks GridNodeBatch
        uses to fold in the convergent granule/Purkinje/stellate synapses,
        DESIGN.md) must be true no-ops when omitted."""
        with_default = NodeBatch(_N_NODES)
        without_arg = NodeBatch(_N_NODES)
        with_default.inject_climbing_fiber_input(0.02)
        without_arg.inject_climbing_fiber_input(0.02)

        for _ in range(200):
            with_default.step(_DT, extra_purkinje_I_nA=0.0, extra_stellate_I_nA=0.0)
            without_arg.step(_DT)

        assert np.array_equal(with_default.purkinje.get_voltage(), without_arg.purkinje.get_voltage())
        assert np.array_equal(with_default.stellate.get_voltage(), without_arg.stellate.get_voltage())

    def test_extra_purkinje_and_stellate_current_are_additive_and_live(self):
        """A nonzero extra_purkinje_I_nA/extra_stellate_I_nA must change
        their respective traces relative to omitting them."""
        driven = NodeBatch(_N_NODES)
        undriven = NodeBatch(_N_NODES)

        for _ in range(200):
            driven.step(_DT, extra_purkinje_I_nA=50.0, extra_stellate_I_nA=50.0)
            undriven.step(_DT)

        assert not np.array_equal(driven.purkinje.get_voltage(), undriven.purkinje.get_voltage())
        assert not np.array_equal(driven.stellate.get_voltage(), undriven.stellate.get_voltage())


class TestHeterogeneity:
    """Sou11-style per-cell heterogeneity pass-through (DESIGN.md):
    NodeBatch routes heterogeneity_seed to granule and Purkinje (with
    distinct offset sub-seeds, so their randomness doesn't correlate) --
    stellate is out of scope, untouched regardless."""

    def test_no_seed_matches_prior_homogeneous_behavior(self):
        node = NodeBatch(_N_NODES)
        assert len(set(node.granule.V)) == 1
        assert len(set(node.purkinje.V_s)) == 1
        assert len(set(node.stellate.V)) == 1

    def test_seed_gives_granule_and_purkinje_real_spread_stellate_untouched(self):
        node = NodeBatch(50, heterogeneity_seed=1)
        assert node.granule.V.std() > 0
        assert node.purkinje.V_s.std() > 0
        assert len(set(node.stellate.V)) == 1

    def test_granule_and_purkinje_draw_independent_not_correlated_seeds(self):
        """Granule and Purkinje must NOT get the identical draw -- distinct
        offset sub-seeds, not the same seed reused (DESIGN.md)."""
        node = NodeBatch(50, heterogeneity_seed=1)
        # If they used the same seed, the two arrays' rank order would be
        # identical (both are uniform(0.8,1.2) draws from the same stream) --
        # compare sorted-argsort order rather than raw values, since the
        # arrays have different physical units/base values.
        granule_order = np.argsort(node.granule.V)
        purkinje_order = np.argsort(node.purkinje.V_s)
        assert not np.array_equal(granule_order, purkinje_order)
