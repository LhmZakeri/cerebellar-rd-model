"""GridNodeBatch: construction at grid positions + no-behavior-change parity
against a bare NodeBatch, the sparse, nullable Golgi slot (see
DESIGN.md for why `golgi` lives
here and not on bare NodeBatch), and the remaining Node connectivity --
Golgi<->granule synapses + Golgi<->Golgi diffusion (DESIGN.md)."""
import numpy as np

from src.simulation.coupling_params import GridCouplingParams
from src.simulation.geometry import FlatGrid, build_golgi_granule_neighbours
from src.simulation.grid_node_batch import GridNodeBatch
from src.simulation.node_batch import NodeBatch

_DT = 0.01
_N_COLS = 3
_N_ROWS = 2
_RESOLUTION_UM = 10.0

# Disables both directions of the Golgi<->granule synapse so tests can
# isolate other pathways (diffusion, or bare vertical-synapse parity) --
# diffusion itself never touches granule/purkinje/stellate, so it needs no
# separate zeroing to preserve bare-NodeBatch parity.
_NO_GOLGI_GRANULE_SYNAPSES = GridCouplingParams(
    gmax_golgi_to_granule=0.0, gmax_granule_to_golgi=0.0
)

# Disables the four convergent granule/Purkinje/stellate pathways
# (DESIGN.md) that GridNodeBatch builds by default in place of
# NodeBatch's own 1:1 vertical synapses -- for tests that want to isolate
# some other pathway from these, or restore bare-NodeBatch parity.
_NO_NEW_VERTICAL_SYNAPSES = GridCouplingParams(
    gmax_granule_to_purkinje=0.0, gmax_stellate_to_purkinje=0.0,
    gmax_purkinje_to_stellate=0.0, gmax_granule_to_stellate=0.0,
)

_NO_GOLGI_GRANULE_AND_NO_NEW_VERTICAL_SYNAPSES = GridCouplingParams(
    gmax_golgi_to_granule=0.0, gmax_granule_to_golgi=0.0,
    gmax_granule_to_purkinje=0.0, gmax_stellate_to_purkinje=0.0,
    gmax_purkinje_to_stellate=0.0, gmax_granule_to_stellate=0.0,
)


def _small_geometry() -> FlatGrid:
    return FlatGrid(
        width_um=_N_COLS * _RESOLUTION_UM,
        height_um=_N_ROWS * _RESOLUTION_UM,
        resolution_um=_RESOLUTION_UM,
    )


class TestGridNodeBatchConstruction:

    def test_n_cells_decoupled_from_grid_size(self):
        """Granule/Purkinje/stellate count is an explicit n_cells, independent
        of the position grid's point count -- decouple GridNodeBatch
        cell count from the position grid (see DESIGN.md)."""
        n_cells = 10_000
        node = GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=n_cells, connectivity_seed=0)
        assert node.positions.n_nodes == _N_ROWS * _N_COLS
        assert node.cells.n_nodes == n_cells

    def test_neighbours_shape_matches_grid_size_not_n_cells(self):
        node = GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=10_000, connectivity_seed=0)
        assert node.neighbours.shape == (_N_ROWS * _N_COLS, 4)

    def test_node_batch_kwargs_pass_through(self):
        node = GridNodeBatch(
            _small_geometry(),
            golgi_seed=0,
            n_cells=_N_ROWS * _N_COLS,
            connectivity_seed=0,
            gmax_exc=0.0,
            gmax_inh=0.0,
            gmax_ps=0.0,
        )
        assert node.cells.exc_to_purkinje._p.gmax == 0.0

    def test_node_batch_vertical_synapses_default_to_zero(self):
        """GridNodeBatch's convergent granule/Purkinje/stellate pathways
        (DESIGN.md) replace NodeBatch's own 1:1 vertical synapses by
        default -- gmax_exc/gmax_inh/gmax_ps default to 0.0 here, unlike
        bare NodeBatch's own default of 1.0."""
        node = GridNodeBatch(
            _small_geometry(), golgi_seed=0, n_cells=_N_ROWS * _N_COLS, connectivity_seed=0,
        )
        assert node.cells.exc_to_purkinje._p.gmax == 0.0
        assert node.cells.inh_to_purkinje._p.gmax == 0.0
        assert node.cells.inh_purkinje_to_stellate._p.gmax == 0.0

    def test_connectivity_seed_is_required(self):
        try:
            GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=_N_ROWS * _N_COLS)
            assert False, "expected TypeError: connectivity_seed has no default"
        except TypeError:
            pass

    def test_exposes_new_connectivity_attributes(self):
        node = GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=_N_ROWS * _N_COLS, connectivity_seed=0)
        assert hasattr(node, "diffusion_neighbours")
        assert hasattr(node, "golgi_to_granule_neighbours")
        assert hasattr(node, "granule_to_golgi_neighbours")


class TestNoBehaviorChangeFromBareNodeBatch:
    """With every GridNodeBatch-added synaptic pathway's gain zeroed
    (isolating them) and NodeBatch's own 1:1 vertical synapses explicitly
    restored to their bare-NodeBatch defaults (GridNodeBatch zeroes them by
    default under DESIGN.md, since its own convergent pathways replace
    them), a GridNodeBatch's granule/Purkinje/stellate cells must still
    evolve bit-for-bit identically to a bare NodeBatch of the same size --
    proving no leakage from the new connectivity machinery when its gain is
    zero, not that no such machinery exists (DESIGN.md added real
    Golgi<->granule synapses + Golgi<->Golgi diffusion; DESIGN.md added
    the four convergent granule/Purkinje/stellate pathways; diffusion never
    touches granule/purkinje/stellate at all, so it needs no separate
    zeroing here)."""

    def test_step_trace_matches_bare_node_batch_exactly(self):
        n_nodes = _N_ROWS * _N_COLS
        grid_node = GridNodeBatch(
            _small_geometry(), golgi_seed=0, n_cells=n_nodes, connectivity_seed=0,
            coupling=_NO_GOLGI_GRANULE_AND_NO_NEW_VERTICAL_SYNAPSES,
            gmax_exc=1.0, gmax_inh=1.0, gmax_ps=1.0,
        )
        reference = NodeBatch(n_nodes)

        # .cells.inject_mossy_fiber_input(), not the GridNodeBatch-level
        # wrapper: mossy-fiber convergence (DESIGN.md) is itself one of
        # GridNodeBatch's added pathways now (like Golgi's own synapses
        # below), so isolating it here means driving the underlying NodeBatch
        # directly with the identical raw current, the same way
        # TestGranuleGolgiExcitation does to isolate Golgi's direct drive.
        grid_node.cells.inject_mossy_fiber_input(0.05)
        grid_node.inject_climbing_fiber_input(0.02)
        reference.inject_mossy_fiber_input(0.05)
        reference.inject_climbing_fiber_input(0.02)

        for _ in range(200):
            grid_node.step(_DT)
            reference.step(_DT)

        assert np.array_equal(grid_node.cells.granule.get_voltage(), reference.granule.get_voltage())
        assert np.array_equal(grid_node.cells.purkinje.get_voltage(), reference.purkinje.get_voltage())
        assert np.array_equal(grid_node.cells.stellate.get_voltage(), reference.stellate.get_voltage())

    def test_reset_restores_initial_conditions(self):
        grid_node = GridNodeBatch(
            _small_geometry(), golgi_seed=0, n_cells=_N_ROWS * _N_COLS, connectivity_seed=0,
        )
        V_g0 = grid_node.cells.granule.get_voltage().copy()

        grid_node.inject_mossy_fiber_input(0.05)
        for _ in range(50):
            grid_node.step(_DT)
        grid_node.reset()

        assert np.array_equal(grid_node.cells.granule.get_voltage(), V_g0)


class TestGolgiGranuleSynapses:
    """Golgi<->granule chemical synapses (DESIGN.md) have
    real dynamical effect during step() -- not just an index mapping."""

    _N_COLS = 30
    _N_ROWS = 20

    def _geometry(self) -> FlatGrid:
        return FlatGrid(
            width_um=self._N_COLS * _RESOLUTION_UM,
            height_um=self._N_ROWS * _RESOLUTION_UM,
            resolution_um=_RESOLUTION_UM,
        )

    def test_golgi_granule_synapses_have_real_effect(self):
        """Granule voltage traces must differ between a real (nonzero-gain)
        Golgi<->granule coupling and the same construction with both
        synaptic gains zeroed -- proving the pathway does something.

        Golgi cells are pacemakers with no external drive here -- their
        chemical synapses onto/from granule only carry current once a Golgi
        cell actually fires (TwoStateDestexhe's [T]-gated release), so the
        run needs to be long enough for at least one spike. Confirmed
        empirically: with this seed/grid, the first Golgi spike lands
        ~8ms in (diffusion-driven asymmetry breaks the identical-initial-
        condition symmetry across the 30 Golgi cells); 3000 steps (30ms)
        gives a comfortable margin.
        """
        n_cells = 200

        def _build(coupling):
            node = GridNodeBatch(
                self._geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=1,
                golgi_ratio=1.0 / 20.0, coupling=coupling,
            )
            node.inject_mossy_fiber_input(0.05)
            return node

        coupled = _build(GridCouplingParams())
        uncoupled = _build(_NO_GOLGI_GRANULE_SYNAPSES)

        for _ in range(3000):
            coupled.step(_DT)
            uncoupled.step(_DT)

        assert not np.array_equal(
            coupled.cells.granule.get_voltage(), uncoupled.cells.granule.get_voltage()
        )


class TestGranuleGolgiExcitation:
    """Granule->Golgi excitatory synapse (DESIGN.md):
    granule cells provide excitatory feedback onto the Golgi cells that
    inhibit them, sharing the same edge list as golgi_to_granule_neighbours
    -- on top of, not instead of, Golgi cells' own direct mossy-fiber
    excitation (TestGolgiMossyFiberExcitation above)."""

    _N_COLS = 30
    _N_ROWS = 20

    def _geometry(self) -> FlatGrid:
        return FlatGrid(
            width_um=self._N_COLS * _RESOLUTION_UM,
            height_um=self._N_ROWS * _RESOLUTION_UM,
            resolution_um=_RESOLUTION_UM,
        )

    def test_granule_to_golgi_synapse_has_real_effect(self):
        """Golgi voltage traces must differ between a real (nonzero-gain)
        granule->Golgi synapse and the same construction with it zeroed --
        proving the pathway does something on its own, isolated from
        diffusion and from Golgi's direct mossy-fiber drive (driven via
        node.cells.inject_mossy_fiber_input() directly, which reaches only
        the granule population, not GridNodeBatch's wrapper that also drives
        golgi)."""
        n_cells = 200

        def _build(coupling):
            node = GridNodeBatch(
                self._geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=1,
                golgi_ratio=1.0 / 20.0, coupling=coupling,
            )
            node.cells.inject_mossy_fiber_input(0.05)
            return node

        coupled = _build(GridCouplingParams(g_gap_nS=0.0))
        uncoupled = _build(GridCouplingParams(g_gap_nS=0.0, gmax_granule_to_golgi=0.0))

        for _ in range(3000):
            coupled.step(_DT)
            uncoupled.step(_DT)

        assert not np.array_equal(coupled.golgi.get_voltage(), uncoupled.golgi.get_voltage())


class TestGolgiSlot:
    """The nullable golgi slot: sparse-sized SolinasBatch + node-ID mapping,
    populated via Poisson-disk placement -- not one-per-node like the other
    three cell types (DESIGN.md)."""

    # A larger grid than the other tests' 3x2: at 1:430 density a 3x2 grid
    # would need golgi_ratio > 1 to place even a single Golgi cell.
    _N_COLS = 30
    _N_ROWS = 20

    def _geometry(self) -> FlatGrid:
        return FlatGrid(
            width_um=self._N_COLS * _RESOLUTION_UM,
            height_um=self._N_ROWS * _RESOLUTION_UM,
            resolution_um=_RESOLUTION_UM,
        )

    def test_golgi_seed_is_required(self):
        try:
            GridNodeBatch(self._geometry(), n_cells=1, connectivity_seed=0)
            assert False, "expected TypeError: golgi_seed has no default"
        except TypeError:
            pass

    def test_golgi_is_sparse_not_one_per_node(self):
        n_nodes = self._N_ROWS * self._N_COLS
        node = GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_nodes, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        assert node.golgi.n_golgi < n_nodes
        assert node.golgi.n_golgi == len(node.golgi_node_ids)

    def test_golgi_node_ids_are_valid_grid_nodes(self):
        n_nodes = self._N_ROWS * self._N_COLS
        node = GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_nodes, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        assert node.golgi_node_ids.min() >= 0
        assert node.golgi_node_ids.max() < n_nodes
        assert len(set(node.golgi_node_ids.tolist())) == len(node.golgi_node_ids)

    def test_same_seed_gives_same_placement(self):
        n_nodes = self._N_ROWS * self._N_COLS
        a = GridNodeBatch(
            self._geometry(), golgi_seed=5, n_cells=n_nodes, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        b = GridNodeBatch(
            self._geometry(), golgi_seed=5, n_cells=n_nodes, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        assert np.array_equal(a.golgi_node_ids, b.golgi_node_ids)

    def test_golgi_placement_is_unaffected_by_n_cells(self):
        """golgi_ratio is applied against positions.n_nodes (the grid), never
        n_cells -- decoupling granule/Purkinje/stellate count from the grid
        must not change Golgi placement at all."""
        n_nodes = self._N_ROWS * self._N_COLS
        a = GridNodeBatch(
            self._geometry(), golgi_seed=5, n_cells=n_nodes, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        b = GridNodeBatch(
            self._geometry(), golgi_seed=5, n_cells=10_000, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        assert np.array_equal(a.golgi_node_ids, b.golgi_node_ids)

    def test_golgi_cells_evolve_with_own_dynamics(self):
        """Each Golgi cell must integrate its own pacemaker dynamics when
        step() is called, matching the granule/Purkinje/stellate
        "independent columns" pattern NodeBatch already follows -- true
        whether or not Golgi<->Golgi diffusion (DESIGN.md) or
        Golgi<->granule synapses are active."""
        n_nodes = self._N_ROWS * self._N_COLS
        node = GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_nodes, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        V0 = node.golgi.get_voltage().copy()
        for _ in range(200):
            node.step(_DT)
        assert not np.array_equal(node.golgi.get_voltage(), V0)

    def test_reset_restores_golgi_initial_conditions(self):
        n_nodes = self._N_ROWS * self._N_COLS
        node = GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_nodes, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        V0 = node.golgi.get_voltage().copy()
        for _ in range(200):
            node.step(_DT)
        node.reset()
        assert np.array_equal(node.golgi.get_voltage(), V0)


class TestSpatialPositions:
    """GridNodeBatch retains, rather than discards, the positions used to
    build real connectivity (DESIGN.md) -- .node_x/.node_y (shared by
    granule/Purkinje/stellate at each index) and .golgi_x/.golgi_y (real
    grid positions), for the spatial activity view."""

    _N_COLS = 30
    _N_ROWS = 20

    def _geometry(self) -> FlatGrid:
        return FlatGrid(
            width_um=self._N_COLS * _RESOLUTION_UM,
            height_um=self._N_ROWS * _RESOLUTION_UM,
            resolution_um=_RESOLUTION_UM,
        )

    def test_node_position_shapes_match_n_cells(self):
        n_cells = 50
        node = GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        assert node.node_x.shape == (n_cells,)
        assert node.node_y.shape == (n_cells,)
        assert node.golgi_x.shape == (node.golgi.n_golgi,)
        assert node.golgi_y.shape == (node.golgi.n_golgi,)

    def test_golgi_position_matches_real_grid_position(self):
        n_nodes = self._N_ROWS * self._N_COLS
        node = GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_nodes, connectivity_seed=0,
            golgi_ratio=1.0 / 20.0,
        )
        np.testing.assert_array_equal(node.golgi_x, node.positions.x[node.golgi_node_ids])
        np.testing.assert_array_equal(node.golgi_y, node.positions.y[node.golgi_node_ids])

    def test_retained_node_position_is_the_one_that_built_connectivity(self):
        """The whole point of retaining rather than regenerating (DESIGN.md): re-deriving golgiToGranuleNeighbors from the retained
        .node_x/.node_y/.golgi_x/.golgi_y with the same divergence/seed must
        reproduce the exact edges GridNodeBatch actually wired -- proving
        the retained array is the one connectivity was built from, not a
        different, independently-drawn one."""
        n_cells = 50
        node = GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=3,
            golgi_ratio=1.0 / 20.0,
        )
        divergence = min(node.coupling.golgi_granule_divergence, n_cells)
        golgi_edge_idx, granule_edge_idx = build_golgi_granule_neighbours(
            node.golgi_x, node.golgi_y, node.node_x, node.node_y,
            divergence=divergence, seed=3,
            locality_sigma_um=node.coupling.golgi_granule_locality_sigma_um,
        )
        np.testing.assert_array_equal(golgi_edge_idx, node.golgi_edge_idx)
        np.testing.assert_array_equal(granule_edge_idx, node.granule_edge_idx)


class TestGolgiMossyFiberExcitation:
    """Golgi cells receive direct mossy-fiber excitation via
    inject_mossy_fiber_input() (DESIGN.md) -- on top of, not instead of,
    the excitatory feedback they get from granule cells (TestGranuleGolgiExcitation
    below)."""

    _N_COLS = 30
    _N_ROWS = 20

    def _geometry(self) -> FlatGrid:
        return FlatGrid(
            width_um=self._N_COLS * _RESOLUTION_UM,
            height_um=self._N_ROWS * _RESOLUTION_UM,
            resolution_um=_RESOLUTION_UM,
        )

    def _build(self, n_cells: int) -> GridNodeBatch:
        # Diffusion and both Golgi<->granule synapses zeroed so the
        # comparison below isolates the mossy-fiber pathway itself.
        coupling = GridCouplingParams(
            g_gap_nS=0.0, gmax_golgi_to_granule=0.0, gmax_granule_to_golgi=0.0
        )
        return GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=1,
            golgi_ratio=1.0 / 20.0, coupling=coupling,
        )

    def test_mossy_fiber_input_drives_golgi_voltage(self):
        n_cells = 200
        driven = self._build(n_cells)
        undriven = self._build(n_cells)
        driven.inject_mossy_fiber_input(0.05)
        undriven.inject_mossy_fiber_input(0.0)

        for _ in range(200):
            driven.step(_DT)
            undriven.step(_DT)

        assert not np.array_equal(driven.golgi.get_voltage(), undriven.golgi.get_voltage())

    def test_per_fiber_array_input_does_not_crash_golgi(self):
        """A per-mossy-fiber-for-granule array (DESIGN.md:
        n_mossy_fibers_granule long) with no I_golgi_nA given must still
        drive Golgi without shape-mismatching -- the granule-pool array gets
        mean-collapsed into a uniform Golgi-pool drive, and the
        n_mossy_fibers_granule/n_mossy_fibers_golgi pools are different
        sizes."""
        n_cells = 200
        node = self._build(n_cells)
        assert node.n_mossy_fibers_granule != node.n_mossy_fibers_golgi
        pattern = np.zeros(node.n_mossy_fibers_granule, dtype=np.float64)
        pattern[:30] = 0.05
        node.inject_mossy_fiber_input(pattern)
        node.step(_DT)  # must not raise

    def test_mismatched_array_shape_raises(self):
        """A stale per-granule-cell-sized array (the pre-DESIGN.md
        contract) must raise a clear error, not silently misapply -- the
        granule-facing pool is now sized n_mossy_fibers_granule, not
        n_cells."""
        n_cells = 200
        node = self._build(n_cells)
        pattern = np.zeros(n_cells, dtype=np.float64)
        try:
            node.inject_mossy_fiber_input(pattern)
            assert False, "expected ValueError on a wrongly-shaped mossy-fiber array"
        except ValueError:
            pass


class TestGranulePurkinjeStellateConvergence:
    """The four convergent granule/Purkinje/stellate pathways (DESIGN.md)
    that replace NodeBatch's own 1:1 vertical synapses under GridNodeBatch --
    each isolated from Golgi (diffusion + both Golgi<->granule directions
    zeroed throughout) and from the other three new pathways, proving each
    one has real, independent dynamical effect."""

    _N_COLS = 30
    _N_ROWS = 20

    def _geometry(self) -> FlatGrid:
        return FlatGrid(
            width_um=self._N_COLS * _RESOLUTION_UM,
            height_um=self._N_ROWS * _RESOLUTION_UM,
            resolution_um=_RESOLUTION_UM,
        )

    def _build(self, n_cells: int, coupling: GridCouplingParams) -> GridNodeBatch:
        return GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=1,
            golgi_ratio=1.0 / 20.0, coupling=coupling,
        )

    def test_granule_to_purkinje_synapse_has_real_effect(self):
        """Isolated from stellate->Purkinje (zeroed) so the comparison
        attributes any difference to granule->Purkinje alone."""
        n_cells = 200

        def _build(gmax_granule_to_purkinje):
            node = self._build(n_cells, GridCouplingParams(
                g_gap_nS=0.0, gmax_golgi_to_granule=0.0, gmax_granule_to_golgi=0.0,
                gmax_granule_to_purkinje=gmax_granule_to_purkinje,
                gmax_stellate_to_purkinje=0.0, gmax_purkinje_to_stellate=0.0,
                gmax_granule_to_stellate=0.0,
            ))
            node.inject_mossy_fiber_input(0.05)
            return node

        coupled = _build(2.8)
        uncoupled = _build(0.0)

        for _ in range(20_000):  # 200 ms -- granule needs ~100+ms to fire repeatedly
            coupled.step(_DT)
            uncoupled.step(_DT)

        assert not np.array_equal(
            coupled.cells.purkinje.get_voltage(), uncoupled.cells.purkinje.get_voltage()
        )

    def test_stellate_to_purkinje_synapse_has_real_effect(self):
        """Stellate has no direct external input in this model -- it's only
        reachable via granule->stellate (kept on here) or Purkinje->stellate
        (zeroed, to keep this a one-way isolation of stellate->Purkinje
        rather than a closed loop). granule->Purkinje is also zeroed so
        Purkinje's only remaining current source is stellate->Purkinje."""
        n_cells = 200

        def _build(gmax_stellate_to_purkinje):
            node = self._build(n_cells, GridCouplingParams(
                g_gap_nS=0.0, gmax_golgi_to_granule=0.0, gmax_granule_to_golgi=0.0,
                gmax_granule_to_purkinje=0.0,
                gmax_stellate_to_purkinje=gmax_stellate_to_purkinje,
                gmax_purkinje_to_stellate=0.0,
                gmax_granule_to_stellate=2.3,
            ))
            node.inject_mossy_fiber_input(0.05)
            return node

        coupled = _build(1.5)
        uncoupled = _build(0.0)

        for _ in range(20_000):  # 200 ms
            coupled.step(_DT)
            uncoupled.step(_DT)

        assert not np.array_equal(
            coupled.cells.purkinje.get_voltage(), uncoupled.cells.purkinje.get_voltage()
        )

    def test_purkinje_to_stellate_synapse_is_hyperpolarizing_not_depolarizing(self):
        """Regression test for the Purkinje->stellate sign fix (DESIGN.md,
        node_batch.py) at the GridNodeBatch convergent-wiring level, mirroring
        test_node_batch.py's NodeBatch-level version: driven stellate voltage
        must end up BELOW an undriven baseline, not merely different from it.
        granule->Purkinje/stellate->Purkinje/granule->stellate all zeroed so
        Purkinje->stellate is the only live pathway reaching stellate."""
        n_cells = 200

        def _build():
            return self._build(n_cells, GridCouplingParams(
                g_gap_nS=0.0, gmax_golgi_to_granule=0.0, gmax_granule_to_golgi=0.0,
                gmax_granule_to_purkinje=0.0, gmax_stellate_to_purkinje=0.0,
                gmax_purkinje_to_stellate=4.0, gmax_granule_to_stellate=0.0,
            ))

        driven = _build()
        undriven = _build()
        driven.inject_climbing_fiber_input(4200.0)
        undriven.inject_climbing_fiber_input(0.0)

        for _ in range(2_000):  # 20 ms
            driven.step(_DT)
            undriven.step(_DT)

        assert np.all(driven.cells.stellate.get_voltage() < undriven.cells.stellate.get_voltage())

    def test_granule_to_stellate_synapse_has_real_effect(self):
        """The new parallel-fiber->stellate pathway (DESIGN.md) -- didn't
        exist before this change. Purkinje->stellate zeroed to isolate this
        pathway from any Purkinje-mediated route to stellate."""
        n_cells = 200

        def _build(gmax_granule_to_stellate):
            node = self._build(n_cells, GridCouplingParams(
                g_gap_nS=0.0, gmax_golgi_to_granule=0.0, gmax_granule_to_golgi=0.0,
                gmax_granule_to_purkinje=0.0, gmax_stellate_to_purkinje=0.0,
                gmax_purkinje_to_stellate=0.0,
                gmax_granule_to_stellate=gmax_granule_to_stellate,
            ))
            node.inject_mossy_fiber_input(0.05)
            return node

        coupled = _build(2.3)
        uncoupled = _build(0.0)

        for _ in range(2_000):  # 20 ms -- granule depolarizes quickly under 0.05nA
            coupled.step(_DT)
            uncoupled.step(_DT)

        assert not np.array_equal(
            coupled.cells.stellate.get_voltage(), uncoupled.cells.stellate.get_voltage()
        )


class TestGolgiDiffusionCoupling:
    """Golgi<->Golgi gap-junction diffusion (DESIGN.md): a
    real D*Laplacian(V)-equivalent term exchanging current every step, not
    just a static graph -- the required sanity test that two diffusion-
    neighbouring Golgi cells' voltages visibly equilibrate under coupling."""

    _N_COLS = 30
    _N_ROWS = 20

    def _geometry(self) -> FlatGrid:
        return FlatGrid(
            width_um=self._N_COLS * _RESOLUTION_UM,
            height_um=self._N_ROWS * _RESOLUTION_UM,
            resolution_um=_RESOLUTION_UM,
        )

    def _build(self, g_gap_nS: float) -> GridNodeBatch:
        coupling = GridCouplingParams(g_gap_nS=g_gap_nS, gmax_golgi_to_granule=0.0)
        return GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=100, connectivity_seed=1,
            golgi_ratio=1.0 / 20.0, coupling=coupling,
        )

    def test_two_diffusion_coupled_golgi_cells_equilibrate(self):
        # Modest, near-rest perturbations (not the extreme swings that
        # trigger an action potential) -- Golgi cells are spiking pacemakers,
        # so a single voltage-gap snapshot taken after either cell has fired
        # reflects arbitrary spike phase, not coupling strength. Staying
        # subthreshold keeps the comparison a clean, deterministic ohmic
        # equilibration check.
        node = self._build(g_gap_nS=5.0)
        assert len(node.diffusion_neighbours) > 0, "expected at least one diffusion edge at this scale"
        i, j = node.diffusion_neighbours[0]
        node.golgi.V[i] = -55.0
        node.golgi.V[j] = -65.0
        gap0 = abs(node.golgi.V[i] - node.golgi.V[j])

        for _ in range(100):
            node.step(_DT)
        gap_coupled = abs(node.golgi.V[i] - node.golgi.V[j])
        assert gap_coupled < gap0, "diffusion-coupled gap did not shrink"

        control = self._build(g_gap_nS=0.0)
        control.golgi.V[i] = -55.0
        control.golgi.V[j] = -65.0
        for _ in range(100):
            control.step(_DT)
        gap_control = abs(control.golgi.V[i] - control.golgi.V[j])

        assert gap_coupled < gap_control, (
            "coupled gap not smaller than a no-diffusion control -- isolates "
            "diffusion's effect from Golgi's own intrinsic pacemaker dynamics"
        )


class TestHeterogeneity:
    """Sou11-style per-cell heterogeneity (DESIGN.md): granule/Golgi/
    Purkinje intrinsic-parameter heterogeneity (routed through NodeBatch for
    granule/Purkinje, direct for Golgi) plus Golgi position jitter off the
    exact grid-resolution lattice. heterogeneity_seed=None (default) must
    match today's exact homogeneous behavior."""

    _N_COLS = 30
    _N_ROWS = 20

    def _geometry(self) -> FlatGrid:
        return FlatGrid(
            width_um=self._N_COLS * _RESOLUTION_UM,
            height_um=self._N_ROWS * _RESOLUTION_UM,
            resolution_um=_RESOLUTION_UM,
        )

    def _build(self, n_cells: int, heterogeneity_seed) -> GridNodeBatch:
        return GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=1,
            golgi_ratio=1.0 / 20.0, heterogeneity_seed=heterogeneity_seed,
        )

    def test_no_seed_matches_prior_homogeneous_behavior(self):
        node = self._build(n_cells=50, heterogeneity_seed=None)
        assert len(set(node.cells.granule.V)) == 1
        assert len(set(node.cells.purkinje.V_s)) == 1
        assert len(set(node.golgi.V)) == 1
        assert np.allclose(node.golgi_x % _RESOLUTION_UM, 0)
        assert np.allclose(node.golgi_y % _RESOLUTION_UM, 0)

    def test_seed_gives_all_three_cell_types_real_spread(self):
        node = self._build(n_cells=200, heterogeneity_seed=7)
        assert node.cells.granule.V.std() > 0
        assert node.cells.purkinje.V_s.std() > 0
        assert node.golgi.V.std() > 0
        assert len(set(node.cells.stellate.V)) == 1  # out of scope, untouched

    def test_golgi_position_jittered_off_lattice_within_20_percent_of_resolution(self):
        node = self._build(n_cells=200, heterogeneity_seed=7)
        assert not np.allclose(node.golgi_x % _RESOLUTION_UM, 0)
        # jitter must stay within +/-20% of resolution_um of the original
        # lattice-snapped position (nearest multiple of resolution_um).
        nearest_lattice_x = np.round(node.golgi_x / _RESOLUTION_UM) * _RESOLUTION_UM
        max_jitter = 0.2 * _RESOLUTION_UM
        assert np.all(np.abs(node.golgi_x - nearest_lattice_x) <= max_jitter + 1e-9)

    def test_same_seed_reproducible(self):
        n1 = self._build(n_cells=50, heterogeneity_seed=42)
        n2 = self._build(n_cells=50, heterogeneity_seed=42)
        assert np.array_equal(n1.golgi_x, n2.golgi_x)
        assert np.array_equal(n1.golgi.V, n2.golgi.V)
        assert np.array_equal(n1.cells.granule.V, n2.cells.granule.V)

    def test_heterogeneous_grid_node_batch_steps_without_crashing(self):
        node = self._build(n_cells=50, heterogeneity_seed=3)
        node.inject_mossy_fiber_input(0.05)
        for _ in range(100):
            node.step(_DT)
        assert np.all(np.isfinite(node.cells.granule.get_voltage()))
        assert np.all(np.isfinite(node.golgi.get_voltage()))


class TestDistanceScaledConductance:
    """Sou11-style exponential distance-scaled conductance (DESIGN.md):
    g(d) = g_base * exp(-decay_per_um * d), applied to mossy->granule,
    mossy->Golgi, and inhibitory Golgi->granule -- NOT the exempted
    excitatory granule->Golgi (PF->GoC-equivalent) direction.
    distance_decay_per_um=None (default) must match today's exact prior
    behavior."""

    _N_COLS = 30
    _N_ROWS = 20

    def _geometry(self) -> FlatGrid:
        return FlatGrid(
            width_um=self._N_COLS * _RESOLUTION_UM,
            height_um=self._N_ROWS * _RESOLUTION_UM,
            resolution_um=_RESOLUTION_UM,
        )

    def _build(self, n_cells: int, decay_per_um) -> GridNodeBatch:
        return GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=1,
            golgi_ratio=1.0 / 20.0,
            coupling=GridCouplingParams(distance_decay_per_um=decay_per_um),
        )

    def test_no_decay_weight_sum_equals_contacts_count(self):
        node = self._build(n_cells=200, decay_per_um=None)
        assert np.all(node._golgi_to_granule_distance_weight == 1.0)
        assert np.allclose(node._mtg_weight_sum, node._mtg_contacts)
        assert np.allclose(node._mtgo_weight_sum, node._mtgo_contacts)

    def test_no_decay_matches_prior_exact_behavior(self):
        """The full step trace with distance_decay_per_um=None must be
        bit-identical to a plain default-coupling GridNodeBatch -- proving
        the weighted-mean refactor is a true no-op at the default."""
        n_cells = 100
        node_default = GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=1,
            golgi_ratio=1.0 / 20.0,
        )
        node_explicit_none = self._build(n_cells=n_cells, decay_per_um=None)
        node_default.inject_mossy_fiber_input(0.05)
        node_explicit_none.inject_mossy_fiber_input(0.05)
        for _ in range(200):
            node_default.step(_DT)
            node_explicit_none.step(_DT)
        assert np.array_equal(
            node_default.cells.granule.get_voltage(), node_explicit_none.cells.granule.get_voltage()
        )
        assert np.array_equal(node_default.golgi.get_voltage(), node_explicit_none.golgi.get_voltage())

    def test_decay_weights_match_exponential_formula(self):
        decay = 0.01
        node = self._build(n_cells=200, decay_per_um=decay)
        d = np.hypot(
            node.golgi_x[node.golgi_edge_idx] - node.node_x[node.granule_edge_idx],
            node.golgi_y[node.golgi_edge_idx] - node.node_y[node.granule_edge_idx],
        )
        expected = np.exp(-decay * d)
        assert np.allclose(node._golgi_to_granule_distance_weight, expected)
        assert np.all(node._golgi_to_granule_distance_weight <= 1.0)
        assert np.all(node._golgi_to_granule_distance_weight > 0.0)

    def test_granule_to_golgi_direction_has_no_weight_array_at_all(self):
        """The exemption is structural, not just numerically a no-op --
        no distance-weight array is ever computed for this direction."""
        node = self._build(n_cells=200, decay_per_um=0.01)
        assert not hasattr(node, "_granule_to_golgi_distance_weight")
        assert hasattr(node, "_golgi_to_granule_distance_weight")

    def test_decay_changes_granule_inhibition_relative_to_no_decay(self):
        """With real, nonzero inter-cell distances, applying decay must
        change the inhibitory Golgi->granule current relative to no decay
        -- proving the weight is actually consumed in step(), not just
        computed and ignored."""
        n_cells = 200

        def _drive(node):
            node.inject_mossy_fiber_input(0.05)
            for _ in range(3000):
                node.step(_DT)
            return node

        no_decay = _drive(self._build(n_cells, decay_per_um=None))
        with_decay = _drive(self._build(n_cells, decay_per_um=0.01))
        assert not np.array_equal(
            no_decay.cells.granule.get_voltage(), with_decay.cells.granule.get_voltage()
        )

    def test_heterogeneity_and_distance_decay_combine_without_crashing(self):
        """Parts A and B are independent features -- confirm they compose
        cleanly (both active at once)."""
        node = GridNodeBatch(
            self._geometry(), golgi_seed=1, n_cells=100, connectivity_seed=1,
            golgi_ratio=1.0 / 20.0, heterogeneity_seed=5,
            coupling=GridCouplingParams(distance_decay_per_um=0.01),
        )
        node.inject_mossy_fiber_input(0.05)
        for _ in range(100):
            node.step(_DT)
        assert np.all(np.isfinite(node.cells.granule.get_voltage()))
        assert np.all(np.isfinite(node.golgi.get_voltage()))
