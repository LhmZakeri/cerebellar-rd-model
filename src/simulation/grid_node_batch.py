"""GridNodeBatch: NodeBatch's cell columns placed at real grid positions,
plus the sparse, nullable Golgi slot and the Golgi<->granule / Golgi<->Golgi
connectivity NodeBatch itself doesn't model (DESIGN.md,
DESIGN.md).

Exposes `.golgi_x`/`.golgi_y` (real Golgi grid positions) and `.node_x`/
`.node_y` (one sampled position per Node index, shared by granule/Purkinje/
stellate) as retained attributes for the spatial activity view
(DESIGN.md) -- neither is read by any cell type's own dynamics or
step().

A thin wrapper: builds a Geometry, then an ordinary NodeBatch sized by
`n_cells`, independent of the grid's point count (DESIGN.md). Only
Golgi placement reads real (x, y) positions, so `golgi_ratio` applies
against `positions.n_nodes`, never `n_cells`.

`golgi` is a sparse SolinasBatch (n_golgi ~= n_nodes/430), not one-per-node
-- `golgi_node_ids` maps each Golgi array index to its host node ID
(DESIGN.md).

Seven coupled pathways, each with real dynamical effect in step():
- `golgi_to_granule_neighbours` / `granule_to_golgi_neighbours`: one shared,
  locality-biased edge list (`build_golgi_granule_neighbours`) carrying both
  the inhibitory Golgi->granule synapse and the excitatory granule->Golgi
  synapse -- one bidirectional anatomical contact, not two independently-
  sampled graphs. Granule cells carry no real position of their own, so
  locality uses a position sampled once at construction
  (`sample_uniform_positions`), retained as `.node_x`/`.node_y` for the
  spatial activity view (DESIGN.md) rather than discarded. Golgi cells
  also get direct mossy-fiber excitation (`inject_mossy_fiber_input`), on
  top of this granule-driven feedback.
- `diffusion_neighbours`: Golgi<->Golgi gap-junction diffusion, built
  directly from placed Golgi positions (`build_golgi_diffusion_neighbours`),
  never by filtering the dense grid list.
- Four convergent granule/Purkinje/stellate pathways (DESIGN.md):
  granule/PF->Purkinje (excitatory, ~100 contacts/Purkinje cell),
  stellate->Purkinje (inhibitory, ~50 contacts/Purkinje cell),
  Purkinje->stellate (inhibitory, ~2 contacts/stellate cell, Erev~-82mV --
  corrected from an original miscoded excitatory version), and
  parallel-fiber->stellate (excitatory, ~3 contacts/stellate cell, a pathway
  that didn't exist before DESIGN.md). All four use
  `build_convergent_neighbours` (the same locality-biased algorithm as
  Golgi<->granule, generalized) over `.node_x`/`.node_y` -- granule,
  Purkinje, and stellate all share this one position array, since they're
  vertically co-located per node. These four *replace* NodeBatch's own 1:1
  vertical synapses under GridNodeBatch: `gmax_exc`/`gmax_inh`/`gmax_ps` are
  defaulted to 0.0 in `node_batch_kwargs` (still overridable by an explicit
  caller) so the convergent versions are the real source of coupling.
  Literature-calibrated single-contact conductances (2.8/1.5/4.0/2.3 nS) are
  individually far too weak to move Purkinje/stellate voltage over a 1:1
  wire (confirmed empirically: ~0.00017 mV from one fully-spiking granule
  cell at the old gmax=1.0 nS placeholder) -- convergence, not gmax
  inflation, is what makes them effective.
- Two convergent mossy-fiber pathways (DESIGN.md), replacing the old
  "raw current straight onto every granule/Golgi cell" `inject_mossy_fiber_
  input()`: mossy fiber->granule (~4 fiber contacts/granule cell, ~39
  granule cells/fiber) and mossy fiber->Golgi (~100 fiber contacts/Golgi
  cell, ~25 Golgi cells/fiber). Two independent sampled mossy-fiber
  populations (`.mossy_granule_x`/`.mossy_granule_y`,
  `.mossy_golgi_x`/`.mossy_golgi_y`), not one shared pool -- the ~430:1
  granule:Golgi cell-count disparity makes a single pool unable to hit both
  target ratios at once (DESIGN.md). Not a chemical synapse (mossy
  fiber current is injected directly, no gating kinetics), so this doesn't
  reuse `TwoStateDestexheBatch` -- `inject_mossy_fiber_input()` gathers each
  target cell's converging fibers via `build_convergent_neighbours` and
  averages (not sums) them, so a given fiber strength stays the same
  physiological scale at the target regardless of that population's contact
  count.

All chemical synapses reuse `TwoStateDestexheBatch` unmodified, sized by
presynaptic cell count (n_golgi / n_cells) rather than edge count -- a
synapse's gating state depends only on its presynaptic cell's own voltage,
shared across all of that cell's outgoing edges, avoiding an
O(n_presynaptic * contacts) memory blowup.
"""
from __future__ import annotations

import numpy as np

from src.models.destexhe_numba import TwoStateDestexheBatch
from src.models.solinas_numba import SolinasBatch
from src.simulation.coupling_params import GridCouplingParams
from src.simulation.geometry import (
    Geometry,
    GridPositions,
    build_convergent_neighbours,
    build_golgi_diffusion_neighbours,
    build_golgi_granule_neighbours,
    build_grid_neighbours,
    place_golgi_cells,
    sample_uniform_positions,
)
from src.simulation.node_batch import NodeBatch

# Real Golgi:granule cell-count ratio in rat anatomy -- see
# DESIGN.md.
_DEFAULT_GOLGI_RATIO = 1.0 / 430.0

_PA_TO_NA = 1e-3  # matches node_batch.py's own constant -- destexhe_numba.py's
# get_conductance()/get_current() are pA, every *Batch.step()'s I_ext is nA.


def _distance_weights(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, decay_per_um: float | None
) -> np.ndarray:
    """Sou11-style exponential distance weight, exp(-decay_per_um * d), one
    weight per edge (x1[k], y1[k]) <-> (x2[k], y2[k]) -- DESIGN.md.
    decay_per_um=None -> all-ones (no scaling, exact prior behavior)."""
    if decay_per_um is None:
        return np.ones_like(x1)
    d = np.hypot(x1 - x2, y1 - y2)
    return np.exp(-decay_per_um * d)


class GridNodeBatch:
    """A NodeBatch whose columns sit at real (x, y) positions on a Geometry,
    with 4-connectivity grid neighbour lists precomputed for later coupling,
    a sparse Golgi cell population placed via Poisson-disk sampling, and the
    Golgi<->granule / Golgi<->Golgi connectivity wired into step()."""

    def __init__(
        self,
        geometry: Geometry,
        golgi_seed: int,
        n_cells: int,
        connectivity_seed: int,
        golgi_ratio: float = _DEFAULT_GOLGI_RATIO,
        coupling: GridCouplingParams | None = None,
        heterogeneity_seed: int | None = None,
        **node_batch_kwargs,
    ) -> None:
        """heterogeneity_seed: None (default) -> every cell identical, exact
        prior behavior. An int -> Sou11-style per-cell heterogeneity (DESIGN.md) for granule, Golgi, and Purkinje. Explicit named parameter
        here (not folded into **node_batch_kwargs) so it's visible in this
        constructor's own signature rather than implicit pass-through magic.
        Golgi draws from its own offset seed, distinct from granule/
        Purkinje's (routed to NodeBatch), so the three cell types'
        randomness doesn't correlate."""
        self.positions: GridPositions = geometry.build()
        self.neighbours = build_grid_neighbours(self.positions)
        # NodeBatch's own 1:1 vertical synapses are disabled by default --
        # GridNodeBatch's convergent granule/Purkinje/stellate pathways below
        # replace them (DESIGN.md). setdefault, not a hard override, so
        # an explicit caller can still opt back into the old 1:1 behavior.
        node_batch_kwargs.setdefault("gmax_exc", 0.0)
        node_batch_kwargs.setdefault("gmax_inh", 0.0)
        node_batch_kwargs.setdefault("gmax_ps", 0.0)
        self.cells = NodeBatch(n_cells, heterogeneity_seed=heterogeneity_seed, **node_batch_kwargs)
        self.coupling = coupling if coupling is not None else GridCouplingParams()
        golgi_heterogeneity_seed = (
            None if heterogeneity_seed is None else heterogeneity_seed + 1_000_000_007
        )

        self.golgi_node_ids = place_golgi_cells(
            self.positions, ratio=golgi_ratio, seed=golgi_seed
        )
        n_golgi = len(self.golgi_node_ids)
        # Real Golgi grid positions, retained (not just local) so the spatial
        # activity view can plot them (DESIGN.md).
        self.golgi_x = self.positions.x[self.golgi_node_ids]
        self.golgi_y = self.positions.y[self.golgi_node_ids]
        if golgi_heterogeneity_seed is not None:
            # Sou11-style position jitter (DESIGN.md, CONTEXT.md's
            # "Discovery-scale golgi_ratio" entry): de-regularizes Golgi
            # cells off the exact grid-resolution lattice they're placed on
            # by place_golgi_cells(). +/-20% of resolution_um (not +/-20% of
            # the position value itself, which has no meaningful "mean" to
            # jitter around at domain scale) -- x and y drawn independently.
            # Offset seed (+2) from golgi_heterogeneity_seed so this doesn't
            # reuse the identical draw sequence SolinasBatch's own
            # area/leak/V heterogeneity uses.
            pos_rng = np.random.default_rng(golgi_heterogeneity_seed + 2)
            jitter_max = 0.2 * self.positions.resolution_um
            self.golgi_x = self.golgi_x + pos_rng.uniform(-jitter_max, jitter_max, size=n_golgi)
            self.golgi_y = self.golgi_y + pos_rng.uniform(-jitter_max, jitter_max, size=n_golgi)
        golgi_x, golgi_y = self.golgi_x, self.golgi_y
        self.golgi = SolinasBatch(n_golgi, heterogeneity_seed=golgi_heterogeneity_seed)
        self._I_mossy_golgi = np.zeros(n_golgi, dtype=np.float64)

        # --- Golgi<->Golgi gap-junction diffusion ---------------
        self.diffusion_neighbours = build_golgi_diffusion_neighbours(
            golgi_x, golgi_y, radius_um=self.coupling.golgi_diffusion_radius_um
        )
        self._diff_i = self.diffusion_neighbours[:, 0]
        self._diff_j = self.diffusion_neighbours[:, 1]

        # --- Golgi<->granule connectivity (DESIGN.md) -------------------
        # Transient positions, sampled once to build a locality-biased
        # neighbour list. Unlike DESIGN.md's original "discarded after
        # this, never touches granule state or step()", these are now
        # retained as self.node_x/self.node_y (DESIGN.md): the spatial
        # activity view needs *some* position per Node index to plot
        # granule/Purkinje/stellate, and reusing this exact array -- rather
        # than sampling a fresh one -- guarantees the plotted positions
        # never drift from the real golgiToGranuleNeighbors/
        # granuleToGolgiNeighbors wiring built from it below. Still never
        # read by granule/Purkinje/stellate's own dynamics or step().
        width_um = self.positions.n_cols * self.positions.resolution_um
        height_um = self.positions.n_rows * self.positions.resolution_um
        self.node_x, self.node_y = sample_uniform_positions(
            width_um, height_um, n_cells, seed=connectivity_seed
        )
        granule_x, granule_y = self.node_x, self.node_y
        divergence = min(self.coupling.golgi_granule_divergence, n_cells)
        self.golgi_edge_idx, self.granule_edge_idx = build_golgi_granule_neighbours(
            golgi_x,
            golgi_y,
            granule_x,
            granule_y,
            divergence=divergence,
            seed=connectivity_seed,
            locality_sigma_um=self.coupling.golgi_granule_locality_sigma_um,
        )
        # One shared edge list, two directions (DESIGN.md):
        # inhibitory Golgi -> granule and excitatory granule -> Golgi both
        # ride the same anatomical contact, not two independently-sampled
        # graphs.
        self.golgi_to_granule_neighbours = (self.golgi_edge_idx, self.granule_edge_idx)
        self.granule_to_golgi_neighbours = (self.granule_edge_idx, self.golgi_edge_idx)

        # Sou11-style distance-scaled conductance (DESIGN.md): applies
        # ONLY to the inhibitory Golgi->granule direction. The excitatory
        # granule->Golgi direction (the PF->GoC-equivalent contact) is
        # exempted -- Sou11's paper keeps it constant along the fiber,
        # replicated here exactly rather than decaying it too.
        self._golgi_to_granule_distance_weight = _distance_weights(
            golgi_x[self.golgi_edge_idx], golgi_y[self.golgi_edge_idx],
            granule_x[self.granule_edge_idx], granule_y[self.granule_edge_idx],
            self.coupling.distance_decay_per_um,
        )

        self.golgi_to_granule_synapse = TwoStateDestexheBatch.inhibitory(
            n_golgi, gmax=self.coupling.gmax_golgi_to_granule
        )
        self.granule_to_golgi_synapse = TwoStateDestexheBatch.excitatory(
            n_cells, gmax=self.coupling.gmax_granule_to_golgi
        )

        # --- mossy fiber -> granule / Golgi, convergent (DESIGN.md) -----
        # Two independent sampled mossy-fiber populations, NOT one shared pool
        # (unlike Golgi<->granule's single shared edge list above, which is
        # one bidirectional anatomical contact) -- granule and Golgi are
        # different postsynaptic populations reached by very different
        # average contact/divergence counts (4/39 vs 100/25), and the ~430:1
        # granule:Golgi cell-count disparity means no single pool size can
        # hit both target ratios at once (DESIGN.md). Offset from
        # connectivity_seed (not reusing _sub_seeds below) so this doesn't
        # perturb the existing granule/Purkinje/stellate pathways'
        # seed-reproducibility.
        _mossy_seeds = np.random.default_rng(connectivity_seed + 1_000_003).integers(
            0, 2**31 - 1, size=4
        )

        n_mf_granule = max(1, round(
            n_cells * self.coupling.mossy_to_granule_contacts
            / self.coupling.mossy_to_granule_divergence
        ))
        self.n_mossy_fibers_granule = n_mf_granule
        self.mossy_granule_x, self.mossy_granule_y = sample_uniform_positions(
            width_um, height_um, n_mf_granule, seed=int(_mossy_seeds[0])
        )
        self._mtg_contacts = min(self.coupling.mossy_to_granule_contacts, n_mf_granule)
        self.mossy_to_granule_edge_idx, self._mtg_source_idx = build_convergent_neighbours(
            self.node_x, self.node_y, self.mossy_granule_x, self.mossy_granule_y,
            n_contacts=self._mtg_contacts, seed=int(_mossy_seeds[1]),
            locality_sigma_um=self.coupling.mossy_to_granule_locality_sigma_um,
        )
        # Sou11-style distance-scaled "conductance" (DESIGN.md): mossy
        # fiber current has no literal conductance parameter (DESIGN.md
        # -- raw injected current, no gating kinetics), so the closest
        # analogous translation is a distance-weighted MEAN instead of the
        # uniform mean DESIGN.md originally used -- nearer fibers
        # contribute more to a target cell's current. weight=1 everywhere
        # (decay_per_um=None) reduces exactly to the prior uniform mean.
        self._mtg_distance_weight = _distance_weights(
            self.node_x[self.mossy_to_granule_edge_idx], self.node_y[self.mossy_to_granule_edge_idx],
            self.mossy_granule_x[self._mtg_source_idx], self.mossy_granule_y[self._mtg_source_idx],
            self.coupling.distance_decay_per_um,
        )
        self._mtg_weight_sum = np.zeros(n_cells, dtype=np.float64)
        np.add.at(self._mtg_weight_sum, self.mossy_to_granule_edge_idx, self._mtg_distance_weight)

        n_mf_golgi = max(1, round(
            n_golgi * self.coupling.mossy_to_golgi_contacts
            / self.coupling.mossy_to_golgi_divergence
        ))
        self.n_mossy_fibers_golgi = n_mf_golgi
        self.mossy_golgi_x, self.mossy_golgi_y = sample_uniform_positions(
            width_um, height_um, n_mf_golgi, seed=int(_mossy_seeds[2])
        )
        self._mtgo_contacts = min(self.coupling.mossy_to_golgi_contacts, n_mf_golgi)
        self.mossy_to_golgi_edge_idx, self._mtgo_source_idx = build_convergent_neighbours(
            golgi_x, golgi_y, self.mossy_golgi_x, self.mossy_golgi_y,
            n_contacts=self._mtgo_contacts, seed=int(_mossy_seeds[3]),
            locality_sigma_um=self.coupling.mossy_to_golgi_locality_sigma_um,
        )
        # Same distance-weighted-mean translation as mossy->granule above.
        self._mtgo_distance_weight = _distance_weights(
            golgi_x[self.mossy_to_golgi_edge_idx], golgi_y[self.mossy_to_golgi_edge_idx],
            self.mossy_golgi_x[self._mtgo_source_idx], self.mossy_golgi_y[self._mtgo_source_idx],
            self.coupling.distance_decay_per_um,
        )
        self._mtgo_weight_sum = np.zeros(n_golgi, dtype=np.float64)
        np.add.at(self._mtgo_weight_sum, self.mossy_to_golgi_edge_idx, self._mtgo_distance_weight)

        self._I_mossy_fiber_granule = np.zeros(n_mf_granule, dtype=np.float64)
        self._I_mossy_fiber_golgi = np.zeros(n_mf_golgi, dtype=np.float64)

        # --- granule/Purkinje/stellate convergent connectivity (DESIGN.md) --
        # Four independent edge lists (no shared-edge-list reuse like Golgi<->
        # granule -- these pathways have no anatomical 1:1 correspondence to
        # each other, e.g. stellate->Purkinje's ~50 contacts vs. Purkinje-
        # >stellate's ~2). All four target/source positions are the same
        # node_x/node_y array (granule/Purkinje/stellate are co-located), so
        # identical seeds across pathways risk correlated edge structure --
        # derive 4 distinct sub-seeds from connectivity_seed instead of
        # reusing it 4 times.
        _sub_seeds = np.random.default_rng(connectivity_seed).integers(0, 2**31 - 1, size=4)

        gp_contacts = min(self.coupling.granule_to_purkinje_contacts, n_cells)
        self.granule_to_purkinje_edge_idx, self._gp_source_idx = build_convergent_neighbours(
            self.node_x, self.node_y, self.node_x, self.node_y,
            n_contacts=gp_contacts, seed=int(_sub_seeds[0]),
            locality_sigma_um=self.coupling.granule_to_purkinje_locality_sigma_um,
        )
        self.granule_to_purkinje_synapse = TwoStateDestexheBatch.excitatory(
            n_cells, gmax=self.coupling.gmax_granule_to_purkinje
        )

        sp_contacts = min(self.coupling.stellate_to_purkinje_contacts, n_cells)
        self.stellate_to_purkinje_edge_idx, self._sp_source_idx = build_convergent_neighbours(
            self.node_x, self.node_y, self.node_x, self.node_y,
            n_contacts=sp_contacts, seed=int(_sub_seeds[1]),
            locality_sigma_um=self.coupling.stellate_to_purkinje_locality_sigma_um,
        )
        self.stellate_to_purkinje_synapse = TwoStateDestexheBatch.inhibitory(
            n_cells, gmax=self.coupling.gmax_stellate_to_purkinje
        )

        ps_contacts = min(self.coupling.purkinje_to_stellate_contacts, n_cells)
        self.purkinje_to_stellate_edge_idx, self._ps_source_idx = build_convergent_neighbours(
            self.node_x, self.node_y, self.node_x, self.node_y,
            n_contacts=ps_contacts, seed=int(_sub_seeds[2]),
            locality_sigma_um=self.coupling.purkinje_to_stellate_locality_sigma_um,
        )
        self.purkinje_to_stellate_synapse = TwoStateDestexheBatch.inhibitory(
            n_cells, gmax=self.coupling.gmax_purkinje_to_stellate,
            Erev=self.coupling.purkinje_to_stellate_Erev_mV,
        )

        gs_contacts = min(self.coupling.granule_to_stellate_contacts, n_cells)
        self.granule_to_stellate_edge_idx, self._gs_source_idx = build_convergent_neighbours(
            self.node_x, self.node_y, self.node_x, self.node_y,
            n_contacts=gs_contacts, seed=int(_sub_seeds[3]),
            locality_sigma_um=self.coupling.granule_to_stellate_locality_sigma_um,
        )
        self.granule_to_stellate_synapse = TwoStateDestexheBatch.excitatory(
            n_cells, gmax=self.coupling.gmax_granule_to_stellate
        )

    def step(self, dt: float) -> None:
        V_golgi = self.golgi.get_voltage()
        V_granule = self.cells.granule.get_voltage()
        V_purkinje = self.cells.purkinje.get_voltage()
        V_stellate = self.cells.stellate.get_voltage()

        # --- Golgi<->Golgi gap-junction diffusion: a real D*Laplacian(V) ----
        # term, not a static graph. Outflow-sign convention throughout (see
        # DESIGN.md): edge (i, j)'s outflow from i is g_gap*(V_i - V_j),
        # and the negative from j -- summed per-cell, then subtracted below,
        # exactly like every chemical synapse's ohmic current already is.
        dV = V_golgi[self._diff_i] - V_golgi[self._diff_j]
        I_diff_out_pA = np.zeros(self.golgi.n_golgi)
        np.add.at(I_diff_out_pA, self._diff_i, self.coupling.g_gap_nS * dV)
        np.add.at(I_diff_out_pA, self._diff_j, -self.coupling.g_gap_nS * dV)

        # --- Golgi <-> granule (inhibitory one way, excitatory the other) ---
        # One synapse-state slot per PRESYNAPTIC cell (n_golgi / n_cells), not
        # per edge -- a synapse's gating state depends only on its
        # presynaptic cell's own voltage, shared across all of that cell's
        # outgoing edges (DESIGN.md). Conductance is gathered onto edges
        # and the ohmic current computed manually against the gathered
        # postsynaptic voltage, since get_current()'s shape assumption
        # (matching pre/post array sizes) doesn't fit edge-gathered data.
        # Golgi excitation also gets direct mossy-fiber drive (folded into
        # I_ext below) on top of this granule-driven feedback.
        self.golgi_to_granule_synapse.step(dt, V_golgi)
        self.granule_to_golgi_synapse.step(dt, V_granule)

        g_inh_edges = (
            self.golgi_to_granule_synapse.get_conductance()[self.golgi_edge_idx]
            * self._golgi_to_granule_distance_weight  # DESIGN.md
        )
        I_granule_inh_out_pA = np.zeros(self.cells.n_nodes)
        np.add.at(
            I_granule_inh_out_pA,
            self.granule_edge_idx,
            g_inh_edges
            * (V_granule[self.granule_edge_idx] - self.golgi_to_granule_synapse._p.Erev),
        )

        g_exc_edges = self.granule_to_golgi_synapse.get_conductance()[self.granule_edge_idx]
        I_golgi_exc_out_pA = np.zeros(self.golgi.n_golgi)
        np.add.at(
            I_golgi_exc_out_pA,
            self.golgi_edge_idx,
            g_exc_edges
            * (V_golgi[self.golgi_edge_idx] - self.granule_to_golgi_synapse._p.Erev),
        )

        # --- granule/Purkinje/stellate convergent synapses (DESIGN.md) --
        # Same gather-conductance-per-edge / np.add.at-scatter pattern as
        # Golgi<->granule above, x4. Each of the four synapses' gating state
        # is stepped once (n_cells-sized, one slot per presynaptic cell),
        # then gathered onto its own edge list and scattered onto its
        # target population.
        self.granule_to_purkinje_synapse.step(dt, V_granule)
        self.stellate_to_purkinje_synapse.step(dt, V_stellate)
        self.purkinje_to_stellate_synapse.step(dt, V_purkinje)
        self.granule_to_stellate_synapse.step(dt, V_granule)

        g_gp_edges = self.granule_to_purkinje_synapse.get_conductance()[self._gp_source_idx]
        I_purkinje_exc_out_pA = np.zeros(self.cells.n_nodes)
        np.add.at(
            I_purkinje_exc_out_pA,
            self.granule_to_purkinje_edge_idx,
            g_gp_edges
            * (V_purkinje[self.granule_to_purkinje_edge_idx] - self.granule_to_purkinje_synapse._p.Erev),
        )

        g_sp_edges = self.stellate_to_purkinje_synapse.get_conductance()[self._sp_source_idx]
        I_purkinje_inh_out_pA = np.zeros(self.cells.n_nodes)
        np.add.at(
            I_purkinje_inh_out_pA,
            self.stellate_to_purkinje_edge_idx,
            g_sp_edges
            * (V_purkinje[self.stellate_to_purkinje_edge_idx] - self.stellate_to_purkinje_synapse._p.Erev),
        )

        g_ps_edges = self.purkinje_to_stellate_synapse.get_conductance()[self._ps_source_idx]
        I_stellate_inh_out_pA = np.zeros(self.cells.n_nodes)
        np.add.at(
            I_stellate_inh_out_pA,
            self.purkinje_to_stellate_edge_idx,
            g_ps_edges
            * (V_stellate[self.purkinje_to_stellate_edge_idx] - self.purkinje_to_stellate_synapse._p.Erev),
        )

        g_gs_edges = self.granule_to_stellate_synapse.get_conductance()[self._gs_source_idx]
        I_stellate_exc_out_pA = np.zeros(self.cells.n_nodes)
        np.add.at(
            I_stellate_exc_out_pA,
            self.granule_to_stellate_edge_idx,
            g_gs_edges
            * (V_stellate[self.granule_to_stellate_edge_idx] - self.granule_to_stellate_synapse._p.Erev),
        )

        extra_purkinje_I_nA = -(I_purkinje_exc_out_pA + I_purkinje_inh_out_pA) * _PA_TO_NA
        extra_stellate_I_nA = -(I_stellate_inh_out_pA + I_stellate_exc_out_pA) * _PA_TO_NA

        self.cells.step(
            dt,
            extra_granule_I_nA=-I_granule_inh_out_pA * _PA_TO_NA,
            extra_purkinje_I_nA=extra_purkinje_I_nA,
            extra_stellate_I_nA=extra_stellate_I_nA,
        )
        self.golgi.step(
            dt, self._I_mossy_golgi - (I_diff_out_pA + I_golgi_exc_out_pA) * _PA_TO_NA
        )

    def reset(self) -> None:
        self.cells.reset()
        self.golgi.reset()
        self.golgi_to_granule_synapse.reset()
        self.granule_to_golgi_synapse.reset()
        self.granule_to_purkinje_synapse.reset()
        self.stellate_to_purkinje_synapse.reset()
        self.purkinje_to_stellate_synapse.reset()
        self.granule_to_stellate_synapse.reset()
        self._I_mossy_golgi = np.zeros(self.golgi.n_golgi, dtype=np.float64)
        self._I_mossy_fiber_granule = np.zeros(self.n_mossy_fibers_granule, dtype=np.float64)
        self._I_mossy_fiber_golgi = np.zeros(self.n_mossy_fibers_golgi, dtype=np.float64)

    def inject_mossy_fiber_input(self, I_nA, I_golgi_nA=None) -> None:
        """Set the tonic mossy-fiber drive, via two independent, real
        convergent mossy-fiber populations (DESIGN.md) -- NOT a 1:1 wire
        to either granule or Golgi: ~4 fiber contacts/granule cell (~39
        granule cells/fiber), ~100 fiber contacts/Golgi cell (~25 Golgi
        cells/fiber, on top of the excitatory feedback Golgi cells get from
        granule cells via granule_to_golgi_synapse -- DESIGN.md). Each
        target cell's drive is the MEAN of its converging fibers' current
        values (not the sum), so a given fiber strength stays the same
        physiological scale at the target regardless of how many contacts
        that population averages -- summing would make Golgi cells (100
        contacts) receive ~25x more current than granule cells (4 contacts)
        at the same fiber strength.

        I_nA: drive for the granule-facing mossy-fiber pool. A scalar
        broadcasts to every fiber in that pool (`n_mossy_fibers_granule`);
        an array must be shape (n_mossy_fibers_granule,) -- one value per
        fiber, NOT per granule cell (see `.mossy_granule_x`/
        `.mossy_granule_y` for that pool's positions, e.g. to build a
        spatially-localized pattern).

        I_golgi_nA: drive for the Golgi-facing mossy-fiber pool
        (`n_mossy_fibers_golgi`), scalar or array. None (default): if I_nA
        is scalar, reuse it for the Golgi pool too (so a uniform tonic-drive
        call stays a single argument, e.g. `inject_mossy_fiber_input(0.05)`);
        if I_nA is an array, collapse it to its mean as a uniform Golgi
        drive (matching the pre-DESIGN.md fallback behavior).
        """
        granule_fiber = self._as_mossy_array(I_nA, self.n_mossy_fibers_granule, "I_nA")
        if I_golgi_nA is not None:
            golgi_fiber = self._as_mossy_array(I_golgi_nA, self.n_mossy_fibers_golgi, "I_golgi_nA")
        elif np.isscalar(I_nA):
            golgi_fiber = self._as_mossy_array(I_nA, self.n_mossy_fibers_golgi, "I_golgi_nA")
        else:
            golgi_fiber = np.full(
                self.n_mossy_fibers_golgi, granule_fiber.mean(), dtype=np.float64
            )

        self._I_mossy_fiber_granule = granule_fiber
        self._I_mossy_fiber_golgi = golgi_fiber

        # Distance-weighted mean (DESIGN.md) -- weight=1 everywhere when
        # distance_decay_per_um=None, so this reduces exactly to the
        # original uniform mean (sum / constant contacts count) DESIGN.md used, verified by parity tests.
        I_granule_nA = np.zeros(self.cells.n_nodes, dtype=np.float64)
        np.add.at(
            I_granule_nA, self.mossy_to_granule_edge_idx,
            granule_fiber[self._mtg_source_idx] * self._mtg_distance_weight,
        )
        I_granule_nA /= self._mtg_weight_sum
        self.cells.inject_mossy_fiber_input(I_granule_nA)

        I_golgi_nA_arr = np.zeros(self.golgi.n_golgi, dtype=np.float64)
        np.add.at(
            I_golgi_nA_arr, self.mossy_to_golgi_edge_idx,
            golgi_fiber[self._mtgo_source_idx] * self._mtgo_distance_weight,
        )
        I_golgi_nA_arr /= self._mtgo_weight_sum
        self._I_mossy_golgi = I_golgi_nA_arr

    def _as_mossy_array(self, I_nA, n: int, arg_name: str) -> np.ndarray:
        if np.isscalar(I_nA):
            return np.full(n, I_nA, dtype=np.float64)
        I_nA = np.asarray(I_nA, dtype=np.float64)
        if I_nA.shape != (n,):
            raise ValueError(f"{arg_name} has shape {I_nA.shape}, expected ({n},)")
        return I_nA

    def inject_climbing_fiber_input(self, I_nA) -> None:
        self.cells.inject_climbing_fiber_input(I_nA)
