"""Spatial layout for the granular layer: node positions, grid adjacency,
sparse Golgi-cell placement, and the remaining Node connectivity (issue:
Golgi<->granule synapses + Golgi<->Golgi diffusion, DESIGN.md).

Geometry only produces positions -- it does not know about neighbours, cell
types, or connectivity. build_grid_neighbours(), place_golgi_cells(),
sample_uniform_positions(), build_golgi_diffusion_neighbours(), and
build_golgi_granule_neighbours() are separate, position-only functions so
that connectivity builders can consume the same GridPositions (or plain
position arrays) without going through FlatGrid.

build_grid_neighbours()'s dense 4-connectivity list is a distinct concept
from DESIGN.md's Golgi-only diffusion_neighbours -- see
DESIGN.md. place_golgi_cells()
decides *which* nodes host a Golgi cell; build_golgi_diffusion_
neighbours() builds the Golgi-Golgi diffusion neighbour graph
directly from the placed Golgi (x, y) positions (a radius search over just
the Golgi-hosting points), not from filtering this module's dense
grid-neighbour list, which is far too fine-grained (10 um steps) to ever
connect two Golgi-hosting nodes at ~1:430 density -- per
DESIGN.md SS4.

Granule/Purkinje/stellate cells carry no position of their own
(DESIGN.md) -- sample_uniform_positions() draws a transient, continuous
position per granule cell (not snapped to this module's discrete grid, to
avoid quantization collisions once granule count decouples from grid node
count) purely so build_golgi_granule_neighbours() can bias its selection
toward nearby cells; these positions are discarded after the neighbour list
is built and never become granule state (DESIGN.md).

Units: x, y are in micrometers (um), per DESIGN.md's units contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# Neighbour array column order.
_UP, _DOWN, _LEFT, _RIGHT = range(4)
NO_NEIGHBOUR = -1

# build_golgi_granule_neighbours: gather at least divergence * this many
# local candidates before weighted-sampling from them, so the Gaussian
# locality weighting has a real pool to bias over rather than being forced
# to take almost every candidate found.
_CANDIDATE_OVERSAMPLE = 5


@dataclass(frozen=True)
class GridPositions:
    """Flat, row-major node positions on a 2D grid.

    node_id = row * n_cols + col. x runs along columns (the grid's width
    axis), y runs along rows.
    """

    x: np.ndarray  # [um], shape (n_rows * n_cols,)
    y: np.ndarray  # [um], shape (n_rows * n_cols,)
    n_rows: int
    n_cols: int
    resolution_um: float

    @property
    def n_nodes(self) -> int:
        return self.n_rows * self.n_cols


class Geometry(ABC):
    """Produces node positions. Swappable (e.g. FlatGrid -> a curved-folium
    implementation in a later phase) without changing anything downstream
    that only consumes GridPositions."""

    @abstractmethod
    def build(self) -> GridPositions:
        """Return this geometry's node positions."""


class FlatGrid(Geometry):
    """A flat 2D grid at fixed resolution.

    width_um is the short/x axis (columns), height_um is the long/y axis
    (rows) -- matching a folium's long, narrow ridge shape, which this flat
    grid is a stepping stone toward.
    """

    def __init__(self, width_um: float, height_um: float, resolution_um: float) -> None:
        if width_um <= 0 or height_um <= 0 or resolution_um <= 0:
            raise ValueError("width_um, height_um, and resolution_um must all be positive")
        self.width_um = width_um
        self.height_um = height_um
        self.resolution_um = resolution_um

    def build(self) -> GridPositions:
        n_cols = round(self.width_um / self.resolution_um)
        n_rows = round(self.height_um / self.resolution_um)
        col_idx, row_idx = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        x = (col_idx.ravel() * self.resolution_um).astype(np.float64)
        y = (row_idx.ravel() * self.resolution_um).astype(np.float64)
        return GridPositions(
            x=x, y=y, n_rows=n_rows, n_cols=n_cols, resolution_um=self.resolution_um
        )


def build_grid_neighbours(positions: GridPositions) -> np.ndarray:
    """4-connectivity neighbour indices for a row-major GridPositions grid.

    Returns an (n_nodes, 4) int32 array, columns [up, down, left, right],
    with NO_NEIGHBOUR (-1) where a node has no neighbour on that side (grid
    edges/corners).
    """
    n_rows, n_cols = positions.n_rows, positions.n_cols
    node_id = np.arange(positions.n_nodes, dtype=np.int32).reshape(n_rows, n_cols)

    neighbours = np.full((positions.n_nodes, 4), NO_NEIGHBOUR, dtype=np.int32)
    grid = neighbours.reshape(n_rows, n_cols, 4)

    grid[1:, :, _UP] = node_id[:-1, :]
    grid[:-1, :, _DOWN] = node_id[1:, :]
    grid[:, 1:, _LEFT] = node_id[:, :-1]
    grid[:, :-1, _RIGHT] = node_id[:, 1:]

    return neighbours


def place_golgi_cells(positions: GridPositions, ratio: float, seed: int) -> np.ndarray:
    """Poisson-disk (hard-core) placement of Golgi-hosting nodes.

    Uniform density with genuine per-cell randomness and no patchy clusters
    -- see DESIGN.md for why this
    was chosen over a thresholded Gaussian field (produces patches, not
    uniform spread) or a jittered lattice (residual periodic structure).

    ratio: target Golgi:node fraction (e.g. 1/430). The minimum inter-Golgi
    distance is derived from this density rather than a real anatomical
    spacing measurement, since this grid is already an idealized areal
    abstraction (DESIGN.md), not a calibrated 3D reconstruction.

    seed: required, explicit -- no default. Golgi placement feeds #7c's
    diffusion coupling and the Scientific Goal A/B/C sweeps, where whether a
    result changed because of the physics or because Golgi cells landed
    differently this run must stay an answerable, explicit question.

    Returns Golgi-hosting node IDs (int64, ascending).
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError("ratio must be in (0, 1)")

    n_nodes = positions.n_nodes
    target_n = max(1, round(ratio * n_nodes))
    width_um = positions.n_cols * positions.resolution_um
    height_um = positions.n_rows * positions.resolution_um
    area_um2 = width_um * height_um

    rng = np.random.default_rng(seed)

    # Sequential random hard-core (Matern-II-style) rejection sampling fills
    # to roughly 55-70% of the theoretical max packing density for a given
    # r_min, not 100% -- so start from a conservative r_min and shrink it a
    # few times if the first pass undershoots the target count. Trimming an
    # oversized accepted set down to target_n afterwards is always safe: a
    # subset of a valid hard-core set is still a valid hard-core set.
    r_min = 0.70 * np.sqrt(area_um2 / target_n)
    selected = np.empty(0, dtype=np.int64)
    for _ in range(6):
        selected = _greedy_hard_core_sample(positions, r_min, rng)
        if len(selected) >= target_n:
            return np.sort(selected[:target_n])
        r_min *= 0.85

    return np.sort(selected)


def _greedy_hard_core_sample(
    positions: GridPositions, r_min: float, rng: np.random.Generator
) -> np.ndarray:
    """Visit nodes in random order, greedily accepting any node at least
    r_min from every already-accepted node. Uses a spatial hash (cell size
    r_min) so acceptance checks stay local instead of O(n * accepted)."""
    x, y = positions.x, positions.y
    r_min_sq = r_min * r_min
    cell_of: dict[tuple[int, int], list[int]] = {}
    selected: list[int] = []

    for idx in rng.permutation(positions.n_nodes):
        px, py = x[idx], y[idx]
        cx, cy = int(px // r_min), int(py // r_min)

        too_close = False
        for dcx in (-1, 0, 1):
            for dcy in (-1, 0, 1):
                for other in cell_of.get((cx + dcx, cy + dcy), ()):
                    dx, dy = px - x[other], py - y[other]
                    if dx * dx + dy * dy < r_min_sq:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break

        if not too_close:
            selected.append(idx)
            cell_of.setdefault((cx, cy), []).append(idx)

    return np.array(selected, dtype=np.int64)


def sample_uniform_positions(
    width_um: float, height_um: float, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n continuous, uniform-random (x, y) positions within
    [0, width_um] x [0, height_um] -- NOT snapped to any discrete grid.

    Used to give granule cells a transient position for locality-biased
    Golgi<->granule connectivity (DESIGN.md), since granule cells no
    longer carry real position as simulation state (DESIGN.md). Snapping
    to the discrete 10 um grid instead would cause quantization collisions
    once granule count decouples from (and typically exceeds) grid node
    count -- continuous sampling avoids that entirely, regardless of the
    ratio between the two.

    seed: required, explicit -- no default, same reproducibility rationale
    as place_golgi_cells (whether a result changed due to physics or due to
    connectivity randomness must stay an answerable question).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, width_um, size=n)
    y = rng.uniform(0.0, height_um, size=n)
    return x, y


def build_golgi_diffusion_neighbours(
    golgi_x: np.ndarray, golgi_y: np.ndarray, radius_um: float | None = None
) -> np.ndarray:
    """Golgi<->Golgi gap-junction proximity graph, built directly
    from placed Golgi (x, y) positions -- per DESIGN.md SS4, never by
    filtering build_grid_neighbours()'s dense list (average Golgi-Golgi
    spacing at 1:430 density is ~207um = sqrt(430)*10um, far coarser than the
    10um grid, so filtering would silently produce an all-empty graph).

    Returns an (n_edges, 2) int64 array of (i, j) pairs with i < j, in
    Golgi-array-index space (i.e. indices into golgi_x/golgi_y, NOT grid-node
    IDs) -- a fixed-width column layout like build_grid_neighbours() doesn't
    fit here since degree varies per Golgi cell under a radius search.

    radius_um: None -> auto-calibrate to 1.5x the median nearest-neighbour
    distance among the placed Golgi cells, so it adapts to whatever
    golgi_ratio/extent was actually used rather than a fixed magic constant.

    Uses a brute-force vectorized pairwise distance, not the spatial-hash
    bucket trick _greedy_hard_core_sample uses -- at Phase 1 scale (~930
    Golgi cells) that's ~864k pairs, trivial, and far easier to hand-verify
    in tests. Revisit only if Golgi count reaches roughly 10k+ (out of scope
    here -- see DESIGN.md's parked real-folium-scale notes).
    """
    n = len(golgi_x)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)

    dx = golgi_x[:, None] - golgi_x[None, :]
    dy = golgi_y[:, None] - golgi_y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    if radius_um is None:
        nn = dist.copy()
        np.fill_diagonal(nn, np.inf)
        radius_um = 1.5 * np.median(nn.min(axis=1))

    i_idx, j_idx = np.triu_indices(n, k=1)
    within = dist[i_idx, j_idx] <= radius_um
    return np.stack([i_idx[within], j_idx[within]], axis=1).astype(np.int64)


def build_convergent_neighbours(
    target_x: np.ndarray,
    target_y: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    n_contacts: int,
    seed: int,
    locality_sigma_um: float = 250.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic locality-biased convergent connectivity: for each cell in the
    target population, locality-biased Gaussian selection of `n_contacts`
    cells from the source population (nearby cells preferred, not
    uniform-random across the whole population) -- since Scientific Goals
    A/B (tipping point / chaos characterization) are sensitive to local vs.
    global coupling structure.

    Returns (target_edge_idx, source_edge_idx), two parallel int64 arrays of
    length len(target_x) * n_contacts: edge k connects target cell
    target_edge_idx[k] to source cell source_edge_idx[k].

    Originally written as build_golgi_granule_neighbours (DESIGN.md) --
    extracted under this generic name (DESIGN.md) once granule/
    Purkinje/stellate convergent pathways needed the identical algorithm
    with different populations playing the target/source roles.
    build_golgi_granule_neighbours below is now a thin wrapper over this
    function; its docstring covers the Golgi<->granule-specific rationale
    (shared bidirectional edge list, ~2000/~3-4 divergence/convergence
    numbers) that doesn't generalize to every caller of this function.

    Locality bias is a Gaussian falloff, weight(d) = exp(-d^2 / (2 sigma^2))
    -- a clean single scale parameter, unlike inverse-square weighting which
    needs an arbitrary softening epsilon to avoid a singularity at d=0.
    locality_sigma_um defaults to 250.0, matching DESIGN.md's already-
    documented synapticNeighbors (stellate->Purkinje) reach of ~200-300um.

    Selection within one target cell's `n_contacts` never repeats a source
    index (rng.choice(..., replace=False)); source indices CAN repeat across
    different target cells' contact sets (real convergence/divergence).

    Efficiency: candidates are gathered via a spatial hash (bucket size
    locality_sigma_um, same pattern as _greedy_hard_core_sample), expanding
    the search ring outward until enough candidates are found -- never an
    O(n_target * n_source) brute-force search, since either population can be
    in the millions (DESIGN.md decoupled granule count from the grid).

    seed: required, explicit -- no default, same reproducibility rationale as
    place_golgi_cells.
    """
    n_source = len(source_x)
    if n_contacts > n_source:
        raise ValueError(
            f"n_contacts ({n_contacts}) cannot exceed the source population ({n_source})"
        )

    rng = np.random.default_rng(seed)
    cell_size = locality_sigma_um
    cell_of: dict[tuple[int, int], list[int]] = {}
    for idx in range(n_source):
        key = (int(source_x[idx] // cell_size), int(source_y[idx] // cell_size))
        cell_of.setdefault(key, []).append(idx)

    n_target = len(target_x)
    target_edges = np.empty(n_target * n_contacts, dtype=np.int64)
    source_edges = np.empty(n_target * n_contacts, dtype=np.int64)
    min_candidates = min(n_contacts * _CANDIDATE_OVERSAMPLE, n_source)

    for t in range(n_target):
        tx, ty = target_x[t], target_y[t]
        cx, cy = int(tx // cell_size), int(ty // cell_size)

        ring = 1
        candidates: list[int] = []
        while True:
            candidates = [
                i
                for dcx in range(-ring, ring + 1)
                for dcy in range(-ring, ring + 1)
                for i in cell_of.get((cx + dcx, cy + dcy), ())
            ]
            if len(candidates) >= min_candidates or len(candidates) >= n_source:
                break
            ring += 1

        cand = np.asarray(candidates, dtype=np.int64)
        d2 = (source_x[cand] - tx) ** 2 + (source_y[cand] - ty) ** 2
        weights = np.exp(-d2 / (2.0 * locality_sigma_um**2))
        wsum = weights.sum()
        weights = weights / wsum if wsum > 0.0 else np.full(len(cand), 1.0 / len(cand))

        chosen = rng.choice(cand, size=n_contacts, replace=False, p=weights)
        target_edges[t * n_contacts : (t + 1) * n_contacts] = t
        source_edges[t * n_contacts : (t + 1) * n_contacts] = chosen

    return target_edges, source_edges


def build_golgi_granule_neighbours(
    golgi_x: np.ndarray,
    golgi_y: np.ndarray,
    granule_x: np.ndarray,
    granule_y: np.ndarray,
    divergence: int,
    seed: int,
    locality_sigma_um: float = 250.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Golgi<->granule connectivity (DESIGN.md): for each Golgi cell,
    locality-biased selection of `divergence` granule targets. Thin wrapper
    over build_convergent_neighbours (DESIGN.md) -- Golgi plays the
    "target" role (the enumerated, divergence-per-cell side), granule plays
    the "source" role (the pool sampled from).

    Returns (golgi_edge_idx, granule_edge_idx), two parallel int64 arrays of
    length n_golgi * divergence: edge k connects Golgi cell golgi_edge_idx[k]
    to granule cell granule_edge_idx[k]. This single edge list serves both
    golgiToGranuleNeighbors (inhibitory, Golgi->granule) and
    granuleToGolgiNeighbors (excitatory, granule->Golgi) -- one bidirectional,
    one-to-many anatomical contact (~2000-cell divergence per Golgi cell, ~3-4
    Golgi cells converging per granule cell), not two independently-sampled
    graphs (DESIGN.md).
    """
    return build_convergent_neighbours(
        golgi_x, golgi_y, granule_x, granule_y, divergence, seed, locality_sigma_um
    )
