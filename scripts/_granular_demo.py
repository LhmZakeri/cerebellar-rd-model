"""Shared setup for the #7a.1 example scripts (record/view/visualize
2d_granular_activity.py) -- not a standalone script itself. Builds the
demo GridNodeBatch: #7a's flat grid (for Golgi placement) plus an
independent granule/Purkinje/stellate count (n_cells, DESIGN.md),
driven by an index-based mossy-fiber patch (-> granule) and a separate
climbing-fiber patch (-> Purkinje), so the recording has non-uniform drive
across cells instead of every one being an identical, identically-driven
column.

Granule/Purkinje/stellate carry no position of their own (see ADR 0005),
so the patch patterns below are defined over cell *index*, not grid (x,y)
-- unlike the grid, whose only remaining consumer is Golgi placement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.simulation.coupling_params import GridCouplingParams
from src.simulation.geometry import FlatGrid
from src.simulation.grid_node_batch import GridNodeBatch

# Pinned to the Phase 1 spec (~400k nodes/layer, 1.2M total,
# 10 um resolution). Do not scale WIDTH_UM/HEIGHT_UM up toward real folium
# size (e.g. 65mm x 110mm), and do not shrink RESOLUTION_UM below 10um --
# either one hits the memory-bandwidth wall documented in DESIGN.md's
# "Performance strategy" section (both crashed/were too slow when tried).
# The uncoupled model is already a throughput floor for the future
# diffusion-coupled one, so finer resolution is a Phase 2 question, once
# the per-node cost itself is addressed.
WIDTH_UM = 2_000.0  # 2 mm
HEIGHT_UM = 20_000.0  # 20 mm
RESOLUTION_UM = 10.0

DT_MS = 0.01  # ODE substep, matches DESIGN.md's "Operator splitting"
DURATION_MS = 200.0
RECORD_EVERY = 20  # -> record dt = 0.2 ms

MOSSY_INPUT_NA = 0.05  # matches run_dangelo2001_prototype.py's "High input"
CLIMBING_INPUT_NA = 3500.0  # Fernandez/Purkinje's current scale is ~1000x D'Angelo's;
# matches run_fernandez2007_prototype.py's tonic-firing pulse
PATCH_FRACTION = 0.15  # driven patch is this fraction of cells, from one end of the index range

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "granular_activity"

# ------------------------------------------------------------------------------


def _mossy_fiber_pattern(n_fibers: int) -> np.ndarray:
    """Localized mossy-fiber drive: the first PATCH_FRACTION of mossy fibers
    (by index) get tonic input, the rest get none -- an index-space analogue
    of the old spatial corner patch. Mossy fiber is its own population
    (DESIGN.md), separate from and much smaller than granule/Purkinje/
    stellate's n_cells -- this works for any n_fibers, so it's called once
    per fiber pool (granule-facing, Golgi-facing) at their own sizes, not
    n_cells."""
    n_patch = max(1, round(PATCH_FRACTION * n_fibers))
    pattern = np.zeros(n_fibers, dtype=np.float64)
    pattern[:n_patch] = MOSSY_INPUT_NA
    return pattern


# ------------------------------------------------------------------------------


def _climbing_fiber_pattern(n_cells: int) -> np.ndarray:
    """Localized climbing-fiber drive: the last PATCH_FRACTION of cells (by
    index) get tonic input -- the index-space "opposite corner" analogue, so
    Purkinje/stellate get real drive of their own rather than only weak
    pass-through from granule's vertical synapse."""
    n_patch = max(1, round(PATCH_FRACTION * n_cells))
    pattern = np.zeros(n_cells, dtype=np.float64)
    #pattern[-n_patch:] = CLIMBING_INPUT_NA
    pattern[:n_patch] = CLIMBING_INPUT_NA
    return pattern


# ------------------------------------------------------------------------------


def build_demo_node(
    width_um: float,
    height_um: float,
    resolution_um: float,
    n_cells: int | None = None,
    golgi_seed: int = 0,
    connectivity_seed: int = 0,
    golgi_ratio: float = 1.0 / 430.0,
    disable_golgi_coupling: bool = False,
) -> GridNodeBatch:
    """disable_golgi_coupling: zero out the Golgi->granule synapse and
    Golgi<->Golgi diffusion (DESIGN.md) -- a before/after toggle for
    comparing a recording with vs. without the new connectivity, since
    otherwise there's no way to see its effect except by inspecting code."""
    geometry = FlatGrid(
        width_um=width_um, height_um=height_um, resolution_um=resolution_um
    )
    if n_cells is None:
        # Preserve today's default demo: one column per grid node at Phase 1
        # scale. Pass n_cells to decouple explicitly (DESIGN.md).
        n_cells = geometry.build().n_nodes

    coupling = (
        GridCouplingParams(g_gap_nS=0.0, gmax_golgi_to_granule=0.0)
        if disable_golgi_coupling
        else None
    )
    node = GridNodeBatch(
        geometry,
        golgi_seed=golgi_seed,
        n_cells=n_cells,
        connectivity_seed=connectivity_seed,
        golgi_ratio=golgi_ratio,
        coupling=coupling,
    )

    node.inject_mossy_fiber_input(
        _mossy_fiber_pattern(node.n_mossy_fibers_granule),
        _mossy_fiber_pattern(node.n_mossy_fibers_golgi),
    )
    node.inject_climbing_fiber_input(_climbing_fiber_pattern(n_cells))
    return node
