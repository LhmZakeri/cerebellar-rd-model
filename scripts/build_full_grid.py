"""Full-scale grid geometry build -- issue #7a acceptance criterion.

Builds the real 2mm x 20mm granular-layer grid at 10um resolution
(~400k nodes), constructs a GridNodeBatch at that size, and reports build
time and memory footprint. Kept out of the default pytest path since it
exercises Numba JIT compilation and ~400k-node array allocation --
tests/test_geometry.py and tests/test_grid_node_batch.py cover correctness
on small grids instead.

Usage:
    python scripts/build_full_grid.py
"""
from __future__ import annotations

import argparse
import time

from src.simulation.geometry import FlatGrid, build_grid_neighbours
from src.simulation.grid_node_batch import GridNodeBatch

WIDTH_UM = 2_000.0     # 2 mm
HEIGHT_UM = 20_000.0   # 20 mm
RESOLUTION_UM = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-cells", type=int, default=None,
        help="Granule/Purkinje/stellate count, independent of grid size "
             "(DESIGN.md). Default: match positions.n_nodes (today's behavior).",
    )
    parser.add_argument(
        "--connectivity-seed", type=int, default=0,
        help="Seed for Golgi<->granule connectivity sampling (DESIGN.md), "
             "independent of --golgi-seed's placement randomness.",
    )
    args = parser.parse_args()

    geometry = FlatGrid(width_um=WIDTH_UM, height_um=HEIGHT_UM, resolution_um=RESOLUTION_UM)

    t0 = time.perf_counter()
    positions = geometry.build()
    t_positions = time.perf_counter() - t0

    t0 = time.perf_counter()
    neighbours = build_grid_neighbours(positions)
    t_neighbours = time.perf_counter() - t0

    n_cells = args.n_cells if args.n_cells is not None else positions.n_nodes

    t0 = time.perf_counter()
    node = GridNodeBatch(
        geometry, golgi_seed=0, n_cells=n_cells, connectivity_seed=args.connectivity_seed
    )
    t_node_batch = time.perf_counter() - t0

    neighbours_mb = neighbours.nbytes / 1e6

    print(f"Grid: {positions.n_cols} cols x {positions.n_rows} rows = {positions.n_nodes:,} nodes")
    print(f"  positions.build():        {t_positions:.3f} s")
    print(f"  build_grid_neighbours():  {t_neighbours:.3f} s  ({neighbours_mb:.1f} MB)")
    print(f"  GridNodeBatch() [incl. JIT]: {t_node_batch:.3f} s")
    print(f"  n_cells requested: {n_cells:,} (positions.n_nodes: {positions.n_nodes:,})")
    print(f"  node.cells.n_nodes == n_cells: {node.cells.n_nodes == n_cells}")
    print(f"  node.golgi.n_golgi (~1:430 of positions.n_nodes): {node.golgi.n_golgi:,}")


if __name__ == "__main__":
    main()
