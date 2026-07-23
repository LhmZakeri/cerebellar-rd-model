"""
plot_cell_types -- a static, one-shot map of where Golgi cells landed
relative to the grid, plus the granule/Purkinje/stellate cell counts.

Golgi placement (Poisson-disk via place_golgi_cells) is a static
property of construction, not something that evolves over a run the way
voltage does -- so this is a standalone plot straight off a GridNodeBatch,
not another mode bolted onto ActivityViewer's time-stepping/recording
machinery (see activity_viewer.py's own docstring for why that module is
built around memmapped time series instead). Re-running with a different
golgi_seed on GridNodeBatch (see scripts/view_cell_types.py) is the whole
workflow for checking how placement changes.

Golgi is the only cell type with a real (x, y) position -- it's the one
that needs one, since Golgi<->Golgi gap-junction diffusion is
real future physics that depends on true position. Granule/Purkinje/
stellate cell count is decoupled from the grid (DESIGN.md) and has no
per-cell position to plot, so this draws a plain 2D scatter of Golgi's real
grid coordinates and reports the other three layers' counts as text,
instead of the old decorative 3D "one flat plane per layer" convention
(DESIGN.md).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.simulation.grid_node_batch import GridNodeBatch

_GOLGI_COLOR = "#DD3F3F"  # red


def plot_cell_types(node: GridNodeBatch, save_path: Path | None = None) -> None:
    """2D scatter of Golgi's real grid placement, with granule/Purkinje/
    stellate counts reported in the title (no per-cell position to plot).

    save_path: if given, saves the figure there instead of showing it
    interactively -- for batch/headless use (scripts/view_cell_types.py's
    positional output_dir)."""
    positions = node.positions
    n_rows, n_cols = positions.n_rows, positions.n_cols
    n_cells = node.cells.n_nodes  # uniform granule/Purkinje/stellate count

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_xlabel("col (x)")
    ax.set_ylabel("row (y)")
    ax.set_xlim(-1, n_cols)
    ax.set_ylim(-1, n_rows)
    ax.set_aspect("equal")

    golgi_col = node.golgi_node_ids % n_cols
    golgi_row = node.golgi_node_ids // n_cols
    ax.scatter(golgi_col, golgi_row, color=_GOLGI_COLOR, s=8, label="golgi")
    ax.legend(loc="upper left")

    n_golgi = len(node.golgi_node_ids)
    fig.suptitle(
        f"{n_golgi:,} golgi / {positions.n_nodes:,} grid nodes"
        f" (1:{positions.n_nodes / n_golgi:.0f})\n"
        f"{n_cells:,} granule / {n_cells:,} purkinje / {n_cells:,} stellate cells"
        f" (no fixed grid position -- DESIGN.md)"
    )
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
