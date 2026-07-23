"""plot_cell_types: 2D Golgi scatter + granule/Purkinje/stellate count
annotation (DESIGN.md). Uses the Agg backend so plt.show() doesn't open
a real window under pytest."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.simulation.cell_type_viewer import plot_cell_types
from src.simulation.geometry import FlatGrid
from src.simulation.grid_node_batch import GridNodeBatch

_N_COLS = 30
_N_ROWS = 20
_RESOLUTION_UM = 10.0


def _geometry() -> FlatGrid:
    return FlatGrid(
        width_um=_N_COLS * _RESOLUTION_UM,
        height_um=_N_ROWS * _RESOLUTION_UM,
        resolution_um=_RESOLUTION_UM,
    )


def test_plot_cell_types_renders_without_error():
    node = GridNodeBatch(
        _geometry(), golgi_seed=1, n_cells=100, connectivity_seed=0, golgi_ratio=1.0 / 20.0
    )
    plot_cell_types(node)
    plt.close("all")
