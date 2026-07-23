"""ActivityViewer: per-layer voltage-vs-cell-index graphs, independent of
grid shape (DESIGN.md). Uses the Agg backend (matching
scripts/run_*_prototype.py's convention) so .show() doesn't open a real
window under pytest."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.simulation.activity_recording import record_grid_activity
from src.simulation.activity_viewer import _DISPLAY_MAX_CELLS, ActivityViewer
from src.simulation.geometry import FlatGrid
from src.simulation.grid_node_batch import GridNodeBatch

_N_COLS = 3
_N_ROWS = 2
_RESOLUTION_UM = 10.0
_N_CELLS = 20  # deliberately different from the 6-node grid


def _small_geometry() -> FlatGrid:
    return FlatGrid(
        width_um=_N_COLS * _RESOLUTION_UM,
        height_um=_N_ROWS * _RESOLUTION_UM,
        resolution_um=_RESOLUTION_UM,
    )


def _record(tmp_path, n_cells=_N_CELLS):
    node = GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=n_cells, connectivity_seed=0)
    node.inject_mossy_fiber_input(0.05)
    node.inject_climbing_fiber_input(0.02)
    record_grid_activity(node, duration_ms=1.0, dt_ms=0.01, record_every=10, output_dir=tmp_path)


def test_viewer_reads_n_cells_from_meta(tmp_path):
    _record(tmp_path)
    viewer = ActivityViewer(tmp_path)
    assert viewer.n_cells == _N_CELLS


def test_viewer_reads_golgi_layer(tmp_path):
    """Golgi's own voltage (DESIGN.md) is a different length (n_golgi,
    sparse) from the other three layers -- proving the coupling's effect on
    Golgi itself is actually visible, not just inferred from granule."""
    _record(tmp_path)
    viewer = ActivityViewer(tmp_path)
    assert viewer.has_golgi
    assert viewer.n_golgi > 0
    values = viewer._frame_values("golgi", 0)
    assert len(values) == viewer.n_golgi  # small enough here to need no downsampling


def test_show_renders_without_error(tmp_path):
    _record(tmp_path)
    viewer = ActivityViewer(tmp_path)
    viewer.show()
    plt.close("all")


def test_frame_values_downsampled_for_large_n_cells(tmp_path):
    n_cells = _DISPLAY_MAX_CELLS * 3
    _record(tmp_path, n_cells=n_cells)
    viewer = ActivityViewer(tmp_path)
    values = viewer._frame_values("granule", 0)
    assert len(values) <= _DISPLAY_MAX_CELLS
    assert len(values) == len(np.arange(0, n_cells, viewer._stride))
