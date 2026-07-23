"""SpatialActivityViewer: per-cell scatter view colored by voltage
(DESIGN.md). Uses the Agg backend (matching test_activity_viewer.py's
convention) so .show() doesn't open a real window under pytest."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.simulation.activity_recording import record_grid_activity
from src.simulation.geometry import FlatGrid
from src.simulation.grid_node_batch import GridNodeBatch
from src.simulation.spatial_activity_viewer import _DISPLAY_MAX_CELLS, SpatialActivityViewer

_N_COLS = 30
_N_ROWS = 20
_RESOLUTION_UM = 10.0
_N_CELLS = 20  # deliberately different from the grid's node count


def _geometry() -> FlatGrid:
    return FlatGrid(
        width_um=_N_COLS * _RESOLUTION_UM,
        height_um=_N_ROWS * _RESOLUTION_UM,
        resolution_um=_RESOLUTION_UM,
    )


def _record(tmp_path, n_cells=_N_CELLS):
    node = GridNodeBatch(
        _geometry(), golgi_seed=1, n_cells=n_cells, connectivity_seed=0, golgi_ratio=1.0 / 20.0
    )
    node.inject_mossy_fiber_input(0.05)
    node.inject_climbing_fiber_input(0.02)
    record_grid_activity(node, duration_ms=1.0, dt_ms=0.01, record_every=10, output_dir=tmp_path)


def test_missing_positions_file_raises(tmp_path):
    """A recording made before DESIGN.md has no positions.npz -- must
    fail loudly, not silently show an empty/wrong plot."""
    node = GridNodeBatch(
        _geometry(), golgi_seed=1, n_cells=_N_CELLS, connectivity_seed=0, golgi_ratio=1.0 / 20.0
    )
    record_grid_activity(node, duration_ms=1.0, dt_ms=0.01, record_every=10, output_dir=tmp_path)
    (tmp_path / "positions.npz").unlink()

    with pytest.raises(FileNotFoundError):
        SpatialActivityViewer(tmp_path)


def test_invalid_wavefront_delta_threshold_rejected(tmp_path):
    _record(tmp_path)
    with pytest.raises(ValueError):
        SpatialActivityViewer(tmp_path, golgi_wavefront_delta_threshold_mv=-5.0)


def test_positions_loaded_and_shaped_correctly(tmp_path):
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path)
    assert viewer.node_x.shape == (_N_CELLS,)
    assert viewer.node_y.shape == (_N_CELLS,)
    assert viewer.golgi_x.shape == (viewer.n_golgi,)
    assert viewer.golgi_y.shape == (viewer.n_golgi,)


def test_wavefront_zero_at_frame_zero(tmp_path):
    """No prior frame to compare against at frame 0 -- the diffusion
    wavefront (CONTEXT.md) must be exactly zero there, not NaN or garbage."""
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path)
    wavefront = viewer._golgi_wavefront(0)
    assert np.all(wavefront == 0.0)


def test_wavefront_matches_manual_delta_v(tmp_path):
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path)
    frame = 3
    v_curr = viewer._layer_voltage("golgi", frame)
    v_prev = viewer._layer_voltage("golgi", frame - 1)
    expected = np.abs(v_curr - v_prev)
    np.testing.assert_allclose(viewer._golgi_wavefront(frame), expected)


def test_hull_none_at_frame_zero(tmp_path):
    """No prior frame to diff against at frame 0 -- there's nothing to
    trace a wavefront boundary through yet."""
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path)
    assert viewer._golgi_wavefront_hull(0) is None


def test_hull_is_closed_polygon_of_real_golgi_positions(tmp_path):
    """Every hull vertex must be an actual recorded Golgi position -- no
    interpolated/fabricated points between cells (DESIGN.md) -- and the
    polygon must close (first vertex == last). Threshold set low so real
    recorded data reliably crosses it."""
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path, golgi_wavefront_delta_threshold_mv=0.0)
    hull = None
    for frame in range(1, viewer.n_frames_total):
        hull = viewer._golgi_wavefront_hull(frame)
        if hull is not None:
            break
    if hull is None:
        pytest.skip("no frame in this recording had enough newly-recruited Golgi cells to form a hull")

    hull_x, hull_y = hull
    assert hull_x[0] == hull_x[-1] and hull_y[0] == hull_y[-1]  # closed loop
    for x, y in zip(hull_x, hull_y):
        matches = (viewer.golgi_x == x) & (viewer.golgi_y == y)
        assert matches.any(), "hull vertex is not a real recorded Golgi position"


def test_hull_none_with_too_few_active_cells(tmp_path, monkeypatch):
    """Fewer than 3 newly-recruited cells (ConvexHull's minimum for a 2D
    polygon) must degrade to None rather than raise."""
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path)
    v_prev = np.full(viewer.n_golgi, -70.0, dtype=np.float32)  # all quiescent
    v_curr = v_prev.copy()
    v_curr[0] += 50.0
    v_curr[1] += 50.0  # exactly 2 newly-recruited cells -- below ConvexHull's minimum of 3

    def fake_layer_voltage(layer, frame):
        return v_curr if frame == 5 else v_prev

    monkeypatch.setattr(viewer, "_layer_voltage", fake_layer_voltage)
    assert viewer._golgi_wavefront_hull(5) is None


def test_hull_excludes_already_active_cells_even_with_large_delta(tmp_path, monkeypatch):
    """The whole point of replacing pure |deltaV| ranking (Q's request): a
    cell already depolarized last frame doesn't count as 'newly recruited'
    into the wavefront, even if its own |deltaV| this step is large."""
    _record(tmp_path)
    viewer = SpatialActivityViewer(
        tmp_path, golgi_active_voltage_mv=-40.0, golgi_wavefront_delta_threshold_mv=10.0
    )
    v_prev = np.full(viewer.n_golgi, -70.0, dtype=np.float32)
    v_prev[:3] = -20.0  # already active/depolarized last frame
    v_curr = v_prev + 50.0  # every cell has an equally large delta this step

    def fake_layer_voltage(layer, frame):
        return v_curr if frame == 5 else v_prev

    monkeypatch.setattr(viewer, "_layer_voltage", fake_layer_voltage)
    hull = viewer._golgi_wavefront_hull(5)
    assert hull is not None
    hull_x, hull_y = hull
    for x, y in zip(viewer.golgi_x[:3], viewer.golgi_y[:3]):
        assert not ((hull_x == x) & (hull_y == y)).any(), (
            "an already-active cell was included in the wavefront boundary"
        )


def test_show_jumps_to_latest_ready_frame(tmp_path, monkeypatch):
    """show() must display the latest ready frame on open, not always
    restart at frame 0 -- matching ActivityViewer's own convention, and
    essential for watching a still-running recording (progress climbing in
    another process)."""
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path)
    assert viewer._frame == 0  # constructor default, before show() is ever called
    monkeypatch.setattr(plt, "show", lambda *a, **kw: None)
    viewer.show()
    assert viewer._frame == viewer.n_frames_total - 1
    plt.close("all")


def test_show_3d_renders_without_error(tmp_path):
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path, use_3d=True)
    viewer.show()
    plt.close("all")


def test_show_renders_without_error(tmp_path):
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path)
    viewer.show()
    plt.close("all")


def test_node_positions_downsampled_for_large_n_cells(tmp_path):
    n_cells = _DISPLAY_MAX_CELLS * 3
    _record(tmp_path, n_cells=n_cells)
    viewer = SpatialActivityViewer(tmp_path)
    assert len(viewer._node_idx) <= _DISPLAY_MAX_CELLS
    assert len(viewer._node_idx) == len(np.arange(0, n_cells, viewer._stride))


def test_golgi_never_downsampled(tmp_path):
    _record(tmp_path)
    viewer = SpatialActivityViewer(tmp_path)
    # Sparse enough (~1:20 here, ~1:430 at real scale) to never hit the cap.
    assert viewer.n_golgi < _DISPLAY_MAX_CELLS
    assert viewer._layer_voltage("golgi", 0).shape == (viewer.n_golgi,)
