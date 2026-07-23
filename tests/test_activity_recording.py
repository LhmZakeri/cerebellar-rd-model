"""record_grid_activity: recorded shapes/meta must follow n_cells (DESIGN.md), not the position grid's node count -- see
DESIGN.md.
Also records Golgi's own voltage (DESIGN.md), a different length
(n_golgi, sparse) from the other three layers."""
import numpy as np

from src.simulation.activity_recording import (
    GOLGI_LAYER,
    LAYERS,
    META_FILE,
    POSITIONS_FILE,
    PROGRESS_FILE,
    VOLTAGE_FILE_TEMPLATE,
    read_progress,
    record_grid_activity,
)
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


def test_recording_shapes_follow_n_cells_not_grid_size(tmp_path):
    node = GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=_N_CELLS, connectivity_seed=0)
    node.inject_mossy_fiber_input(0.05)
    node.inject_climbing_fiber_input(0.02)

    record_grid_activity(node, duration_ms=1.0, dt_ms=0.01, record_every=10, output_dir=tmp_path)

    meta = np.load(tmp_path / META_FILE)
    assert int(meta["n_cells"]) == _N_CELLS
    n_golgi = int(meta["n_golgi"])
    assert n_golgi == node.golgi.n_golgi
    n_frames = int(meta["n_frames"])
    assert n_frames > 0

    for layer in LAYERS:
        arr = np.load(tmp_path / VOLTAGE_FILE_TEMPLATE.format(layer=layer), mmap_mode="r")
        assert arr.shape == (n_frames, _N_CELLS)

    golgi_arr = np.load(tmp_path / VOLTAGE_FILE_TEMPLATE.format(layer=GOLGI_LAYER), mmap_mode="r")
    assert golgi_arr.shape == (n_frames, n_golgi)

    progress = np.load(tmp_path / PROGRESS_FILE, mmap_mode="r")
    assert int(progress[0]) == n_frames


def test_positions_recorded_directly_not_regenerated(tmp_path):
    """DESIGN.md: positions.npz holds the exact arrays GridNodeBatch
    used to build the real Golgi<->granule wiring, not seeds to regenerate
    them from -- so the recorded arrays must equal node.node_x/node.golgi_x
    etc. bit-for-bit (up to the float32 downcast)."""
    node = GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=_N_CELLS, connectivity_seed=0)
    node.inject_mossy_fiber_input(0.05)

    record_grid_activity(node, duration_ms=1.0, dt_ms=0.01, record_every=10, output_dir=tmp_path)

    positions = np.load(tmp_path / POSITIONS_FILE)
    assert positions["node_x"].dtype == np.float32
    assert positions["node_x"].shape == (_N_CELLS,)
    assert positions["node_y"].shape == (_N_CELLS,)
    assert positions["golgi_x"].shape == (node.golgi.n_golgi,)
    assert positions["golgi_y"].shape == (node.golgi.n_golgi,)

    np.testing.assert_allclose(positions["node_x"], node.node_x, rtol=1e-6)
    np.testing.assert_allclose(positions["node_y"], node.node_y, rtol=1e-6)
    np.testing.assert_allclose(positions["golgi_x"], node.golgi_x, rtol=1e-6)
    np.testing.assert_allclose(positions["golgi_y"], node.golgi_y, rtol=1e-6)


def test_time_windowed_stimulation_switches_on_and_off(tmp_path):
    """mossy_pattern/mossy_start_ms/mossy_end_ms confine that pathway's drive
    to [start_ms, end_ms) instead of holding it on for the whole run -- spy
    on node.inject_mossy_fiber_input (still calling through to the real
    method) to check it fires exactly at the two transition steps, with the
    right values, rather than once up front."""
    node = GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=_N_CELLS, connectivity_seed=0)

    calls = []
    real_inject = node.inject_mossy_fiber_input

    def spy_inject(I_nA):
        calls.append(I_nA)
        real_inject(I_nA)

    node.inject_mossy_fiber_input = spy_inject

    record_grid_activity(
        node, duration_ms=0.05, dt_ms=0.01, record_every=1, output_dir=tmp_path,
        mossy_pattern=0.05, mossy_start_ms=0.02, mossy_end_ms=0.04,
    )

    assert calls == [0.05, 0.0]


def test_no_window_matches_prior_always_on_behavior(tmp_path):
    """Leaving mossy_pattern (and thus start/end_ms) unpassed must behave
    exactly as before this feature existed: whatever the caller injected
    ahead of time stays on for the whole run, with record_grid_activity
    never touching it."""
    node = GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=_N_CELLS, connectivity_seed=0)
    node.inject_mossy_fiber_input(0.05)

    calls = []
    real_inject = node.inject_mossy_fiber_input
    node.inject_mossy_fiber_input = lambda I_nA: (calls.append(I_nA), real_inject(I_nA))

    record_grid_activity(node, duration_ms=1.0, dt_ms=0.01, record_every=10, output_dir=tmp_path)

    assert calls == []


def test_read_progress_matches_finished_recording(tmp_path):
    node = GridNodeBatch(_small_geometry(), golgi_seed=0, n_cells=_N_CELLS, connectivity_seed=0)
    node.inject_mossy_fiber_input(0.05)

    record_grid_activity(node, duration_ms=1.0, dt_ms=0.01, record_every=10, output_dir=tmp_path)

    frames_done, n_frames = read_progress(tmp_path)
    assert frames_done == n_frames
    assert n_frames == int(np.load(tmp_path / META_FILE)["n_frames"])
