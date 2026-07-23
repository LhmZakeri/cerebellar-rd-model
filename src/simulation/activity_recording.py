"""
Disk-backed recording of GridNodeBatch layer voltages.

The recorder and viewer (activity_viewer.py) communicate only through this
module's on-disk file contract (meta + progress marker + per-layer memmaps),
never in-process objects -- so a recording still in progress in one process
can be watched live from another (scripts/record_2d_granular_activity.py
recording, scripts/view_activity.py watching the same --output-dir).

File layout in output_dir:
  meta.npz               -- n_cells, n_golgi, dt/record_every, n_frames.
                             Granule/Purkinje/stellate carry no grid position
                             of their own (DESIGN.md), so nothing here
                             assumes or records grid shape.
  progress.npy            -- single int64 memmap: how many frames are valid
                             so far (an unwritten frame is zeros, not "no
                             data", so the viewer needs this to avoid
                             displaying frames not yet recorded)
  activity_<layer>.npy   -- one (n_frames, n_cells) float16 memmap per layer
                             in LAYERS, plus one (n_frames, n_golgi) memmap
                             for GOLGI_LAYER, since Golgi is sparse
                             (DESIGN.md), not one-per-cell
                             like the other three.
  positions.npz            -- visualization-only positions for the spatial
                             activity view (DESIGN.md): node_x/node_y
                             (float32, length n_cells -- shared by granule/
                             Purkinje/stellate at each index, since they're
                             vertically synapsed at the same Node) and
                             golgi_x/golgi_y (float32, length n_golgi, real
                             grid positions). Recorded directly from
                             `node.node_x`/`node.golgi_x` etc. rather than
                             regenerated from a seed, so the plotted
                             positions can never drift from the positions
                             that built the real Golgi<->granule wiring.

Voltage is stored as float16: plenty of precision for a visual heatmap, at a
fraction of float32's size (raw array size, not compute, is this project's
usual bottleneck -- see DESIGN.md's "Performance strategy"). Recording writes
straight into the pre-allocated memmaps rather than an in-memory array, so
RAM use stays O(one frame) regardless of grid size or duration.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from src.simulation.grid_node_batch import GridNodeBatch

# ------------------------------------------------------------------------------
LAYERS = ("granule", "purkinje", "stellate")
GOLGI_LAYER = "golgi"
V_MIN, V_MAX = -90.0, 60.0

META_FILE = "meta.npz"
PROGRESS_FILE = "progress.npy"
VOLTAGE_FILE_TEMPLATE = "activity_{layer}.npy"
POSITIONS_FILE = "positions.npz"


# ------------------------------------------------------------------------------
def record_grid_activity(
    node: GridNodeBatch,
    duration_ms: float,
    dt_ms: float,
    record_every: int,
    output_dir: Path,
    mossy_pattern=None,
    mossy_golgi_pattern=None,
    mossy_start_ms: float | None = None,
    mossy_end_ms: float | None = None,
    climbing_pattern=None,
    climbing_start_ms: float | None = None,
    climbing_end_ms: float | None = None,
) -> None:
    """Step `node` for duration_ms, recording every layer's voltage (granule/
    Purkinje/stellate, plus Golgi -- see GOLGI_LAYER) every record_every
    steps to output_dir. Writes meta.npz and progress.npy before the step
    loop starts, so a viewer pointed at output_dir can already set up its
    plot and see "0/n_frames ready" immediately -- it does not have to wait
    for the run to finish.

    mossy_pattern/climbing_pattern (scalar or per-cell array, as accepted by
    node.inject_*_fiber_input()) are optional: leave them None (the default)
    to keep this function's original behavior -- whatever the caller already
    injected via node.inject_*_fiber_input() before calling this function
    stays on, unchanged, for the whole run. Pass a pattern to have this
    function manage that pathway's drive itself, confined to
    [*_start_ms, *_end_ms) (defaults: 0 and duration_ms, i.e. the whole run
    if the window bounds are left None) -- the current is switched on at
    *_start_ms and back to 0 at *_end_ms, rather than held on for the full
    duration.

    mossy_golgi_pattern (DESIGN.md): the mossy-fiber pathway now has two
    independent populations (granule-facing, sized node.n_mossy_fibers_
    granule, and Golgi-facing, sized node.n_mossy_fibers_golgi) -- mossy_
    pattern drives the former, mossy_golgi_pattern the latter, passed through
    to inject_mossy_fiber_input(mossy_pattern, mossy_golgi_pattern) as its
    I_nA/I_golgi_nA. Leave mossy_golgi_pattern None to fall back to
    inject_mossy_fiber_input's own default (scalar mossy_pattern reused for
    Golgi too; an array mean-collapsed to a uniform Golgi drive)."""
    n_cells = node.cells.n_nodes
    n_golgi = node.golgi.n_golgi
    n_steps = round(duration_ms / dt_ms)
    n_frames = n_steps // record_every

    mossy_on_step = round(mossy_start_ms / dt_ms) if mossy_start_ms is not None else 0
    mossy_off_step = round(mossy_end_ms / dt_ms) if mossy_end_ms is not None else n_steps
    climbing_on_step = round(climbing_start_ms / dt_ms) if climbing_start_ms is not None else 0
    climbing_off_step = round(climbing_end_ms / dt_ms) if climbing_end_ms is not None else n_steps

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / META_FILE,
        n_cells=n_cells,
        n_golgi=n_golgi,
        dt_ms=dt_ms,
        record_every=record_every,
        n_frames=n_frames,
        layers=np.array(LAYERS),
    )

    np.savez(
        output_dir / POSITIONS_FILE,
        node_x=node.node_x.astype(np.float32),
        node_y=node.node_y.astype(np.float32),
        golgi_x=node.golgi_x.astype(np.float32),
        golgi_y=node.golgi_y.astype(np.float32),
    )

    progress = np.lib.format.open_memmap(
        output_dir / PROGRESS_FILE, mode="w+", dtype=np.int64, shape=(1,)
    )
    progress[0] = 0
    progress.flush()

    memmaps = {
        layer: np.lib.format.open_memmap(
            output_dir / VOLTAGE_FILE_TEMPLATE.format(layer=layer),
            mode="w+",
            dtype=np.float16,
            shape=(n_frames, n_cells),
        )
        for layer in LAYERS
    }
    golgi_memmap = np.lib.format.open_memmap(
        output_dir / VOLTAGE_FILE_TEMPLATE.format(layer=GOLGI_LAYER),
        mode="w+",
        dtype=np.float16,
        shape=(n_frames, n_golgi),
    )

    frame = 0
    for step in range(n_steps):
        if mossy_pattern is not None and step == mossy_on_step:
            if mossy_golgi_pattern is not None:
                node.inject_mossy_fiber_input(mossy_pattern, mossy_golgi_pattern)
            else:
                node.inject_mossy_fiber_input(mossy_pattern)
        if mossy_pattern is not None and step == mossy_off_step:
            node.inject_mossy_fiber_input(0.0)
        if climbing_pattern is not None and step == climbing_on_step:
            node.inject_climbing_fiber_input(climbing_pattern)
        if climbing_pattern is not None and step == climbing_off_step:
            node.inject_climbing_fiber_input(0.0)

        node.step(dt_ms)
        if (step + 1) % record_every == 0:
            memmaps["granule"][frame] = node.cells.granule.get_voltage()
            memmaps["purkinje"][frame] = node.cells.purkinje.get_voltage()
            memmaps["stellate"][frame] = node.cells.stellate.get_voltage()
            golgi_memmap[frame] = node.golgi.get_voltage()
            frame += 1
            progress[0] = frame
            progress.flush()

    for m in memmaps.values():
        m.flush()
    golgi_memmap.flush()

    print(
        f"Recorded {n_frames} frames x {n_cells:,} cells x {len(LAYERS)} layers"
        f" (+ {n_golgi:,} Golgi cells) to {output_dir}"
    )


# ------------------------------------------------------------------------------
def read_progress(output_dir: Path) -> tuple[int, int]:
    """Read (frames_done, n_frames) for a recording in output_dir, in progress
    or finished. Only touches meta.npz and the 8-byte progress.npy -- never
    the (potentially huge) activity_<layer>.npy memmaps -- so it's cheap and
    safe to call from a separate process while the recording is still
    running (scripts/check_progress.py)."""
    n_frames = int(np.load(output_dir / META_FILE)["n_frames"])
    frames_done = int(np.load(output_dir / PROGRESS_FILE, mmap_mode="r")[0])
    return frames_done, n_frames
