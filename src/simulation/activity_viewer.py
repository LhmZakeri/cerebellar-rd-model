"""
ActivityViewer -- per-layer voltage-vs-cell-index graphs over a recording
made by activity_recording.record_grid_activity(). Reads whatever layers
meta.npz lists and draws one line-graph panel per layer, plus a separate
Golgi panel (DESIGN.md).

Fully decoupled from recording (activity_recording.py): it only reads that
module's on-disk file contract, so it can point at a still-running
recording (in another process) or a finished one. progress.npy caps how
far it steps, since an unwritten pre-allocated frame is zeros, not "no
data."

Layers are stacked subplots (granule at the bottom, molecular/stellate
above it, Golgi last -- matching real cortical layer order), each a plain
voltage-vs-cell-index line graph, not a reshaped spatial surface: granule/
Purkinje/stellate carry no grid position of their own (DESIGN.md,
DESIGN.md). Golgi is a different length (n_golgi, sparse) from the
other three (n_cells), so it gets its own x-axis/stride.

Line data is downsampled to at most _DISPLAY_MAX_CELLS points per panel,
and a redraw only fires when the ready-frame count changes.

Navigation is keyboard-only: Left/Right steps one recorded frame, Home/End
jumps to the first/last frame. The current simulated time and frame index
are shown at the top of the panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.simulation.activity_recording import (
    GOLGI_LAYER,
    LAYERS,
    META_FILE,
    PROGRESS_FILE,
    V_MAX,
    V_MIN,
    VOLTAGE_FILE_TEMPLATE,
)

_DISPLAY_MAX_CELLS = 5000  # cap on cells/Golgi cells plotted per panel; visual only.


class ActivityViewer:
    """Per-layer voltage-vs-cell-index graphs (plus a Golgi panel) over a
    recording made by activity_recording.record_grid_activity()."""

    def __init__(self, output_dir: Path, refresh_ms: int = 200) -> None:
        self.output_dir = Path(output_dir)
        self.refresh_ms = refresh_ms

        meta_path = self.output_dir / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No recording found at {self.output_dir} (missing {META_FILE}). "
                "Point this at a directory a record_grid_activity() call has started writing to."
            )
        meta = np.load(meta_path)
        self.n_cells = int(meta["n_cells"])
        self.n_golgi = int(meta["n_golgi"]) if "n_golgi" in meta else 0
        self.dt_ms = float(meta["dt_ms"])
        self.record_every = int(meta["record_every"])
        self.n_frames_total = int(meta["n_frames"])
        self.layers = (
            [str(layer) for layer in meta["layers"]]
            if "layers" in meta
            else list(LAYERS)
        )
        self.drive_onset_ms = float(meta["drive_onset_ms"]) if "drive_onset_ms" in meta else None
        climbing_pulse_onset_ms = (
            float(meta["climbing_pulse_onset_ms"]) if "climbing_pulse_onset_ms" in meta else float("nan")
        )
        self.climbing_pulse_onset_ms = (
            None if np.isnan(climbing_pulse_onset_ms) else climbing_pulse_onset_ms
        )

        self._progress = np.load(
            self.output_dir / PROGRESS_FILE, mmap_mode="r"
        )  # frames_done
        self._voltages = {
            layer: np.load(
                self.output_dir / VOLTAGE_FILE_TEMPLATE.format(layer=layer),
                mmap_mode="r",
            )
            for layer in self.layers
        }
        self._stride = max(
            1, self.n_cells // _DISPLAY_MAX_CELLS
        )  # downsample of the space

        # Golgi is sparse (DESIGN.md) -- a genuinely different length
        # from the other layers, so it gets its own stride/x-axis rather
        # than sharing self.n_cells/self._stride. Older recordings (made
        # before DESIGN.md added Golgi recording) have no golgi file --
        # self.has_golgi lets show() skip that panel gracefully.
        golgi_path = self.output_dir / VOLTAGE_FILE_TEMPLATE.format(layer=GOLGI_LAYER)
        self.has_golgi = self.n_golgi > 0 and golgi_path.exists()
        if self.has_golgi:
            self._voltages[GOLGI_LAYER] = np.load(golgi_path, mmap_mode="r")
            self._golgi_stride = max(1, self.n_golgi // _DISPLAY_MAX_CELLS)

        self._frame = 0
        self._last_drawn_frame = (
            -1
        )  # sentinel: no valid frame index is negative, forces first draw
        # Single-step navigation is unusable at real-scale frame counts --
        # scroll wheel / Page Up/Down jump by ~0.5% of the recording per
        # tick instead, scaling with however long the recording actually is.
        self._scroll_step_frames = max(1, self.n_frames_total // 200)

    def _n_ready(self) -> int:
        return int(self._progress[0])

    def _frame_values(self, layer: str, frame: int) -> np.ndarray:
        stride = self._golgi_stride if layer == GOLGI_LAYER else self._stride
        return self._voltages[layer][frame].astype(np.float32)[::stride]

    def _timestamp_ms(self, frame: int) -> float:
        """How far into the simulation that frame represents (ms)"""
        return (frame + 1) * self.record_every * self.dt_ms

    def _ms_to_frame(self, ms: float) -> int:
        frame = round(ms / (self.record_every * self.dt_ms)) - 1
        return max(0, min(frame, self.n_frames_total - 1))

    def show(self) -> None:
        n_ready = self._n_ready()
        self._frame = max(n_ready - 1, 0)

        panels = list(self.layers) + ([GOLGI_LAYER] if self.has_golgi else [])
        cell_idx = np.arange(0, self.n_cells, self._stride)
        golgi_idx = (
            np.arange(0, self.n_golgi, self._golgi_stride) if self.has_golgi else None
        )

        fig, axes = plt.subplots(
            len(panels), 1, sharex=False, figsize=(8, 2.5 * len(panels))
        )
        axes = np.atleast_1d(axes)

        lines: dict[str, object] = {}
        for ax, layer in zip(axes, panels):
            x = golgi_idx if layer == GOLGI_LAYER else cell_idx
            ax.set_ylim(V_MIN, V_MAX)
            ax.set_ylabel(f"{layer}\nV (mV)")
            ax.set_xlabel("golgi index" if layer == GOLGI_LAYER else "cell index")
            (lines[layer],) = ax.plot(x, np.zeros_like(x, dtype=np.float32))

        def draw_layer(layer: str, frame: int | None) -> None:
            n_points = len(golgi_idx) if layer == GOLGI_LAYER else len(cell_idx)
            values = (
                self._frame_values(layer, frame)
                if frame is not None
                else np.zeros(n_points, dtype=np.float32)
            )
            lines[layer].set_ydata(values)

        def redraw() -> None:
            n_ready = self._n_ready()
            frame = min(self._frame, n_ready - 1) if n_ready else None

            # A no-op redraw when the displayed frame hasn't moved keeps a
            # background timer tick from doing any work at all.
            if frame == self._last_drawn_frame:
                return
            self._last_drawn_frame = frame

            for layer in panels:
                draw_layer(layer, frame)

            if frame is None:
                fig.suptitle(f"0/{self.n_frames_total} frames ready")
            else:
                t_ms = self._timestamp_ms(frame)
                fig.suptitle(
                    f"t = {t_ms:.2f} ms   frame {frame + 1}/{self.n_frames_total}"
                    f"   ({n_ready}/{self.n_frames_total} frames ready)"
                )
            fig.canvas.draw_idle()

        def on_key_press(event) -> None:
            if event.key == "right":
                self._frame = min(self._frame + 1, self.n_frames_total - 1)
                redraw()
            elif event.key == "left":
                self._frame = max(self._frame - 1, 0)
                redraw()
            elif event.key == "pageup":
                self._frame = min(self._frame + self._scroll_step_frames, self.n_frames_total - 1)
                redraw()
            elif event.key == "pagedown":
                self._frame = max(self._frame - self._scroll_step_frames, 0)
                redraw()
            elif event.key == "home":
                self._frame = 0
                redraw()
            elif event.key == "end":
                self._frame = self.n_frames_total - 1
                redraw()
            elif event.key == "s" and self.drive_onset_ms is not None:
                self._frame = self._ms_to_frame(self.drive_onset_ms)
                redraw()
            elif event.key == "c" and self.climbing_pulse_onset_ms is not None:
                self._frame = self._ms_to_frame(self.climbing_pulse_onset_ms)
                redraw()

        def on_scroll(event) -> None:
            if event.button == "up":
                self._frame = min(self._frame + self._scroll_step_frames, self.n_frames_total - 1)
                redraw()
            elif event.button == "down":
                self._frame = max(self._frame - self._scroll_step_frames, 0)
                redraw()

        def on_timer() -> None:
            # Only does anything once more frames have actually become
            # ready -- see redraw()'s early-return above.
            redraw()

        fig.canvas.mpl_connect("key_press_event", on_key_press)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        timer = fig.canvas.new_timer(interval=self.refresh_ms)
        timer.add_callback(on_timer)
        timer.start()

        redraw()
        help_lines = [
            "Keys: Left/Right arrows step one frame backward/forward, "
            f"Page Up/Down or mouse scroll jump {self._scroll_step_frames} frames, "
            "Home/End jump to the first/last frame.",
        ]
        if self.drive_onset_ms is not None:
            help_lines.append(f"'s' jumps to drive onset ({self.drive_onset_ms:.0f} ms).")
        if self.climbing_pulse_onset_ms is not None:
            help_lines.append(f"'c' jumps to climbing-fiber pulse onset ({self.climbing_pulse_onset_ms:.0f} ms).")
        print(" ".join(help_lines))
        plt.show()
        timer.stop()
