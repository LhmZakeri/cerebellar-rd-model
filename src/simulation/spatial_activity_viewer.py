"""
SpatialActivityViewer -- per-cell scatter view of a recording made by
activity_recording.record_grid_activity(), colored by voltage (DESIGN.md).

Additive to, not a replacement for, ActivityViewer's per-cell-index line
graphs (CONTEXT.md's "Spatial activity view"): this reads the same on-disk
file contract, plus positions.npz, so a recording made once can be viewed
either way without re-recording, live or finished, from a separate process.

Three cell populations, each its own colormap family so they stay visually
distinct even where panels overlap (granule=Blues, Golgi=Reds, Purkinje=
Greens, stellate=Purples). One shared colorbar for the whole figure rather
than one per population, though: it's rendered in a neutral grayscale
(darker = higher voltage) rather than matching any one population's hue --
a generic magnitude reference, not a literal legend for each panel's actual
color, since the per-population hue is what visually separates populations.
  - Granular layer: Golgi and granule cells overlaid at their own positions
    -- safe, since they're never at the same coordinate, unlike Purkinje/
    stellate (CONTEXT.md's Node entry: granule/Purkinje/stellate share one
    position per Node index, since they're vertically synapsed). Golgi's
    own voltage is drawn as a small, opaque Reds-colored dot. Its diffusion
    wavefront (CONTEXT.md) is drawn as a dashed gray boundary curve -- the
    convex hull of whichever real Golgi cells were quiescent last frame
    (V[frame-1] < golgi_active_voltage_mv) and just crossed
    golgi_wavefront_delta_threshold_mv (default 40mV) this step: "newly
    recruited" cells, not merely "the most-changing ones this frame" (an
    earlier percentile-ranking version always found *some* cells even when
    nothing was really propagating -- this version can correctly show no
    hull at all when nothing crosses the bar). Not an interpolated field or
    a color/alpha overlay on Golgi's dot either (two earlier attempts: an
    RGB blend diluted both signals into an unreadable color, and a per-cell
    alpha halo was more legible but still just a shading cue, not a genuine
    curve; a hull over real cell positions only, no fabricated values
    between them, is what actually reads as "a wavefront" -- DESIGN.md's
    amendments).
  - Purkinje: alone, Greens.
  - Molecular (stellate): alone, Purples.
Purkinje/stellate stay in their own panels specifically because they share
granule's exact coordinates and would occlude if merged into one panel.
Every Node genuinely does host one granule + one Purkinje + one stellate
cell (DESIGN.md's uniform n_cells) -- so the same positions legitimately
recur across all three panels; that's the real model, not a rendering bug.

Two layouts, chosen via use_3d: a 2x2 grid of 2D panels (default, one quadrant
unused/hidden since there are only three panels), or one 3D axes with
granular layer at z=0, Purkinje at z=1, stellate at z=2 --
viable now that granule/Purkinje/stellate carry a real shared position
(DESIGN.md), unlike when DESIGN.md rejected 3D stacking for having
no spatial data to stack.

Discrete per-cell scatter points only, never interpolated into a continuous
field -- granule/Purkinje/stellate somata have no continuous tissue to
interpolate across. Reuses ActivityViewer's _DISPLAY_MAX_CELLS downsampling
for granule/Purkinje/stellate; Golgi is always shown at full population
(sparse enough -- ~1:430 -- that it never needs it). Measured cost at full
Phase 1 scale: ~53ms/redraw downsampled vs. ~3s/redraw undownsampled --
see DESIGN.md for the benchmark this is based on.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

from src.simulation.activity_recording import (
    GOLGI_LAYER,
    LAYERS,
    META_FILE,
    POSITIONS_FILE,
    PROGRESS_FILE,
    V_MAX,
    V_MIN,
    VOLTAGE_FILE_TEMPLATE,
)

_DISPLAY_MAX_CELLS = 5000  # matches activity_viewer.py's cap; visual only.
_DEFAULT_GOLGI_ACTIVE_VOLTAGE_MV = -20.0  # above this, a Golgi cell counts as already depolarized.
_DEFAULT_GOLGI_WAVEFRONT_DELTA_THRESHOLD_MV = 20.0  # minimum |deltaV| to count as "just recruited".

def _truncated(cmap, low: float = 0.3) -> LinearSegmentedColormap:
    """Drops the near-white bottom of a sequential colormap (matplotlib's
    Blues/Greens/Purples/Reds all start at ~white) so a resting/quiescent
    cell still renders as a visibly colored dot against the white figure
    background, not an invisible one -- only actual activation was showing
    up before, making a mostly-quiescent population look like empty space
    rather than "present but not firing"."""
    return LinearSegmentedColormap.from_list(f"{cmap.name}_trunc", cmap(np.linspace(low, 1.0, 256)))


# Perceptually distinct hue families, one per cell population -- chosen so
# granule (shares the granular panel with Golgi) and Golgi never overlap in
# hue, and so every population keeps one signature color everywhere it's
# drawn (DESIGN.md's color-scheme amendment). Truncated (see
# _truncated()) so voltage near V_MIN still reads as a visible dot.
_GRANULE_CMAP = _truncated(plt.cm.Blues)
_GOLGI_CMAP = _truncated(plt.cm.Reds)
_PURKINJE_CMAP = _truncated(plt.cm.Greens)
_STELLATE_CMAP = _truncated(plt.cm.Purples)
# The one shared colorbar is deliberately plain grayscale (darker = higher
# voltage) rather than any population's own hue -- it's a generic magnitude
# reference, not a literal readout of what's drawn in a given panel.
_COLORBAR_CMAP = plt.cm.Greys
_WAVEFRONT_COLOR = "#555555"  # neutral gray -- deliberately not a cell-type hue.

_CELL_MARKER_SIZE = 4  # fallback only -- _adaptive_marker_size() normally
# overrides this once real axes geometry is known (see show()); kept as a
# literal fallback so a marker size always exists even if that computation
# is ever skipped.
_GOLGI_DOT_SCALE = 1.6  # golgi's dot is drawn this much bigger (by area)
# than the adaptive per-cell marker -- Golgi is never downsampled (unlike
# granule/Purkinje/stellate) so at real scale (~1000s of Golgi cells) a much
# larger multiplier would swamp the plot; this keeps Golgi legible without
# dominating.
_MARKER_SIZE_MIN, _MARKER_SIZE_MAX = 2.0, 140.0  # points^2, sanity clamp


class SpatialActivityViewer:
    """Per-cell scatter graphs (granular/Purkinje/molecular), colored by
    voltage, over a recording made by activity_recording.record_grid_activity()."""

    def __init__(
        self,
        output_dir: Path,
        refresh_ms: int = 200,
        golgi_active_voltage_mv: float = _DEFAULT_GOLGI_ACTIVE_VOLTAGE_MV,
        golgi_wavefront_delta_threshold_mv: float = _DEFAULT_GOLGI_WAVEFRONT_DELTA_THRESHOLD_MV,
        use_3d: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.refresh_ms = refresh_ms
        if golgi_wavefront_delta_threshold_mv < 0.0:
            raise ValueError("golgi_wavefront_delta_threshold_mv must be >= 0")
        self.golgi_active_voltage_mv = golgi_active_voltage_mv
        self.golgi_wavefront_delta_threshold_mv = golgi_wavefront_delta_threshold_mv
        self.use_3d = use_3d

        meta_path = self.output_dir / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No recording found at {self.output_dir} (missing {META_FILE}). "
                "Point this at a directory a record_grid_activity() call has started writing to."
            )
        positions_path = self.output_dir / POSITIONS_FILE
        if not positions_path.exists():
            raise FileNotFoundError(
                f"No {POSITIONS_FILE} found at {self.output_dir}. Recordings made before "
                "DESIGN.md have no positions file -- re-record to use the spatial view."
            )

        meta = np.load(meta_path)
        self.n_cells = int(meta["n_cells"])
        self.n_golgi = int(meta["n_golgi"]) if "n_golgi" in meta else 0
        self.dt_ms = float(meta["dt_ms"])
        self.record_every = int(meta["record_every"])
        self.n_frames_total = int(meta["n_frames"])
        self.layers = [str(layer) for layer in meta["layers"]] if "layers" in meta else list(LAYERS)
        self.has_stellate = "stellate" in self.layers
        self.drive_onset_ms = float(meta["drive_onset_ms"]) if "drive_onset_ms" in meta else None
        climbing_pulse_onset_ms = (
            float(meta["climbing_pulse_onset_ms"]) if "climbing_pulse_onset_ms" in meta else float("nan")
        )
        self.climbing_pulse_onset_ms = (
            None if np.isnan(climbing_pulse_onset_ms) else climbing_pulse_onset_ms
        )

        positions = np.load(positions_path)
        self.node_x = positions["node_x"]
        self.node_y = positions["node_y"]
        self.golgi_x = positions["golgi_x"]
        self.golgi_y = positions["golgi_y"]

        self._progress = np.load(self.output_dir / PROGRESS_FILE, mmap_mode="r")
        self._voltages = {
            layer: np.load(self.output_dir / VOLTAGE_FILE_TEMPLATE.format(layer=layer), mmap_mode="r")
            for layer in self.layers
        }
        golgi_path = self.output_dir / VOLTAGE_FILE_TEMPLATE.format(layer=GOLGI_LAYER)
        self.has_golgi = self.n_golgi > 0 and golgi_path.exists()
        if self.has_golgi:
            self._voltages[GOLGI_LAYER] = np.load(golgi_path, mmap_mode="r")

        self._stride = max(1, self.n_cells // _DISPLAY_MAX_CELLS)
        self._node_idx = np.arange(0, self.n_cells, self._stride)

        self._frame = 0
        self._last_drawn_frame = -1  # sentinel: no valid frame index is negative, forces first draw
        # Single-step navigation is unusable at real-scale frame counts
        # (e.g. 40000 frames for a 2000ms/0.01ms/record_every=5 recording) --
        # scroll wheel / Page Up/Down jump by ~0.5% of the recording per
        # tick instead, scaling with however long the recording actually is.
        self._scroll_step_frames = max(1, self.n_frames_total // 200)

    def _n_ready(self) -> int:
        return int(self._progress[0])

    def _layer_voltage(self, layer: str, frame: int) -> np.ndarray:
        return self._voltages[layer][frame].astype(np.float32)

    def _ms_to_frame(self, ms: float) -> int:
        frame = round(ms / (self.record_every * self.dt_ms)) - 1
        return max(0, min(frame, self.n_frames_total - 1))

    def _panel_figsize(self, n_panels: int) -> tuple[float, float]:
        """Sizes the figure from the tissue's own aspect ratio instead of a
        fixed square-ish box -- a long, narrow strip (e.g. 3000x300um)
        rendered into a near-square panel leaves most of the panel as dead
        white space and squeezes the actual data into a thin sliver. Wide
        for elongated tissue, roughly square for the old discovery-scale
        square grid (backward-compatible look for existing recordings)."""
        x_span = max(float(self.node_x.max() - self.node_x.min()), 1.0)
        y_span = max(float(self.node_y.max() - self.node_y.min()), 1.0)
        data_aspect = x_span / y_span
        panel_width_in = float(np.clip(data_aspect * 2.2, 8.0, 20.0))
        panel_height_in = float(np.clip(panel_width_in / data_aspect, 1.6, 5.0))
        return panel_width_in, panel_height_in * n_panels + 2.2  # +legend/colorbar/title

    def _set_panel_extent(self, ax) -> None:
        pad_x = max((self.node_x.max() - self.node_x.min()) * 0.03, 5.0)
        pad_y = max((self.node_y.max() - self.node_y.min()) * 0.03, 5.0)
        ax.set_xlim(self.node_x.min() - pad_x, self.node_x.max() + pad_x)
        ax.set_ylim(self.node_y.min() - pad_y, self.node_y.max() + pad_y)
        ax.set_aspect("equal")

    def _adaptive_marker_size(self, fig, ax) -> float:
        """Marker area (points^2) chosen so adjacent cells' dots roughly
        touch, computed from the axes' ACTUAL rendered pixel geometry
        (requires a draw() first) rather than a fixed constant -- a fixed
        size that looked fine on the old ~300x300um/1024-cell grid becomes
        either invisible slivers (thousands of cells packed into a wide
        panel) or a blob (few cells stretched over a big panel) at a
        different tissue size/cell count, which is exactly what made
        activation hard to see on the new 3000x300um/9000-cell recording."""
        unique_x = np.unique(self.node_x)
        spacing_um = float(np.min(np.diff(unique_x))) if len(unique_x) > 1 else 10.0
        spacing_um *= self._stride
        x0, x1 = self.node_x.min(), self.node_x.min() + spacing_um
        y_mid = (self.node_y.min() + self.node_y.max()) / 2.0
        p0 = ax.transData.transform((x0, y_mid))
        p1 = ax.transData.transform((x1, y_mid))
        pixels_per_spacing = abs(p1[0] - p0[0])
        diameter_points = pixels_per_spacing * 72.0 / fig.dpi
        return float(np.clip(diameter_points**2, _MARKER_SIZE_MIN, _MARKER_SIZE_MAX))

    def _golgi_wavefront(self, frame: int) -> np.ndarray:
        """|V[frame] - V[frame-1]| per Golgi cell -- the diffusion wavefront
        (CONTEXT.md): high where voltage is actively changing (diffusion
        propagating through), near-zero where a cell has already settled or
        hasn't been reached yet, unlike raw voltage which stays high at an
        already-settled cell. Zero at frame 0 (no prior frame to compare)."""
        v_curr = self._layer_voltage(GOLGI_LAYER, frame)
        if frame == 0:
            return np.zeros_like(v_curr)
        v_prev = self._layer_voltage(GOLGI_LAYER, frame - 1)
        return np.abs(v_curr - v_prev)

    def _golgi_wavefront_hull(self, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Closed polygon boundary (x, y) around whichever real Golgi cells
        were just recruited into activity this step: quiescent last frame
        (V[frame-1] < golgi_active_voltage_mv) and now crossing
        golgi_wavefront_delta_threshold_mv -- traced through actual cell
        positions only, no interpolated values between them (DESIGN.md,
        keeping faith with Q5's "no fabricated continuous field" decision
        even for this curve). Deliberately not "the most-changing cells
        this frame" (an earlier percentile-ranking version always found
        *some* cells, even ones merely continuing an existing oscillation,
        not a genuine new recruitment event) -- this version can correctly
        return None when nothing is actually propagating, which is itself
        meaningful information. Also None at frame 0 (no prior frame to
        compare against) or with fewer than 3 qualifying cells (ConvexHull's
        minimum for a 2D polygon)."""
        if frame == 0:
            return None
        wavefront = self._golgi_wavefront(frame)
        v_prev = self._layer_voltage(GOLGI_LAYER, frame - 1)
        was_quiescent = v_prev < self.golgi_active_voltage_mv
        just_crossed = wavefront >= self.golgi_wavefront_delta_threshold_mv
        active = was_quiescent & just_crossed
        if active.sum() < 3:
            return None
        points = np.column_stack([self.golgi_x[active], self.golgi_y[active]])
        hull = ConvexHull(points)
        order = np.append(hull.vertices, hull.vertices[0])  # close the loop
        return points[order, 0], points[order, 1]

    def _legend_handles(self) -> list:
        """One shared legend for the whole figure, listing every
        population's own hue (the shared colorbar is plain grayscale and
        doesn't carry this information on its own)."""
        handles = [
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=_GRANULE_CMAP(0.7), markeredgecolor="none",
                   label=f"granule voltage ({_GRANULE_CMAP.name})"),
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=_PURKINJE_CMAP(0.7), markeredgecolor="none",
                   label=f"purkinje voltage ({_PURKINJE_CMAP.name})"),
        ]
        if self.has_stellate:
            handles.append(
                Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor=_STELLATE_CMAP(0.7), markeredgecolor="none",
                       label=f"stellate voltage ({_STELLATE_CMAP.name})")
            )
        if self.has_golgi:
            handles.append(
                Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor=_GOLGI_CMAP(0.7), markeredgecolor="none",
                       label=f"golgi voltage ({_GOLGI_CMAP.name})")
            )
            handles.append(
                Line2D([0], [0], color=_WAVEFRONT_COLOR, linestyle="--",
                       label=f"golgi wavefront boundary (newly active, "
                             f"deltaV >= {self.golgi_wavefront_delta_threshold_mv:.0f}mV)")
            )
        return handles

    def show(self) -> None:
        n_ready = self._n_ready()
        self._frame = max(n_ready - 1, 0)
        if self.use_3d:
            self._show_3d()
        else:
            self._show_2d()

    # ------------------------------------------------------------------------
    def _show_2d(self) -> None:
        # One row per layer, stacked vertically (granular, purkinje,
        # molecular/stellate if present) -- not a 2x2 grid. A 2x2 grid left
        # one quadrant permanently unused and, worse, forced every panel
        # into a near-square box regardless of the tissue's actual shape:
        # for a long, narrow strip that means most of each panel is dead
        # white space with the real data squeezed into a thin sliver.
        # Stacking rows lets each panel's own width span the figure and use
        # _panel_figsize()'s tissue-aspect-aware sizing properly, matching
        # ActivityViewer's own established "stacked subplots" layout.
        panel_specs = [
            ("granular", "granular layer (golgi + granule)"),
            ("purkinje", "purkinje"),
        ]
        if self.has_stellate:
            panel_specs.append(("stellate", "molecular (stellate)"))
        n_panels = len(panel_specs)

        figsize = self._panel_figsize(n_panels)
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)
        axes = axes[:, 0]
        axes_by_layer = dict(zip((spec[0] for spec in panel_specs), axes))
        ax_granular = axes_by_layer["granular"]
        ax_purkinje = axes_by_layer["purkinje"]
        ax_stellate = axes_by_layer.get("stellate")

        for ax, (_, title) in zip(axes, panel_specs):
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            ax.set_title(title)
            self._set_panel_extent(ax)

        fig.canvas.draw()  # finalize layout/transforms before measuring pixel geometry
        marker_size = self._adaptive_marker_size(fig, ax_granular)
        golgi_dot_size = marker_size * _GOLGI_DOT_SCALE

        node_x_ds, node_y_ds = self.node_x[self._node_idx], self.node_y[self._node_idx]
        voltage_norm = Normalize(vmin=V_MIN, vmax=V_MAX)
        placeholder = np.zeros(len(node_x_ds), dtype=np.float32)

        granule_sc = ax_granular.scatter(
            node_x_ds, node_y_ds, c=placeholder, s=marker_size,
            cmap=_GRANULE_CMAP, norm=voltage_norm,
        )

        golgi_hull_line = golgi_dot_sc = None
        if self.has_golgi:
            (golgi_hull_line,) = ax_granular.plot(
                [], [], color=_WAVEFRONT_COLOR, linestyle="--", linewidth=1.5
            )
            golgi_dot_sc = ax_granular.scatter(
                self.golgi_x, self.golgi_y, s=golgi_dot_size,
                c=np.zeros(self.n_golgi, dtype=np.float32), cmap=_GOLGI_CMAP, norm=voltage_norm,
            )

        purkinje_sc = ax_purkinje.scatter(
            node_x_ds, node_y_ds, c=placeholder, s=marker_size,
            cmap=_PURKINJE_CMAP, norm=voltage_norm,
        )

        stellate_sc = None
        if self.has_stellate:
            stellate_sc = ax_stellate.scatter(
                node_x_ds, node_y_ds, c=placeholder, s=marker_size,
                cmap=_STELLATE_CMAP, norm=voltage_norm,
            )

        # Vertical layout budget scales with panel count instead of a fixed
        # split tuned for exactly 2 rows -- legend/colorbar/title need
        # roughly the same absolute vertical space regardless of how many
        # panels are stacked beneath them, so their fraction of the total
        # figure height shrinks as n_panels grows.
        top = 1.0 - 1.9 / figsize[1]
        fig.subplots_adjust(top=top, bottom=0.06, hspace=0.55)
        fig.legend(
            handles=self._legend_handles(),
            loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize="large", ncol=2,
        )
        # One shared colorbar, placed horizontally after the last stack
        # instead of one per panel -- kept thin (small fraction) so it
        # reads as a compact scale reference, not a fourth panel. Deliberately
        # plain grayscale (_COLORBAR_CMAP), not any one population's own hue
        # -- a standalone ScalarMappable since no single scatter's cmap
        # should stand in for all four populations.
        colorbar_mappable = plt.cm.ScalarMappable(norm=voltage_norm, cmap=_COLORBAR_CMAP)
        colorbar_axes = [ax_granular, ax_purkinje] + ([ax_stellate] if self.has_stellate else [])
        fig.colorbar(
            colorbar_mappable, ax=colorbar_axes,
            orientation="horizontal", location="bottom", shrink=0.5, fraction=0.03, pad=0.35 / n_panels,
            label="voltage (mV) -- darker = higher",
        )

        def draw(frame: int) -> None:
            granule_sc.set_array(self._layer_voltage("granule", frame)[self._node_idx])
            purkinje_sc.set_array(self._layer_voltage("purkinje", frame)[self._node_idx])
            if stellate_sc is not None:
                stellate_sc.set_array(self._layer_voltage("stellate", frame)[self._node_idx])
            if golgi_dot_sc is not None:
                golgi_dot_sc.set_array(self._layer_voltage(GOLGI_LAYER, frame))
                hull = self._golgi_wavefront_hull(frame)
                if hull is not None:
                    golgi_hull_line.set_data(hull[0], hull[1])
                else:
                    golgi_hull_line.set_data([], [])

        self._run_event_loop(fig, draw)

    # ------------------------------------------------------------------------
    def _show_3d(self) -> None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_zlabel("layer")
        ax.set_zticks([0, 1, 2])
        ax.set_zticklabels(["granular", "purkinje", "molecular"])

        node_x_ds, node_y_ds = self.node_x[self._node_idx], self.node_y[self._node_idx]
        z_granular = np.zeros(len(node_x_ds))
        z_purkinje = np.ones(len(node_x_ds))
        voltage_norm = Normalize(vmin=V_MIN, vmax=V_MAX)
        placeholder = np.zeros(len(node_x_ds), dtype=np.float32)

        granule_sc = ax.scatter(
            node_x_ds, node_y_ds, z_granular, c=placeholder, s=_CELL_MARKER_SIZE,
            cmap=_GRANULE_CMAP, norm=voltage_norm, depthshade=False,
        )
        purkinje_sc = ax.scatter(
            node_x_ds, node_y_ds, z_purkinje, c=placeholder, s=_CELL_MARKER_SIZE,
            cmap=_PURKINJE_CMAP, norm=voltage_norm, depthshade=False,
        )
        stellate_sc = None
        if self.has_stellate:
            z_stellate = np.full(len(node_x_ds), 2.0)
            stellate_sc = ax.scatter(
                node_x_ds, node_y_ds, z_stellate, c=placeholder, s=_CELL_MARKER_SIZE,
                cmap=_STELLATE_CMAP, norm=voltage_norm, depthshade=False,
            )

        golgi_hull_line = golgi_dot_sc = None
        if self.has_golgi:
            z_golgi = np.zeros(self.n_golgi)
            (golgi_hull_line,) = ax.plot(
                [], [], [], color=_WAVEFRONT_COLOR, linestyle="--", linewidth=1.5
            )
            golgi_dot_sc = ax.scatter(
                self.golgi_x, self.golgi_y, z_golgi, s=_CELL_MARKER_SIZE * _GOLGI_DOT_SCALE,
                c=np.zeros(self.n_golgi, dtype=np.float32), cmap=_GOLGI_CMAP, norm=voltage_norm,
                depthshade=False,
            )

        # One shared colorbar, at left -- kept thin (small fraction) so it
        # reads as a compact scale reference, not a fourth panel. Deliberately
        # plain grayscale (_COLORBAR_CMAP), not any one population's own hue
        # -- a standalone ScalarMappable since no single scatter's cmap
        # should stand in for all four populations.
        colorbar_mappable = plt.cm.ScalarMappable(norm=voltage_norm, cmap=_COLORBAR_CMAP)
        fig.colorbar(
            colorbar_mappable, ax=ax, location="left", shrink=0.5, fraction=0.04, pad=0.1,
            label="voltage (mV) -- darker = higher",
        )
        # One shared legend, placed after the title (default y=0.98) and
        # before the stack, with `top` further below still so there's
        # visible clearance before the axes, not just enough for the legend
        # box itself.
        fig.subplots_adjust(top=0.78)
        fig.legend(
            handles=self._legend_handles(),
            loc="upper center", bbox_to_anchor=(0.5, 0.90), fontsize="large",
        )

        def draw(frame: int) -> None:
            granule_sc.set_array(self._layer_voltage("granule", frame)[self._node_idx])
            purkinje_sc.set_array(self._layer_voltage("purkinje", frame)[self._node_idx])
            if stellate_sc is not None:
                stellate_sc.set_array(self._layer_voltage("stellate", frame)[self._node_idx])
            if golgi_dot_sc is not None:
                golgi_dot_sc.set_array(self._layer_voltage(GOLGI_LAYER, frame))
                hull = self._golgi_wavefront_hull(frame)
                if hull is not None:
                    golgi_hull_line.set_data_3d(hull[0], hull[1], np.zeros_like(hull[0]))
                else:
                    golgi_hull_line.set_data_3d([], [], [])

        self._run_event_loop(fig, draw)

    # ------------------------------------------------------------------------
    def _run_event_loop(self, fig, draw) -> None:
        """Shared frame-stepping/timer/keyboard wiring for both _show_2d and
        _show_3d -- only what gets drawn differs between them."""

        def redraw() -> None:
            n_ready = self._n_ready()
            frame = min(self._frame, n_ready - 1) if n_ready else None

            if frame == self._last_drawn_frame:
                return
            self._last_drawn_frame = frame

            if frame is not None:
                draw(frame)

            if frame is None:
                fig.suptitle(f"0/{self.n_frames_total} frames ready", y=0.998)
            else:
                t_ms = (frame + 1) * self.record_every * self.dt_ms
                fig.suptitle(
                    f"t = {t_ms:.2f} ms   frame {frame + 1}/{self.n_frames_total}"
                    f"   ({n_ready}/{self.n_frames_total} frames ready)",
                    y=0.998,
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
