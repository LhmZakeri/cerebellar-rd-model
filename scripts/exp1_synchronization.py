"""
Experiment 1 of 3 (Sou11-style synchronization protocol, Scientific Goal A):
tests whether the Golgi<->Golgi gap-junction network produces population
oscillations, by comparing identical runs with GJ off vs on. NOT mixed with
the chaos test (Experiment 2, scripts/exp2_chaos_test.py) -- per the user's
explicit instruction not to mix synchronization and chaos in the first run.

Protocol (fixed, not swept here):
  0-1000 ms   no mossy-fiber input at all.
  1000-2000 ms all mossy fibers active simultaneously, each firing its own
              INDEPENDENT random (Poisson-approximated) spike train at the
              same mean rate (--rate-hz) -- not one fiber at a time.
  Climbing fiber OFF throughout by default (CLIMBING_STRENGTH_NA = 0.0).

Experiment 3 add-on (DESIGN.md): an optional climbing-fiber pulse to a
selected Purkinje-cell group, via --climbing-pulse-na (0.0 = off, the
default -- identical to Experiment 1's own protocol with zero behavior
change). Lives in this same script rather than a separate one, since the
user's spec describes Experiment 3 as Experiment 1 (or 2) PLUS a
climbing-fiber pulse, not an independent protocol -- "for the first two
experiments, keep climbing fibers off" is honored by the off-by-default.

Same cell kernels/synapse equations as the full-scale model (default
GridCouplingParams() except g_gap_nS, which IS the thing being compared).
Network size 1024 cells, golgi_ratio=0.008 (the elevated discovery-phase
ratio established in scripts/discovery_base_run.py -- this experiment
specifically needs a real Golgi<->Golgi network to test, so the true 1/430
ratio, which gave only ~2 Golgi cells at this scale, would defeat the point).

Mossy-fiber drive: DESIGN.md's inject_mossy_fiber_input(I_nA, I_golgi_nA)
is called every simulation step with a live, spike-train-filtered current
(src/simulation/poisson_drive.py) -- confirmed safe/cheap to call every step
(pure state-setting, no dynamical side effects). This bypasses
activity_recording.py's record_grid_activity() (which only supports one
on/off window per pathway per run), following the same hand-rolled-loop
precedent scripts/demo_complex_cascade.py already established for the same
reason.

Analysis (DESIGN.md's new population spike-metrics): Golgi/granule/
Purkinje rasters, pooled PSTH, Golgi power spectrum (scipy.signal.welch),
Golgi pairwise correlation, Purkinje firing rate -- per GJ condition, always
saved as a JSON summary + NPZ of derived arrays. The detailed 6-panel PNG
and full raw-voltage recording (DESIGN.md) are BOTH opt-in
(--save-figure / --save-raw-voltage, both default off) -- a full grid run
produces one of these per grid point, which stopped being useful at 20-40
grid points; use scripts/summarize_grid_results.py for a whole-grid
overview instead, and reach for --save-figure/--save-raw-voltage only for a
specific condition you already know you want to inspect closely.

Usage:
    # Fast correctness smoke test (seconds, not minutes)
    python scripts/exp1_synchronization.py --n-cells 128 --pre-ms 20 --drive-ms 20 --gj both

    # Real validation run (~5-10 min total for both conditions)
    python scripts/exp1_synchronization.py --n-cells 1024 --rate-hz 20 --seed 0 --gj both

    # A specific condition, with the detailed figure and raw voltage for
    # scripts/view_activity.py (DESIGN.md)
    python scripts/exp1_synchronization.py --n-cells 1024 --rate-hz 40 --seed 0 --gj both \\
        --save-figure --save-raw-voltage
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.simulation.activity_recording import (
    GOLGI_LAYER,
    META_FILE,
    POSITIONS_FILE,
    PROGRESS_FILE,
    VOLTAGE_FILE_TEMPLATE,
)
from src.simulation.coupling_params import GridCouplingParams
from src.simulation.geometry import FlatGrid
from src.simulation.grid_node_batch import GridNodeBatch
from src.simulation.poisson_drive import ExponentialCurrentFilter, generate_poisson_spike_train
from src.simulation.spike_metrics import (
    binned_spike_counts,
    mean_offdiag_correlation,
    pairwise_correlation,
    pooled_isi_cv,
    population_power_spectrum,
    population_spike_times_ms,
    psth,
)

RESOLUTION_UM = 10.0
GOLGI_RATIO = 1.0 / 36.0  # Discovery-scale golgi_ratio (CONTEXT.md) -- a
# COMPUTATIONALLY ENLARGED ratio, not the biological one (real rat anatomy
# ~1:430), matching Sou11's own reduced-network methodology (2.7%, the top
# of their range) for obtaining observable population oscillations in a
# small model. Reverted to the true 1:430 ratio for the eventual 10,000+
# node confirmation run.
DT_MS = 0.01
CLIMBING_STRENGTH_NA = 0.0  # OFF throughout -- Experiment 3 (deferred) adds this back.
SPIKE_THRESHOLD_MV = -20.0  # matches this repo's existing convention.

ONSET_BUFFER_MS = 50.0  # trim this much off the start of the drive window
# before computing spectrum/correlation, so the sudden stimulus-onset
# transient doesn't dominate the spectral estimate.
BIN_MS = 5.0  # PSTH/spectrum/correlation bin width.

DRY_RUN_MS = 200.0  # drive-amplitude pre-flight probe duration.
DRY_RUN_SEED = 12345
WALL_CLOCK_BUDGET_S_PER_CONDITION = 6 * 60.0  # abort before a condition that
# would blow the ~5-10 minute total target for both conditions combined.

RASTER_DISPLAY_CELLS = 150  # subsample granule/Purkinje rasters to this many
# cells for figure legibility/render speed -- analysis (spectra/correlation)
# always uses the full population regardless of this.

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "exp1_synchronization"


def _square_grid(n_cells: int) -> FlatGrid:
    """A grid sized so its node count is close to n_cells -- golgi_ratio then
    applies against ~n_cells nodes. Duplicated from discovery_base_run.py
    (DESIGN.md): no third caller yet to justify extracting it."""
    n_side = max(2, round(n_cells**0.5))
    side_um = n_side * RESOLUTION_UM
    return FlatGrid(width_um=side_um, height_um=side_um, resolution_um=RESOLUTION_UM)


def _build_geometry(args: argparse.Namespace) -> FlatGrid:
    """Square by default (_square_grid), or an explicit width_um x height_um
    rectangle when both --width-um/--height-um are given -- the discovery-
    scale square (a few hundred um/side) is too small to visually read a
    propagating wavefront on; a long, narrow strip is easier to look at and
    also closer to the eventual real folium's long-ridge shape (DESIGN.md).
    Caller is responsible for setting --n-cells to match the resulting grid
    node count so golgi_ratio's density assumption (~n_cells nodes) still
    holds, same as _square_grid's own convention."""
    if args.width_um is not None and args.height_um is not None:
        return FlatGrid(width_um=args.width_um, height_um=args.height_um, resolution_um=RESOLUTION_UM)
    return _square_grid(args.n_cells)


def _build_node(args: argparse.Namespace, g_gap_nS: float, golgi_seed: int, connectivity_seed: int) -> GridNodeBatch:
    geometry = _build_geometry(args)
    return GridNodeBatch(
        geometry,
        golgi_seed=golgi_seed,
        n_cells=args.n_cells,
        connectivity_seed=connectivity_seed,
        golgi_ratio=args.golgi_ratio,
        heterogeneity_seed=args.heterogeneity_seed,
        coupling=GridCouplingParams(
            g_gap_nS=g_gap_nS,
            golgi_granule_divergence=args.golgi_granule_divergence,
            mossy_to_golgi_divergence=args.mossy_to_golgi_divergence,
            distance_decay_per_um=args.distance_decay_per_um,
        ),
    )


def _dry_run_check(args: argparse.Namespace) -> tuple[float, bool, float, float]:
    """Cheap, independent pre-flight probe (DRY_RUN_MS of CONTINUOUS drive
    from a fresh network, not embedded in the real 0-1000/1000-2000ms
    timeline -- the silent 0-1000ms window would otherwise have to be
    fast-forwarded through twice, doubling cost for no benefit, since the
    question this answers ("does this amplitude/tau ever reach spike
    threshold at this rate") doesn't depend on the silent baseline at all).

    Returns (steps_per_sec, reached_threshold, max_V_granule, max_V_golgi).
    reached_threshold=False is a real warning worth seeing before committing
    the full run's wall-clock, not necessarily a bug -- could be a genuine
    "this rate/amplitude doesn't drive the network" finding.
    """
    node = _build_node(args, g_gap_nS=1.0, golgi_seed=DRY_RUN_SEED, connectivity_seed=DRY_RUN_SEED)
    node.inject_climbing_fiber_input(0.0)

    n_dry_steps = round(DRY_RUN_MS / args.dt_ms)
    spikes_g = generate_poisson_spike_train(
        node.n_mossy_fibers_granule, n_dry_steps, args.dt_ms, args.rate_hz, seed=DRY_RUN_SEED + 1
    )
    spikes_go = generate_poisson_spike_train(
        node.n_mossy_fibers_golgi, n_dry_steps, args.dt_ms, args.rate_hz, seed=DRY_RUN_SEED + 2
    )
    filt_g = ExponentialCurrentFilter(node.n_mossy_fibers_granule, args.mf_amplitude_na, args.mf_tau_ms)
    filt_go = ExponentialCurrentFilter(node.n_mossy_fibers_golgi, args.mf_amplitude_na, args.mf_tau_ms)

    max_v_granule, max_v_golgi = -np.inf, -np.inf
    t0 = time.perf_counter()
    for step in range(n_dry_steps):
        I_g = filt_g.step(args.dt_ms, spikes_g[step])
        I_go = filt_go.step(args.dt_ms, spikes_go[step])
        node.inject_mossy_fiber_input(I_g, I_go)
        node.step(args.dt_ms)
        max_v_granule = max(max_v_granule, float(node.cells.granule.get_voltage().max()))
        max_v_golgi = max(max_v_golgi, float(node.golgi.get_voltage().max()))
    elapsed = time.perf_counter() - t0
    steps_per_sec = n_dry_steps / elapsed
    reached_threshold = max_v_granule > SPIKE_THRESHOLD_MV or max_v_golgi > SPIKE_THRESHOLD_MV
    return steps_per_sec, reached_threshold, max_v_granule, max_v_golgi


def _save_raw_voltage(
    output_dir: Path,
    node: GridNodeBatch,
    t_ms: np.ndarray,
    V_granule: np.ndarray,
    V_purkinje: np.ndarray,
    V_golgi: np.ndarray,
    V_stellate: np.ndarray,
    dt_ms: float,
    record_every: int,
    drive_onset_ms: float,
    climbing_pulse_onset_ms: float | None,
) -> None:
    """Writes activity_recording.py's exact file contract (DESIGN.md)
    so scripts/view_activity.py can open this recording directly. Granule,
    Purkinje, Golgi, and stellate -- stellate is outside this experiment's
    own analysis scope but is included here so the raw-voltage case can be
    viewed in full. This is a direct array-to-memmap write, not a live
    per-step recording: the voltage arrays are already fully computed by
    the time this is called, since this script's hand-rolled loop (not
    record_grid_activity() itself, which can't handle the time-varying
    per-step mossy-fiber injection this experiment needs -- DESIGN.md)
    already ran the whole simulation before this point.

    drive_onset_ms/climbing_pulse_onset_ms are stashed in meta.npz purely
    for the viewers' "jump to stimulation start" keys -- climbing_pulse_onset_ms
    is stored as NaN (not omitted) when no climbing-fiber pulse is active,
    since np.savez keys must be unconditionally present for a fixed reader
    contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_frames = len(t_ms)
    n_cells = V_granule.shape[1]
    n_golgi = V_golgi.shape[1]

    np.savez(
        output_dir / META_FILE,
        n_cells=n_cells, n_golgi=n_golgi, dt_ms=dt_ms, record_every=record_every,
        n_frames=n_frames, layers=np.array(["granule", "purkinje", "stellate"]),
        drive_onset_ms=drive_onset_ms,
        climbing_pulse_onset_ms=(
            climbing_pulse_onset_ms if climbing_pulse_onset_ms is not None else np.nan
        ),
    )
    np.savez(
        output_dir / POSITIONS_FILE,
        node_x=node.node_x.astype(np.float32), node_y=node.node_y.astype(np.float32),
        golgi_x=node.golgi_x.astype(np.float32), golgi_y=node.golgi_y.astype(np.float32),
    )
    progress = np.lib.format.open_memmap(
        output_dir / PROGRESS_FILE, mode="w+", dtype=np.int64, shape=(1,)
    )
    progress[0] = n_frames
    progress.flush()

    for layer, V in (("granule", V_granule), ("purkinje", V_purkinje), ("stellate", V_stellate)):
        mm = np.lib.format.open_memmap(
            output_dir / VOLTAGE_FILE_TEMPLATE.format(layer=layer),
            mode="w+", dtype=np.float16, shape=(n_frames, n_cells),
        )
        mm[:] = V.astype(np.float16)
        mm.flush()

    golgi_mm = np.lib.format.open_memmap(
        output_dir / VOLTAGE_FILE_TEMPLATE.format(layer=GOLGI_LAYER),
        mode="w+", dtype=np.float16, shape=(n_frames, n_golgi),
    )
    golgi_mm[:] = V_golgi.astype(np.float16)
    golgi_mm.flush()

    print(f"saved raw voltage (view_activity.py-compatible): {output_dir}")


def run_condition(
    args: argparse.Namespace, gj_label: str, g_gap_nS: float, output_dir: Path
) -> dict:
    print(f"\n=== GJ {gj_label} (g_gap_nS={g_gap_nS}) ===")
    node = _build_node(args, g_gap_nS=g_gap_nS, golgi_seed=args.seed, connectivity_seed=args.seed)
    n_golgi = node.golgi.n_golgi
    print(f"n_golgi={n_golgi}, n_mossy_fibers_granule={node.n_mossy_fibers_granule}, "
          f"n_mossy_fibers_golgi={node.n_mossy_fibers_golgi}")

    pre_steps = round(args.pre_ms / args.dt_ms)
    total_steps = round((args.pre_ms + args.drive_ms) / args.dt_ms)

    # --- Experiment 3 add-on: an optional climbing-fiber pulse to a
    # selected Purkinje-cell group (DESIGN.md) -- OFF by default
    # (args.climbing_pulse_na == 0.0), which keeps this function's behavior
    # for Experiment 1 itself completely unchanged (a single
    # inject_climbing_fiber_input(0.0) call, exactly as before this
    # add-on existed). "For the first two experiments, keep climbing fibers
    # off": this add-on lives in the same script rather than a separate one,
    # since the user's own spec describes it as Experiment 1/2 PLUS a
    # climbing-fiber pulse, not an independent protocol.
    climbing_pulse_active = args.climbing_pulse_na != 0.0
    if climbing_pulse_active:
        climbing_pulse_start_ms = args.climbing_pulse_start_ms
        if climbing_pulse_start_ms is None:
            climbing_pulse_start_ms = args.pre_ms + args.drive_ms / 2.0
        climbing_pulse_end_ms = args.climbing_pulse_end_ms
        if climbing_pulse_end_ms is None:
            climbing_pulse_end_ms = climbing_pulse_start_ms + 50.0
        climbing_pulse_on_step = round(climbing_pulse_start_ms / args.dt_ms)
        climbing_pulse_off_step = round(climbing_pulse_end_ms / args.dt_ms)

        pulse_seed = args.climbing_pulse_seed if args.climbing_pulse_seed is not None else args.seed + 1_000
        rng = np.random.default_rng(pulse_seed)
        n_selected = max(1, round(args.climbing_pulse_fraction * args.n_cells))
        selected_purkinje = rng.choice(args.n_cells, size=n_selected, replace=False)
        climbing_pattern = np.zeros(args.n_cells, dtype=np.float64)
        climbing_pattern[selected_purkinje] = args.climbing_pulse_na
        print(f"climbing-fiber pulse: {args.climbing_pulse_na} nA on {n_selected}/{args.n_cells} "
              f"Purkinje cells, [{climbing_pulse_start_ms:.0f}, {climbing_pulse_end_ms:.0f}) ms")
    node.inject_climbing_fiber_input(CLIMBING_STRENGTH_NA)

    # Same drive seeds/arrays across BOTH GJ conditions (only g_gap_nS
    # differs) -- the correctness-critical part of "everything else
    # identical" for this comparison. Distinct from GridNodeBatch's own
    # internal connectivity_seed + 1_000_003 offset (DESIGN.md), no
    # collision.
    spikes_granule = generate_poisson_spike_train(
        node.n_mossy_fibers_granule, total_steps, args.dt_ms, args.rate_hz,
        seed=args.seed * 2 + 1, on_step=pre_steps, off_step=total_steps,
    )
    spikes_golgi = generate_poisson_spike_train(
        node.n_mossy_fibers_golgi, total_steps, args.dt_ms, args.rate_hz,
        seed=args.seed * 2 + 2, on_step=pre_steps, off_step=total_steps,
    )
    filt_granule = ExponentialCurrentFilter(node.n_mossy_fibers_granule, args.mf_amplitude_na, args.mf_tau_ms)
    filt_golgi = ExponentialCurrentFilter(node.n_mossy_fibers_golgi, args.mf_amplitude_na, args.mf_tau_ms)

    record_every = args.record_every
    n_frames = total_steps // record_every
    V_granule = np.empty((n_frames, args.n_cells), dtype=np.float16)
    V_purkinje = np.empty((n_frames, args.n_cells), dtype=np.float16)
    V_golgi = np.empty((n_frames, n_golgi), dtype=np.float16)
    # Stellate is outside this experiment's own analysis scope (DESIGN.md), so it's only recorded when --save-raw-voltage will actually use
    # it -- every other grid point in the overnight/exp3 grids skips the
    # extra per-step copy and memory.
    V_stellate = np.empty((n_frames, args.n_cells), dtype=np.float16) if args.save_raw_voltage else None

    t0 = time.perf_counter()
    frame = 0
    for step in range(total_steps):
        if climbing_pulse_active:
            if step == climbing_pulse_on_step:
                node.inject_climbing_fiber_input(climbing_pattern)
            elif step == climbing_pulse_off_step:
                node.inject_climbing_fiber_input(CLIMBING_STRENGTH_NA)
        I_granule = filt_granule.step(args.dt_ms, spikes_granule[step])
        I_golgi = filt_golgi.step(args.dt_ms, spikes_golgi[step])
        node.inject_mossy_fiber_input(I_granule, I_golgi)
        node.step(args.dt_ms)
        if (step + 1) % record_every == 0:
            V_granule[frame] = node.cells.granule.get_voltage()
            V_purkinje[frame] = node.cells.purkinje.get_voltage()
            V_golgi[frame] = node.golgi.get_voltage()
            if V_stellate is not None:
                V_stellate[frame] = node.cells.stellate.get_voltage()
            frame += 1
    elapsed = time.perf_counter() - t0
    print(f"recorded run: {elapsed:.1f}s for {total_steps} steps ({total_steps / elapsed:.1f} steps/s)")

    t_ms = (np.arange(1, n_frames + 1) * record_every) * args.dt_ms
    V_granule_f32 = np.asarray(V_granule, dtype=np.float32)
    V_purkinje_f32 = np.asarray(V_purkinje, dtype=np.float32)
    V_golgi_f32 = np.asarray(V_golgi, dtype=np.float32)
    V_stellate_f32 = np.asarray(V_stellate, dtype=np.float32) if V_stellate is not None else None

    golgi_spike_times = population_spike_times_ms(t_ms, V_golgi_f32, SPIKE_THRESHOLD_MV)
    granule_spike_times = population_spike_times_ms(t_ms, V_granule_f32, SPIKE_THRESHOLD_MV)
    purkinje_spike_times = population_spike_times_ms(t_ms, V_purkinje_f32, SPIKE_THRESHOLD_MV)

    total_ms = args.pre_ms + args.drive_ms
    golgi_isi_cv, golgi_n_isis = pooled_isi_cv(V_golgi_f32, args.dt_ms * record_every, SPIKE_THRESHOLD_MV)
    n_purkinje_spikes = sum(len(st) for st in purkinje_spike_times)
    purkinje_rate_hz = n_purkinje_spikes / max(1, args.n_cells) / (args.drive_ms * 1e-3)

    # Clamp the onset buffer to at most half the drive window, so a short
    # --drive-ms (e.g. a fast smoke test) never pushes drive_start_ms past
    # drive_end_ms and produces a negative bin count.
    onset_buffer_ms = min(ONSET_BUFFER_MS, args.drive_ms / 2.0)
    drive_start_ms = args.pre_ms + onset_buffer_ms
    drive_end_ms = total_ms
    golgi_binned = binned_spike_counts(golgi_spike_times, drive_start_ms, drive_end_ms, BIN_MS)
    golgi_corr = pairwise_correlation(golgi_binned)
    golgi_mean_pairwise_corr = mean_offdiag_correlation(golgi_corr)
    freqs_hz, power = population_power_spectrum(golgi_spike_times, drive_start_ms, drive_end_ms, BIN_MS)
    dominant_freq_hz = float(freqs_hz[np.argmax(power)]) if len(power) else float("nan")

    psth_bins_golgi, psth_rate_golgi = psth(golgi_spike_times, 0.0, total_ms, BIN_MS)
    psth_bins_granule, psth_rate_granule = psth(granule_spike_times, 0.0, total_ms, BIN_MS)

    print(f"golgi_isi_cv={golgi_isi_cv:.3f} (n_isis={golgi_n_isis}), "
          f"purkinje_rate_hz={purkinje_rate_hz:.2f}, dominant_golgi_freq_hz={dominant_freq_hz:.2f}")

    # --- Figure: 2x3, raster row + PSTH/spectrum/correlation row ---
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"gj_{gj_label}_rate{args.rate_hz:g}hz_seed{args.seed}"
    if climbing_pulse_active:
        tag += f"_cf{args.climbing_pulse_na:g}na"

    if args.save_figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Experiment 1 -- GJ {gj_label} (g_gap_nS={g_gap_nS}), rate={args.rate_hz} Hz, "
                     f"n_cells={args.n_cells}, seed={args.seed}")

        axes[0, 0].eventplot(golgi_spike_times, color="black", linewidths=0.8)
        axes[0, 0].set_title(f"Golgi raster (n={n_golgi})")
        axes[0, 0].set_xlabel("time (ms)")
        axes[0, 0].axvline(args.pre_ms, color="red", linestyle="--", linewidth=0.8)

        n_display = min(RASTER_DISPLAY_CELLS, args.n_cells)
        axes[0, 1].eventplot(granule_spike_times[:n_display], color="black", linewidths=0.5)
        axes[0, 1].set_title(f"Granule raster (showing {n_display}/{args.n_cells})")
        axes[0, 1].set_xlabel("time (ms)")
        axes[0, 1].axvline(args.pre_ms, color="red", linestyle="--", linewidth=0.8)

        axes[0, 2].eventplot(purkinje_spike_times[:n_display], color="black", linewidths=0.5)
        axes[0, 2].set_title(f"Purkinje raster (showing {n_display}/{args.n_cells})")
        axes[0, 2].set_xlabel("time (ms)")
        axes[0, 2].axvline(args.pre_ms, color="red", linestyle="--", linewidth=0.8)

        axes[1, 0].plot(psth_bins_golgi, psth_rate_golgi, label="Golgi")
        axes[1, 0].plot(psth_bins_granule, psth_rate_granule, label="Granule", alpha=0.7)
        axes[1, 0].set_title("Pooled PSTH")
        axes[1, 0].set_xlabel("time (ms)")
        axes[1, 0].set_ylabel("pop. rate (Hz)")
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].semilogy(freqs_hz, power + 1e-12)
        axes[1, 1].set_title("Golgi power spectrum (drive window)")
        axes[1, 1].set_xlabel("frequency (Hz)")

        im = axes[1, 2].imshow(golgi_corr, vmin=-1, vmax=1, cmap="RdBu_r")
        axes[1, 2].set_title("Golgi pairwise correlation")
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

        fig.text(
            0.01, 0.01,
            f"Purkinje firing rate: {purkinje_rate_hz:.2f} Hz  |  Golgi ISI CV: {golgi_isi_cv:.3f}",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))

        fig_path = output_dir / f"{tag}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"saved figure: {fig_path}")

    if args.save_raw_voltage:
        _save_raw_voltage(
            output_dir / f"{tag}_activity", node, t_ms, V_granule_f32, V_purkinje_f32, V_golgi_f32,
            V_stellate_f32, args.dt_ms, record_every,
            drive_onset_ms=args.pre_ms,
            climbing_pulse_onset_ms=(climbing_pulse_start_ms if climbing_pulse_active else None),
        )

    summary = {
        "gj_label": gj_label,
        "g_gap_nS": g_gap_nS,
        "rate_hz": args.rate_hz,
        "seed": args.seed,
        "n_cells": args.n_cells,
        "n_golgi": n_golgi,
        "n_mossy_fibers_granule": node.n_mossy_fibers_granule,
        "n_mossy_fibers_golgi": node.n_mossy_fibers_golgi,
        "golgi_granule_divergence": args.golgi_granule_divergence,
        "mossy_to_golgi_divergence": args.mossy_to_golgi_divergence,
        "heterogeneity_seed": args.heterogeneity_seed,
        "distance_decay_per_um": args.distance_decay_per_um,
        "golgi_isi_cv": golgi_isi_cv,
        "golgi_n_isis": golgi_n_isis,
        "purkinje_rate_hz": purkinje_rate_hz,
        "dominant_golgi_freq_hz": dominant_freq_hz,
        "golgi_mean_pairwise_corr": golgi_mean_pairwise_corr,
        "mf_amplitude_na": args.mf_amplitude_na,
        "mf_tau_ms": args.mf_tau_ms,
        "climbing_pulse_na": args.climbing_pulse_na,
        "climbing_pulse_active": climbing_pulse_active,
        "elapsed_s": elapsed,
    }
    with open(output_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        output_dir / f"{tag}_derived.npz",
        psth_bins_golgi=psth_bins_golgi, psth_rate_golgi=psth_rate_golgi,
        psth_bins_granule=psth_bins_granule, psth_rate_granule=psth_rate_granule,
        freqs_hz=freqs_hz, power=power, golgi_corr=golgi_corr,
        golgi_spike_times=np.array(golgi_spike_times, dtype=object),
        granule_spike_times=np.array(granule_spike_times, dtype=object),
        purkinje_spike_times=np.array(purkinje_spike_times, dtype=object),
    )

    print(f"saved: {output_dir / f'{tag}_summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-cells", type=int, default=1024)
    parser.add_argument("--width-um", type=float, default=None,
                         help="Explicit tissue width (x, short axis) in um -- requires --height-um "
                              "too. Overrides the default n_cells-derived square grid (_square_grid) "
                              "with a rectangle; set --n-cells to match the resulting grid node count "
                              "(width_um/10 * height_um/10) so golgi_ratio's density assumption holds.")
    parser.add_argument("--height-um", type=float, default=None,
                         help="Explicit tissue height (y, long axis) in um -- see --width-um.")
    parser.add_argument("--wall-clock-budget-s", type=float, default=WALL_CLOCK_BUDGET_S_PER_CONDITION,
                         help="Abort before a condition projected to exceed this many seconds "
                              f"(default {WALL_CLOCK_BUDGET_S_PER_CONDITION:.0f}s, sized for the "
                              "discovery-scale grid) -- raise explicitly for a deliberate larger run.")
    parser.add_argument("--rate-hz", type=float, default=20.0,
                         help="Mean per-fiber spike rate during the drive window (eventual "
                              "sweep: 5/10/20/40/70 Hz -- this run uses one fixed rate).")
    parser.add_argument("--gj", choices=("off", "on", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--golgi-ratio", type=float, default=GOLGI_RATIO)
    parser.add_argument("--golgi-granule-divergence", type=int, default=100,
                         help="Discovery-scale override (default GridCouplingParams is 2000, "
                              "which exceeds n_cells and clamps to the full population at this "
                              "scale -- see pilot_probe_connectivity.py / DESIGN.md).")
    parser.add_argument("--mossy-to-golgi-divergence", type=int, default=8,
                         help="Discovery-scale override (default GridCouplingParams is 25, "
                              "which exceeds n_golgi at this scale and clamps the mossy-fiber-> "
                              "Golgi pool to the full population -- see DESIGN.md).")
    parser.add_argument("--heterogeneity-seed", type=int, default=None,
                         help="Sou11-style per-cell parameter/position heterogeneity (DESIGN.md). None (default) = every cell identical, prior behavior.")
    parser.add_argument("--distance-decay-per-um", type=float, default=None,
                         help="Sou11-style exponential distance-scaled conductance (DESIGN.md), decay parameter in 1/um -- Sou11's own paper value is 0.01. "
                              "None (default) = no distance scaling, prior behavior.")
    parser.add_argument("--dt-ms", type=float, default=DT_MS)
    parser.add_argument("--pre-ms", type=float, default=1000.0)
    parser.add_argument("--drive-ms", type=float, default=1000.0)
    parser.add_argument("--record-every", type=int, default=5)  # -> 0.05 ms resolution
    parser.add_argument("--mf-amplitude-na", type=float, default=0.05)
    parser.add_argument("--mf-tau-ms", type=float, default=3.0)
    parser.add_argument("--climbing-pulse-na", type=float, default=0.0,
                         help="Experiment 3 add-on (DESIGN.md): climbing-fiber pulse strength "
                              "[nA] onto a selected Purkinje-cell group. 0.0 (default) = off, "
                              "identical to Experiment 1's own protocol.")
    parser.add_argument("--climbing-pulse-start-ms", type=float, default=None,
                         help="Default: mid-drive-window (pre_ms + drive_ms/2).")
    parser.add_argument("--climbing-pulse-end-ms", type=float, default=None,
                         help="Default: --climbing-pulse-start-ms + 50ms.")
    parser.add_argument("--climbing-pulse-fraction", type=float, default=0.1,
                         help="Fraction of Purkinje cells in the selected group.")
    parser.add_argument("--climbing-pulse-seed", type=int, default=None,
                         help="Default: --seed + 1000.")
    parser.add_argument("--save-figure", action="store_true",
                         help="Save the detailed 6-panel PNG for each grid point (DESIGN.md). "
                              "Off by default -- use scripts/summarize_grid_results.py for a "
                              "whole-grid overview instead; reach for this only when you already "
                              "know you want one specific condition's detail.")
    parser.add_argument("--save-raw-voltage", action="store_true",
                         help="Save raw voltage in scripts/view_activity.py's native format "
                              "(DESIGN.md). Off by default -- ~246MB per condition, so "
                              "deliberate, not automatic for a full grid.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("--- Drive-amplitude/throughput pre-flight ---")
    steps_per_sec, reached_threshold, max_v_granule, max_v_golgi = _dry_run_check(args)
    print(f"dry run: {steps_per_sec:.1f} steps/s, "
          f"max V granule={max_v_granule:.1f} mV, max V golgi={max_v_golgi:.1f} mV")
    if not reached_threshold:
        print(f"WARNING: neither granule nor Golgi crossed {SPIKE_THRESHOLD_MV} mV during the "
              f"{DRY_RUN_MS:.0f} ms dry run at rate={args.rate_hz} Hz, amplitude={args.mf_amplitude_na} nA, "
              f"tau={args.mf_tau_ms} ms. This could mean the drive is genuinely too weak at this rate, "
              f"or the placeholder amplitude/tau need retuning (--mf-amplitude-na/--mf-tau-ms) -- "
              f"proceeding anyway, but inspect the output critically.")

    total_steps = round((args.pre_ms + args.drive_ms) / args.dt_ms)
    projected_s = total_steps / steps_per_sec
    print(f"projected time per condition: {projected_s:.1f}s")
    if projected_s > args.wall_clock_budget_s:
        print(f"ABORTING: projected {projected_s:.1f}s/condition exceeds the "
              f"{args.wall_clock_budget_s:.0f}s budget -- reduce --n-cells, --pre-ms, --drive-ms, "
              "or raise --wall-clock-budget-s for a deliberate larger run.")
        return

    conditions = {"off": [("off", 0.0)], "on": [("on", 1.0)], "both": [("off", 0.0), ("on", 1.0)]}[args.gj]
    summaries = [run_condition(args, label, g_gap, output_dir) for label, g_gap in conditions]

    if len(summaries) == 2:
        off_s, on_s = summaries
        print("\n=== GJ off vs on comparison ===")
        print(f"Golgi ISI CV:        off={off_s['golgi_isi_cv']:.3f}  on={on_s['golgi_isi_cv']:.3f}")
        print(f"Purkinje rate (Hz):  off={off_s['purkinje_rate_hz']:.2f}  on={on_s['purkinje_rate_hz']:.2f}")
        print(f"Dominant Golgi freq (Hz): off={off_s['dominant_golgi_freq_hz']:.2f}  "
              f"on={on_s['dominant_golgi_freq_hz']:.2f}")

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
