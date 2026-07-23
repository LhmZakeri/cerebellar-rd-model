"""
Experiment 2 of 3 (deterministic chaos test, Scientific Goal B): twin
simulations with IDENTICAL deterministic mossy-fiber input, differing only
by a tiny perturbation to one Golgi cell's initial voltage, swept over
g_gap_nS -- tests sensitive dependence on initial conditions (DESIGN.md).
NOT mixed with Experiment 1's random Poisson-driven synchronization protocol
or Experiment 3's climbing-fiber drive (both explicitly kept separate, per
the user's "do not mix synchronization and chaos" instruction).

Protocol (fixed structure, deterministic -- no RNG in the drive signal):
  0-500 ms      tonic baseline mossy-fiber drive (uniform, both fiber pools).
  500-510 ms    brief, spatially localized pulse superimposed on the
                baseline (a subset of fibers near a chosen center, selected
                by radius -- NOT a global pulse to every fiber, since a
                global pulse would trivially synchronize the whole network
                rather than testing whether a spatially real perturbation
                propagates chaotically).
  510-3000 ms   back to the same tonic baseline drive, continuing
                deterministically to the end of the run.
  Climbing fiber OFF throughout (Experiment 3, deferred, adds it back).

Twin simulations, both driven by the EXACT SAME precomputed deterministic
pattern (same object, not independently regenerated) and built from
IDENTICAL seeds (golgi_seed=connectivity_seed=--seed for both):
  Simulation A: unperturbed initial conditions.
  Simulation B: node.golgi.V[--perturb-golgi-index] += --perturbation-mv
                immediately after construction, before any stepping.

Swept over --g-gap-values (default "0,0.5,1,2", matching the user's spec).
For each value: D(t) = ||V_A(t)-V_B(t)|| (src/simulation/divergence_metrics.py)
computed separately for granule/Purkinje/Golgi; log(D(t)) slope over
--fit-start-ms/--fit-end-ms (a real judgment call, not auto-detected); a
broadband-spectrum check on simulation A's Golgi population during the
post-pulse window (population_power_spectrum, DESIGN.md) as the second
half of the "positive divergence + broadband spectrum" chaos-candidate
signature the user's protocol asks for. Always saved as a JSON summary; the
D(t)/log-D/spectrum PNG is opt-in (--save-figure, DESIGN.md) -- use
scripts/summarize_grid_results.py for a whole-sweep overview instead.

Usage:
    # Fast correctness smoke test (seconds, not minutes)
    python scripts/exp2_chaos_test.py --n-cells 128 --total-ms 60 --g-gap-values 0,1

    # Real validation run, one g_gap value
    python scripts/exp2_chaos_test.py --n-cells 1024 --g-gap-values 1.0 --seed 0

    # Full sweep (long -- intended for an overnight run, not interactive use)
    python scripts/exp2_chaos_test.py --n-cells 1024 --g-gap-values 0,0.5,1,2 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.simulation.coupling_params import GridCouplingParams
from src.simulation.divergence_metrics import log_divergence_growth_rate, trajectory_distance
from src.simulation.geometry import FlatGrid
from src.simulation.grid_node_batch import GridNodeBatch
from src.simulation.spike_metrics import population_power_spectrum, population_spike_times_ms

RESOLUTION_UM = 10.0
GOLGI_RATIO = 1.0 / 36.0  # Discovery-scale golgi_ratio (CONTEXT.md) -- see
# exp1_synchronization.py's docstring for the full rationale (Sou11's
# reduced-network methodology, computationally enlarged not biological).
DT_MS = 0.01
CLIMBING_STRENGTH_NA = 0.0  # OFF throughout -- Experiment 3 (deferred) adds this back.
SPIKE_THRESHOLD_MV = -20.0

BASELINE_MS = 500.0
PULSE_MS = 10.0
TOTAL_MS = 3000.0
MOSSY_BASELINE_NA = 0.05  # matches MOSSY_STRENGTH_NA used throughout this repo.
MOSSY_PULSE_NA = 0.15  # UNCALIBRATED placeholder, 3x baseline -- flagged, not
# validated against firing-rate output; --mossy-pulse-na overrides.
PULSE_RADIUS_UM = 60.0  # UNCALIBRATED placeholder for the localized-pulse
# footprint -- overridable via --pulse-radius-um.

BIN_MS = 5.0
DEFAULT_FIT_START_MS = BASELINE_MS + PULSE_MS + 50.0  # a bit after the pulse ends
DEFAULT_FIT_END_MS = BASELINE_MS + PULSE_MS + 500.0  # a real judgment call --
# inspect the saved log(D(t)) curve and override via --fit-start-ms/--fit-end-ms
# if this window doesn't land before saturation for a given g_gap value.

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "exp2_chaos_test"


def _square_grid(n_cells: int) -> FlatGrid:
    """Duplicated from discovery_base_run.py/exp1_synchronization.py
    (DESIGN.md) -- no shared extraction yet, no third-plus caller
    justifying it beyond this repo's existing no-premature-abstraction
    convention."""
    n_side = max(2, round(n_cells**0.5))
    side_um = n_side * RESOLUTION_UM
    return FlatGrid(width_um=side_um, height_um=side_um, resolution_um=RESOLUTION_UM)


def _build_node(args: argparse.Namespace, g_gap_nS: float) -> GridNodeBatch:
    geometry = _square_grid(args.n_cells)
    return GridNodeBatch(
        geometry,
        golgi_seed=args.seed,
        n_cells=args.n_cells,
        connectivity_seed=args.seed,
        golgi_ratio=args.golgi_ratio,
        heterogeneity_seed=args.heterogeneity_seed,
        coupling=GridCouplingParams(
            g_gap_nS=g_gap_nS,
            golgi_granule_divergence=args.golgi_granule_divergence,
            mossy_to_golgi_divergence=args.mossy_to_golgi_divergence,
            distance_decay_per_um=args.distance_decay_per_um,
        ),
    )


def _build_deterministic_patterns(
    node: GridNodeBatch, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed baseline + pulse current arrays for both mossy-fiber pools --
    computed ONCE (not per step, not per simulation) so simulations A and B
    are driven by the literal same arrays, not independently regenerated
    ones. The pulse subset is chosen by radius around the domain center over
    each pool's own positions (localized, not global -- see module
    docstring) via a fixed seed (not connectivity/golgi_seed -- this is a
    presentation/pulse-shape choice, orthogonal to network construction).
    """
    width_um = node.positions.n_cols * node.positions.resolution_um
    height_um = node.positions.n_rows * node.positions.resolution_um
    center_x, center_y = width_um / 2.0, height_um / 2.0

    baseline_granule = np.full(node.n_mossy_fibers_granule, args.mossy_baseline_na, dtype=np.float64)
    baseline_golgi = np.full(node.n_mossy_fibers_golgi, args.mossy_baseline_na, dtype=np.float64)

    dist_granule = np.hypot(node.mossy_granule_x - center_x, node.mossy_granule_y - center_y)
    dist_golgi = np.hypot(node.mossy_golgi_x - center_x, node.mossy_golgi_y - center_y)
    pulse_granule = baseline_granule.copy()
    pulse_granule[dist_granule <= args.pulse_radius_um] = args.mossy_pulse_na
    pulse_golgi = baseline_golgi.copy()
    pulse_golgi[dist_golgi <= args.pulse_radius_um] = args.mossy_pulse_na

    return baseline_granule, baseline_golgi, pulse_granule, pulse_golgi


def _run_twin(
    args: argparse.Namespace, g_gap_nS: float, perturb: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs one simulation (A if perturb=False, B if perturb=True) through
    the full deterministic protocol, recording granule/Purkinje/Golgi
    voltage. Returns (t_ms, V_granule, V_purkinje, V_golgi)."""
    node = _build_node(args, g_gap_nS)
    node.inject_climbing_fiber_input(CLIMBING_STRENGTH_NA)
    if perturb:
        node.golgi.V[args.perturb_golgi_index] += args.perturbation_mv

    baseline_g, baseline_go, pulse_g, pulse_go = _build_deterministic_patterns(node, args)

    total_steps = round(args.total_ms / args.dt_ms)
    pulse_start_step = round(args.baseline_ms / args.dt_ms)
    pulse_end_step = round((args.baseline_ms + args.pulse_ms) / args.dt_ms)

    record_every = args.record_every
    n_frames = total_steps // record_every
    n_golgi = node.golgi.n_golgi
    # float32, NOT float16 (unlike exp1_synchronization.py/discovery_base_run.py):
    # this experiment measures a tiny (1e-6 to 1e-3 mV) perturbation's growth.
    # float16's precision near a typical -70 to +40 mV operating range is
    # ~0.03-0.06 mV (its mantissa is only 10 bits) -- coarser than the whole
    # perturbation range, so it would silently round D(t) to exactly zero
    # until the trajectories had already diverged by orders of magnitude more
    # than the phenomenon being measured (confirmed empirically: the first
    # real-scale validation run recorded in float16 and got D(t)=0 throughout
    # the entire fit window, DESIGN.md). float32's ~8e-6 mV precision at
    # this magnitude resolves the smallest requested perturbation (1e-6 mV)
    # with headroom.
    V_granule = np.empty((n_frames, args.n_cells), dtype=np.float32)
    V_purkinje = np.empty((n_frames, args.n_cells), dtype=np.float32)
    V_golgi = np.empty((n_frames, n_golgi), dtype=np.float32)

    frame = 0
    for step in range(total_steps):
        if pulse_start_step <= step < pulse_end_step:
            node.inject_mossy_fiber_input(pulse_g, pulse_go)
        else:
            node.inject_mossy_fiber_input(baseline_g, baseline_go)
        node.step(args.dt_ms)
        if (step + 1) % record_every == 0:
            V_granule[frame] = node.cells.granule.get_voltage()
            V_purkinje[frame] = node.cells.purkinje.get_voltage()
            V_golgi[frame] = node.golgi.get_voltage()
            frame += 1

    t_ms = (np.arange(1, n_frames + 1) * record_every) * args.dt_ms
    return t_ms, V_granule, V_purkinje, V_golgi


def run_g_gap_value(args: argparse.Namespace, g_gap_nS: float, output_dir: Path) -> dict:
    print(f"\n=== g_gap_nS={g_gap_nS} ===")
    t0 = time.perf_counter()
    t_ms, V_granule_A, V_purkinje_A, V_golgi_A = _run_twin(args, g_gap_nS, perturb=False)
    _, V_granule_B, V_purkinje_B, V_golgi_B = _run_twin(args, g_gap_nS, perturb=True)
    elapsed = time.perf_counter() - t0
    print(f"twin simulations: {elapsed:.1f}s wall-clock")

    D_granule = trajectory_distance(np.asarray(V_granule_A, np.float32), np.asarray(V_granule_B, np.float32))
    D_purkinje = trajectory_distance(np.asarray(V_purkinje_A, np.float32), np.asarray(V_purkinje_B, np.float32))
    D_golgi = trajectory_distance(np.asarray(V_golgi_A, np.float32), np.asarray(V_golgi_B, np.float32))

    fit_start_ms = args.fit_start_ms
    fit_end_ms = min(args.fit_end_ms, args.total_ms)
    slope_granule, _ = log_divergence_growth_rate(D_granule, t_ms, fit_start_ms, fit_end_ms)
    slope_purkinje, _ = log_divergence_growth_rate(D_purkinje, t_ms, fit_start_ms, fit_end_ms)
    slope_golgi, _ = log_divergence_growth_rate(D_golgi, t_ms, fit_start_ms, fit_end_ms)

    # Broadband-spectrum check on simulation A's Golgi population, post-pulse.
    golgi_spike_times_A = population_spike_times_ms(
        t_ms, np.asarray(V_golgi_A, np.float32), SPIKE_THRESHOLD_MV
    )
    spectrum_start_ms = args.baseline_ms + args.pulse_ms + 50.0
    freqs_hz, power = population_power_spectrum(
        golgi_spike_times_A, spectrum_start_ms, args.total_ms, BIN_MS
    )
    # crude broadband indicator: ratio of the single largest bin's power to
    # the median -- low ratio ~= broadband/flat, high ratio ~= a dominant
    # narrow peak (periodic, not broadband).
    peak_to_median = float(np.max(power) / np.median(power)) if len(power) and np.median(power) > 0 else float("nan")

    print(f"log-D growth rate (per ms): granule={slope_granule:.4g}  purkinje={slope_purkinje:.4g}  "
          f"golgi={slope_golgi:.4g}")
    print(f"Golgi (sim A) spectrum peak/median ratio: {peak_to_median:.2f} "
          f"(lower = more broadband)")

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ggap{g_gap_nS:g}_seed{args.seed}"

    if args.save_figure:
        # --- Figure: D(t), log(D(t)) with fit, spectrum ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f"Experiment 2 -- g_gap_nS={g_gap_nS}, perturbation={args.perturbation_mv} mV "
                     f"on golgi[{args.perturb_golgi_index}], n_cells={args.n_cells}, seed={args.seed}")

        axes[0].plot(t_ms, D_granule, label="granule", alpha=0.8)
        axes[0].plot(t_ms, D_purkinje, label="purkinje", alpha=0.8)
        axes[0].plot(t_ms, D_golgi, label="golgi", alpha=0.8)
        axes[0].axvline(args.baseline_ms, color="red", linestyle="--", linewidth=0.8)
        axes[0].set_title("D(t) = ||V_A - V_B||")
        axes[0].set_xlabel("time (ms)")
        axes[0].legend(fontsize=8)

        for D, label in ((D_granule, "granule"), (D_purkinje, "purkinje"), (D_golgi, "golgi")):
            with np.errstate(divide="ignore"):
                log_D = np.where(D > 0, np.log(D), np.nan)
            axes[1].plot(t_ms, log_D, label=label, alpha=0.8)
        axes[1].axvspan(fit_start_ms, fit_end_ms, color="gray", alpha=0.2, label="fit window")
        axes[1].set_title("log D(t)")
        axes[1].set_xlabel("time (ms)")
        axes[1].legend(fontsize=8)

        axes[2].semilogy(freqs_hz, power + 1e-12)
        axes[2].set_title(f"Golgi (sim A) spectrum, post-pulse\npeak/median={peak_to_median:.2f}")
        axes[2].set_xlabel("frequency (Hz)")

        fig.tight_layout()
        fig_path = output_dir / f"{tag}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"saved figure: {fig_path}")

    summary = {
        "g_gap_nS": g_gap_nS,
        "seed": args.seed,
        "n_cells": args.n_cells,
        "golgi_granule_divergence": args.golgi_granule_divergence,
        "mossy_to_golgi_divergence": args.mossy_to_golgi_divergence,
        "heterogeneity_seed": args.heterogeneity_seed,
        "distance_decay_per_um": args.distance_decay_per_um,
        "perturbation_mv": args.perturbation_mv,
        "perturb_golgi_index": args.perturb_golgi_index,
        "fit_start_ms": fit_start_ms,
        "fit_end_ms": fit_end_ms,
        "log_D_slope_per_ms_granule": slope_granule,
        "log_D_slope_per_ms_purkinje": slope_purkinje,
        "log_D_slope_per_ms_golgi": slope_golgi,
        "golgi_spectrum_peak_to_median": peak_to_median,
        "elapsed_s": elapsed,
    }
    with open(output_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {output_dir / f'{tag}_summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-cells", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--golgi-ratio", type=float, default=GOLGI_RATIO)
    parser.add_argument("--golgi-granule-divergence", type=int, default=100,
                         help="Discovery-scale override -- see DESIGN.md.")
    parser.add_argument("--mossy-to-golgi-divergence", type=int, default=8,
                         help="Discovery-scale override -- see DESIGN.md.")
    parser.add_argument("--heterogeneity-seed", type=int, default=None,
                         help="Sou11-style per-cell heterogeneity (DESIGN.md). Same seed used "
                              "for BOTH twin simulations (A and B) -- only the deliberate IC "
                              "perturbation should differ between them, not the heterogeneity draw.")
    parser.add_argument("--distance-decay-per-um", type=float, default=None,
                         help="Sou11-style distance-scaled conductance (DESIGN.md), 1/um. "
                              "None (default) = off.")
    parser.add_argument("--dt-ms", type=float, default=DT_MS)
    parser.add_argument("--baseline-ms", type=float, default=BASELINE_MS)
    parser.add_argument("--pulse-ms", type=float, default=PULSE_MS)
    parser.add_argument("--total-ms", type=float, default=TOTAL_MS)
    parser.add_argument("--mossy-baseline-na", type=float, default=MOSSY_BASELINE_NA)
    parser.add_argument("--mossy-pulse-na", type=float, default=MOSSY_PULSE_NA)
    parser.add_argument("--pulse-radius-um", type=float, default=PULSE_RADIUS_UM)
    parser.add_argument("--perturbation-mv", type=float, default=1e-4,
                         help="Initial Golgi-voltage perturbation [mV] in simulation B (spec range: 1e-6-1e-3).")
    parser.add_argument("--perturb-golgi-index", type=int, default=0)
    parser.add_argument("--g-gap-values", type=str, default="0,0.5,1,2",
                         help="Comma-separated g_gap_nS sweep values.")
    parser.add_argument("--fit-start-ms", type=float, default=DEFAULT_FIT_START_MS)
    parser.add_argument("--fit-end-ms", type=float, default=DEFAULT_FIT_END_MS)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--save-figure", action="store_true",
                         help="Save the D(t)/log-D/spectrum PNG for each g_gap value (DESIGN.md). Off by default -- use scripts/summarize_grid_results.py "
                              "for a whole-sweep overview instead.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    g_gap_values = [float(v) for v in args.g_gap_values.split(",")]
    output_dir = Path(args.output_dir)

    summaries = [run_g_gap_value(args, g_gap, output_dir) for g_gap in g_gap_values]

    print("\n=== g_gap_nS sweep summary (log-D growth rate, Golgi) ===")
    for s in summaries:
        print(f"  g_gap_nS={s['g_gap_nS']:>4}: golgi slope={s['log_D_slope_per_ms_golgi']:.4g}/ms, "
              f"spectrum peak/median={s['golgi_spectrum_peak_to_median']:.2f}")

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
