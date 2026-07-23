"""
Overnight grid runner for Experiments 1 and 2 (Scientific Goals A/B) --
the full multi-rate/multi-g_gap x multi-seed sweeps both experiment
scripts' own real-scale validation runs (this session) established were
too expensive to run interactively (~250-400s per single condition at
n_cells=1024). Not attempted in this session at full grid size; intended
for an unattended 8-10 hour run.

Grid (sized against THIS session's measured throughput at n_cells=1024,
dt=0.01ms, ~800 steps/s):
  Experiment 1 (scripts/exp1_synchronization.py): RATES_HZ x SEEDS, each
    invocation runs --gj both internally (both GJ conditions in one call).
    Measured: ~496s/invocation (2 x ~248s for the 2000ms off+on protocol).
  Experiment 2 (scripts/exp2_chaos_test.py): SEEDS, each invocation runs
    --g-gap-values "0,0.5,1,2" internally (all 4 g_gap values, each a twin
    A/B pair, in one call). Measured: ~733-772s per g_gap value pair of
    twin sims -> ~3100s/invocation for all 4.

At 4 seeds each: Exp1 ~5*4*496s ~= 9920s (2.8h), Exp2 ~4*3100s = 12400s
(3.4h), combined ~6.2h -- comfortably inside an 8-10h window with margin
for measurement variance (this machine's throughput could differ from a
quiet overnight run's, in either direction). Experiment 3 (climbing-fiber
add-on) is deliberately NOT included in this grid -- it's described in the
user's own protocol as an exploration to run "afterward", once Experiments
1/2's results are in, not a third parallel discovery scan competing for the
same budget.

Exp1 runs to completion first, then Exp2 -- not interleaved -- so a run
that gets cut short (machine restart, wall-clock overrun) leaves at least
one experiment's grid FULLY complete rather than both half-done.

Each grid point is a subprocess call, wrapped in try/except so one failure
(crash, unexpected exception) doesn't abort the rest of the night -- logged
and skipped, not silently swallowed. A running per-experiment average
actual-time-per-run refines the remaining-budget projection as the night
progresses; if the projected remaining time for an experiment's remaining
grid points would exceed the budget, the rest of that experiment's grid is
skipped (logged, not run partially/truncated) rather than risking an
external kill mid-run.

REQUIRES --golgi-granule-divergence/--mossy-to-golgi-divergence (DESIGN.md): run scripts/pilot_probe_connectivity.py first and pass its winning
combination. No default is provided on purpose -- silently falling back to
GridCouplingParams' own defaults (2000/25) would make g_gap_nS have
provably zero effect on Golgi dynamics at this network scale (both exceed
the actual population sizes, degenerating random-subset connectivity into
full-population connectivity -- see DESIGN.md for the full diagnosis),
wasting the entire night on a foregone-conclusion result.

Usage:
    python scripts/run_overnight.py
    python scripts/run_overnight.py --budget-hours 9 --seeds 0,1,2,3
    nohup python scripts/run_overnight.py > outputs/overnight/run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RATES_HZ = [5.0, 10.0, 20.0, 40.0, 70.0]
G_GAP_VALUES = "0,0.5,1,2"
N_CELLS = 1024

# Initial estimates from this session's real-scale validation runs --
# refined by a running average of actual measured times as the night
# progresses, so later budget checks get more accurate as data accumulates.
ESTIMATED_EXP1_RUN_S = 500.0
ESTIMATED_EXP2_RUN_S = 3100.0

OUTPUT_ROOT = Path(__file__).parent.parent / "outputs" / "overnight"
LOG_FILE_NAME = "overnight_progress.log"


def _log(log_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def _run_one(cmd: list[str], log_path: Path) -> tuple[bool, float]:
    """Runs one grid point as a subprocess. Returns (succeeded, elapsed_s).
    Never raises -- a failed grid point is logged and the night continues."""
    _log(log_path, f"START: {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            _log(log_path, f"FAILED (exit {result.returncode}, {elapsed:.0f}s): {' '.join(cmd)}")
            _log(log_path, f"  stderr tail: {result.stderr[-2000:]}")
            return False, elapsed
        _log(log_path, f"OK ({elapsed:.0f}s)")
        return True, elapsed
    except Exception as e:  # noqa: BLE001 -- deliberately broad, see module docstring
        elapsed = time.perf_counter() - t0
        _log(log_path, f"EXCEPTION ({elapsed:.0f}s): {' '.join(cmd)} -> {e!r}")
        return False, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3",
                         help="Comma-separated seed list, used for BOTH Experiment 1 and 2's grids.")
    parser.add_argument("--rates-hz", type=str, default=",".join(str(r) for r in RATES_HZ),
                         help="Comma-separated Experiment 1 rate grid -- override for a quick "
                              "correctness smoke test of this driver script itself.")
    parser.add_argument("--budget-hours", type=float, default=9.0,
                         help="Stop starting new grid points once the projected remaining time for "
                              "an experiment's remaining grid would exceed this budget.")
    parser.add_argument("--n-cells", type=int, default=N_CELLS)
    parser.add_argument("--golgi-granule-divergence", type=int, default=None,
                         help="Discovery-scale connectivity override, required -- winning value "
                              "from scripts/pilot_probe_connectivity.py (DESIGN.md). No "
                              "default: an unset value would silently run the whole grid on the "
                              "known-broken GridCouplingParams default (2000).")
    parser.add_argument("--mossy-to-golgi-divergence", type=int, default=None,
                         help="Discovery-scale connectivity override, required -- see "
                              "--golgi-granule-divergence above.")
    parser.add_argument("--heterogeneity-seed", type=int, default=None,
                         help="Sou11-style per-cell heterogeneity (DESIGN.md). None (default) "
                              "= off, prior behavior.")
    parser.add_argument("--distance-decay-per-um", type=float, default=None,
                         help="Sou11-style distance-scaled conductance (DESIGN.md), 1/um. "
                              "None (default) = off.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    if args.golgi_granule_divergence is None or args.mossy_to_golgi_divergence is None:
        parser.error(
            "--golgi-granule-divergence and --mossy-to-golgi-divergence are both required -- "
            "run scripts/pilot_probe_connectivity.py first and pass its winning combination. "
            "Running with the GridCouplingParams defaults would silently make g_gap_nS have zero "
            "effect on Golgi dynamics at this network scale (DESIGN.md)."
        )

    seeds = [int(s) for s in args.seeds.split(",")]
    rates_hz = [float(r) for r in args.rates_hz.split(",")]
    budget_s = args.budget_hours * 3600.0
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / LOG_FILE_NAME

    run_start = time.perf_counter()
    _log(log_path, f"=== Overnight run starting: seeds={seeds}, budget={args.budget_hours}h, "
                    f"n_cells={args.n_cells} ===")

    # --- Experiment 1: RATES_HZ x seeds, --gj both per invocation ---
    exp1_out = output_root / "exp1"
    exp1_grid = [(rate, seed) for rate in rates_hz for seed in seeds]
    exp1_times: list[float] = []
    exp1_completed = 0
    exp1_skipped = 0

    _log(log_path, f"--- Experiment 1: {len(exp1_grid)} grid points "
                    f"({len(rates_hz)} rates x {len(seeds)} seeds) ---")
    for rate, seed in exp1_grid:
        avg_s = (sum(exp1_times) / len(exp1_times)) if exp1_times else ESTIMATED_EXP1_RUN_S
        elapsed_so_far = time.perf_counter() - run_start
        if elapsed_so_far + avg_s > budget_s:
            _log(log_path, f"BUDGET: skipping remaining Experiment 1 grid points "
                            f"({len(exp1_grid) - exp1_completed - exp1_skipped} left) -- "
                            f"projected next run would exceed the {args.budget_hours}h budget.")
            exp1_skipped = len(exp1_grid) - exp1_completed
            break
        cmd = [
            sys.executable, "scripts/exp1_synchronization.py",
            "--n-cells", str(args.n_cells), "--rate-hz", str(rate), "--seed", str(seed),
            "--gj", "both", "--output-dir", str(exp1_out),
            "--golgi-granule-divergence", str(args.golgi_granule_divergence),
            "--mossy-to-golgi-divergence", str(args.mossy_to_golgi_divergence),
        ]
        if args.heterogeneity_seed is not None:
            cmd += ["--heterogeneity-seed", str(args.heterogeneity_seed)]
        if args.distance_decay_per_um is not None:
            cmd += ["--distance-decay-per-um", str(args.distance_decay_per_um)]
        ok, elapsed = _run_one(cmd, log_path)
        exp1_times.append(elapsed)
        if ok:
            exp1_completed += 1
        else:
            exp1_skipped += 1  # counted as skipped-from-success, not retried

    # --- Experiment 2: seeds, --g-gap-values "0,0.5,1,2" per invocation ---
    exp2_out = output_root / "exp2"
    exp2_times: list[float] = []
    exp2_completed = 0
    exp2_skipped = 0

    _log(log_path, f"--- Experiment 2: {len(seeds)} grid points (seeds), "
                    f"each sweeping g_gap_nS in [{G_GAP_VALUES}] ---")
    for seed in seeds:
        avg_s = (sum(exp2_times) / len(exp2_times)) if exp2_times else ESTIMATED_EXP2_RUN_S
        elapsed_so_far = time.perf_counter() - run_start
        if elapsed_so_far + avg_s > budget_s:
            _log(log_path, f"BUDGET: skipping remaining Experiment 2 grid points "
                            f"({len(seeds) - exp2_completed - exp2_skipped} left) -- "
                            f"projected next run would exceed the {args.budget_hours}h budget.")
            exp2_skipped = len(seeds) - exp2_completed
            break
        cmd = [
            sys.executable, "scripts/exp2_chaos_test.py",
            "--n-cells", str(args.n_cells), "--g-gap-values", G_GAP_VALUES, "--seed", str(seed),
            "--output-dir", str(exp2_out),
            "--golgi-granule-divergence", str(args.golgi_granule_divergence),
            "--mossy-to-golgi-divergence", str(args.mossy_to_golgi_divergence),
        ]
        if args.heterogeneity_seed is not None:
            cmd += ["--heterogeneity-seed", str(args.heterogeneity_seed)]
        if args.distance_decay_per_um is not None:
            cmd += ["--distance-decay-per-um", str(args.distance_decay_per_um)]
        ok, elapsed = _run_one(cmd, log_path)
        exp2_times.append(elapsed)
        if ok:
            exp2_completed += 1
        else:
            exp2_skipped += 1

    total_elapsed = time.perf_counter() - run_start
    _log(log_path, "=== Overnight run finished ===")
    _log(log_path, f"Experiment 1: {exp1_completed} completed, {exp1_skipped} skipped/failed "
                    f"(of {len(exp1_grid)})")
    _log(log_path, f"Experiment 2: {exp2_completed} completed, {exp2_skipped} skipped/failed "
                    f"(of {len(seeds)})")
    _log(log_path, f"Total wall-clock: {total_elapsed / 3600.0:.2f}h")
    _log(log_path, f"Output: {output_root}")


if __name__ == "__main__":
    main()
