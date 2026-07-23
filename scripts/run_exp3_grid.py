"""
Experiment 3 grid runner: the climbing-fiber-pulse add-on (DESIGN.md)
layered onto EXACTLY Experiment 1's rate/seed grid (same RATES_HZ, same
SEEDS, same --gj both per invocation), so results are directly comparable
to the already-collected Experiment 1 dataset (outputs/overnight/exp1/) --
the only difference is climbing_pulse_na enabled. Per the user's own
protocol, Experiment 3 runs "afterward", once Experiment 1/2's results are
in -- both completed (outputs/overnight/, 6.07h, 24/24 grid points).

Reuses scripts/run_overnight.py's RATES_HZ/N_CELLS constants and its
_log/_run_one helpers directly rather than duplicating them -- this is a
thin variant of that grid, not an independent design.

Uses the SAME discovery-scale connectivity fix as the overnight run
(golgi_granule_divergence=100, mossy_to_golgi_divergence=25 -- DESIGN.md,
selected via scripts/pilot_probe_connectivity.py's 3-seed deep pilot).
CLIMBING_PULSE_NA=3500.0 matches this session's validated single real-scale
Experiment 3 run (which correctly drove Purkinje from 0.00 Hz to 1.89 Hz)
and this repo's existing CLIMBING_INPUT_NA convention.

Usage:
    python scripts/run_exp3_grid.py
    python scripts/run_exp3_grid.py --heterogeneity-seed 11 --distance-decay-per-um 0.01
    nohup python scripts/run_exp3_grid.py > outputs/exp3_grid/nohup.log 2>&1 &
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from scripts.run_overnight import N_CELLS, RATES_HZ, _log, _run_one

SEEDS = [0, 1, 2, 3]
CLIMBING_PULSE_NA = 3500.0
GOLGI_GRANULE_DIVERGENCE = 100
MOSSY_TO_GOLGI_DIVERGENCE = 25

OUTPUT_ROOT = Path(__file__).parent.parent / "outputs" / "exp3_grid"
LOG_FILE_NAME = "exp3_grid_progress.log"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--heterogeneity-seed", type=int, default=None,
                         help="Sou11-style per-cell heterogeneity (DESIGN.md). None (default) "
                              "= off, prior behavior.")
    parser.add_argument("--distance-decay-per-um", type=float, default=None,
                         help="Sou11-style distance-scaled conductance (DESIGN.md), 1/um. "
                              "None (default) = off.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / LOG_FILE_NAME

    run_start = time.perf_counter()
    grid = [(rate, seed) for rate in RATES_HZ for seed in SEEDS]
    _log(log_path, f"=== Experiment 3 grid starting: {len(grid)} grid points "
                    f"({len(RATES_HZ)} rates x {len(SEEDS)} seeds), "
                    f"climbing_pulse_na={CLIMBING_PULSE_NA}, "
                    f"heterogeneity_seed={args.heterogeneity_seed}, "
                    f"distance_decay_per_um={args.distance_decay_per_um} ===")

    completed = 0
    failed = 0
    times: list[float] = []
    for rate, seed in grid:
        cmd = [
            sys.executable, "scripts/exp1_synchronization.py",
            "--n-cells", str(N_CELLS), "--rate-hz", str(rate), "--seed", str(seed),
            "--gj", "both", "--output-dir", str(output_root),
            "--golgi-granule-divergence", str(GOLGI_GRANULE_DIVERGENCE),
            "--mossy-to-golgi-divergence", str(MOSSY_TO_GOLGI_DIVERGENCE),
            "--climbing-pulse-na", str(CLIMBING_PULSE_NA),
        ]
        if args.heterogeneity_seed is not None:
            cmd += ["--heterogeneity-seed", str(args.heterogeneity_seed)]
        if args.distance_decay_per_um is not None:
            cmd += ["--distance-decay-per-um", str(args.distance_decay_per_um)]
        ok, elapsed = _run_one(cmd, log_path)
        times.append(elapsed)
        if ok:
            completed += 1
        else:
            failed += 1

    total_elapsed = time.perf_counter() - run_start
    _log(log_path, "=== Experiment 3 grid finished ===")
    _log(log_path, f"{completed} completed, {failed} failed (of {len(grid)})")
    _log(log_path, f"Total wall-clock: {total_elapsed / 3600.0:.2f}h")
    _log(log_path, f"Output: {output_root}")


if __name__ == "__main__":
    main()
