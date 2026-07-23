"""
View-only half of issue #7a.1 -- opens ActivityViewer (or, with --spatial,
SpatialActivityViewer -- DESIGN.md) on an existing output_dir. No
simulation code at all: this can point at a recording made by
scripts/record_2d_granular_activity.py, either while it's still running
in another process (pass --view to that script instead if you just want to
view your own run once it finishes), or after it's finished.

--spatial requires positions.npz (recordings made before DESIGN.md
don't have one -- re-record to use it). --3d (only meaningful with
--spatial) stacks the three layers in one 3D axes (granular layer z=0,
Purkinje z=1, molecular z=2) instead of three separate 2D panels.

The wavefront boundary (--spatial's granular-layer panel) can legitimately
show nothing in some frames if --record-every is coarse relative to a
Golgi spike's rise time -- a cell's previous recorded voltage may already
be above --golgi-active-voltage-mv by the time a large-enough jump lands in
the same recorded step. Confirmed working at the current defaults (-20mV
active-voltage cutoff, 20mV delta) against a real 0.2ms/frame recording:
10/499 frames had a qualifying hull (DESIGN.md). Try a smaller
--record-every, or loosen the two thresholds further, if you want it to
fire more often -- an empty hull in a given frame isn't necessarily a bug.

Usage:
    python scripts/view_activity.py
    python scripts/view_activity.py outputs/granular_activity
    python scripts/view_activity.py outputs/granular_activity --spatial
    python scripts/view_activity.py outputs/granular_activity --spatial --3d
    python scripts/view_activity.py outputs/granular_activity --spatial --golgi-wavefront-delta-threshold-mv 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts._granular_demo import OUTPUT_DIR
from src.simulation.activity_viewer import ActivityViewer
from src.simulation.spatial_activity_viewer import SpatialActivityViewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=str, nargs="?", default=str(OUTPUT_DIR),
                         help="Recording directory to view (default: %(default)s).")
    parser.add_argument("--refresh-ms", type=int, default=200,
                         help="How often to re-check the recording's progress marker.")
    parser.add_argument(
        "--spatial", action="store_true",
        help="Open the per-cell spatial scatter view instead of the per-cell-index line graphs.",
    )
    parser.add_argument(
        "--3d", action="store_true", dest="use_3d",
        help="--spatial only: stack all three layers in one 3D axes instead of three 2D panels.",
    )
    parser.add_argument(
        "--golgi-active-voltage-mv", type=float, default=-20.0,
        help="--spatial only: a Golgi cell must have been below this voltage last frame to "
             "qualify as newly recruited into the wavefront boundary (default -20).",
    )
    parser.add_argument(
        "--golgi-wavefront-delta-threshold-mv", type=float, default=20.0,
        help="--spatial only: minimum |deltaV| this frame for a quiescent Golgi cell to count "
             "as newly recruited into the wavefront boundary curve (default 20).",
    )
    args = parser.parse_args()

    if args.spatial:
        viewer = SpatialActivityViewer(
            Path(args.output_dir),
            refresh_ms=args.refresh_ms,
            golgi_active_voltage_mv=args.golgi_active_voltage_mv,
            golgi_wavefront_delta_threshold_mv=args.golgi_wavefront_delta_threshold_mv,
            use_3d=args.use_3d,
        )
    else:
        viewer = ActivityViewer(Path(args.output_dir), refresh_ms=args.refresh_ms)
    viewer.show()


if __name__ == "__main__":
    main()
