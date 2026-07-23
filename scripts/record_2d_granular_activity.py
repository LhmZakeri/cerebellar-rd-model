"""
Record a granular-layer GridNodeBatch run to disk, with full independent
control over the shape and strength of mossy-fiber (-> granule, + Golgi)
and climbing-fiber (-> Purkinje) drive.

With no --mossy-*/--climbing-* flags, defaults reproduce the classic demo:
a patch of PATCH_FRACTION of cells (by index) driven at each end of the
population -- see scripts/_granular_demo.py. Pass --mossy-shape/
--climbing-shape to override either pathway independently, so any
combination is possible in one run (e.g. a sine-wave mossy-fiber drive
together with a random climbing-fiber subset, or one pathway disabled
entirely):

    none      no stimulation
    uniform   every cell/fiber gets --*-strength
    patch     --*-strength over index range [--*-start-frac, --*-end-frac),
              zero elsewhere (the default shape for both pathways)
    gradient  linear ramp 0 -> --*-strength over the same index range
    sine      --*-strength * sin(2*pi*index / --*-period-cells)
    random    --*-strength on a random --*-fraction of cells/fibers (--*-seed)
    radius    --*-strength on cells/fibers within --*-radius-um of
              (--*-center-x-um, --*-center-y-um) (default: grid center),
              zero outside -- the only shape defined by *physical* position
              rather than index (see note below)
    file      literal per-target current [nA], loaded from a --*-file .npy;
              escape hatch for any pattern the presets above don't cover;
              overrides that pathway's other --*-* options. For --climbing-*
              shape (n_cells,); for --mossy-shape (n_mossy_fibers_granule,)
              -- see the mossy-fiber note below, since a single array can't
              also drive the differently-sized Golgi-facing fiber pool, so
              this shape leaves that pool on its uniform-mean-drive fallback.

Granule/Purkinje/stellate carry no grid position of their own (DESIGN.md) -- GridNodeBatch always places them via a single independent random
draw (sample_uniform_positions), regardless of whether n_cells happens to
match the grid's node count, so cell *index* never corresponds to physical
location. none/uniform/patch/gradient/sine/random/file above are all
defined over index, so e.g. "patch" is a contiguous block of cell indices,
not a spatial patch of tissue -- driven cells land scattered across the
whole domain. Use "radius" for an actual spatially-localized stimulus.

--mossy-* shapes are defined over the mossy-fiber population, NOT over
granule cells directly (DESIGN.md: mossy fibers are not a 1:1 wire to
either granule or Golgi cells -- ~4 fiber contacts/granule cell, ~39
granule cells/fiber; ~100 fiber contacts/Golgi cell, ~25 Golgi cells/fiber).
There are two independent fiber pools (granule-facing, Golgi-facing,
different sizes/positions) -- every --mossy-shape is built once per pool
(node.mossy_granule_x/y and node.mossy_golgi_x/y, node.n_mossy_fibers_
granule/golgi) and the resulting per-fiber currents are converged onto
granule/Golgi cells by GridNodeBatch.inject_mossy_fiber_input() itself, via
each cell's own real anatomical contact list -- this script never touches
granule/Golgi cells' currents directly for the mossy pathway.

Each pathway's drive is also independently confinable in *time* via
--{mossy,climbing}-start-ms/--{mossy,climbing}-end-ms (default: 0 and
--duration-ms, i.e. on for the whole run, matching prior behavior). The
current switches on at *-start-ms and back to 0 at *-end-ms -- e.g. a
climbing-fiber pulse from 50-60 ms during an otherwise-500 ms run, rather
than driven for the whole recording.

Recording and viewing are otherwise fully decoupled (activity_recording.py
/ activity_viewer.py): pass --view to open the viewer here once recording
finishes, or leave a recording running and point scripts/view_activity.py
at the same --output-dir from another process to watch it live.

Usage:
    # Classic demo: fixed patch at each end of the population, then exit
    python scripts/record_2d_granular_activity.py

    # Same, but open the viewer once it finishes
    python scripts/record_2d_granular_activity.py --view

    # Uniform mossy-fiber drive only
    python scripts/record_2d_granular_activity.py --mossy-shape uniform --mossy-strength 0.05

    # Both pathways, independent shapes
    python scripts/record_2d_granular_activity.py \\
        --mossy-shape sine --mossy-strength 0.08 --mossy-period-cells 500 \\
        --climbing-shape random --climbing-strength 3500 --climbing-fraction 0.05 --climbing-seed 1

    # Fully custom per-cell array, any shape not covered above
    python scripts/record_2d_granular_activity.py --mossy-shape file --mossy-file my_pattern.npy

    # Spatially-localized mossy-fiber patch: cells within 100 um of grid
    # center, on only 20-40 ms -- for testing spatial spread
    python scripts/record_2d_granular_activity.py \\
        --mossy-shape radius --mossy-radius-um 100 --mossy-strength 0.08 \\
        --mossy-start-ms 20 --mossy-end-ms 40 --view

    # Climbing-fiber pulse only from 50-60 ms of a 500 ms run (mossy fiber
    # still on for the whole run, its default)
    python scripts/record_2d_granular_activity.py \\
        --duration-ms 500 --climbing-start-ms 50 --climbing-end-ms 60 --view

    # Compare with/without the Golgi<->granule + Golgi<->Golgi coupling
    # (DESIGN.md) -- run into two different --output-dir, then view
    # each with scripts/view_activity.py to see the difference directly
    # (including Golgi's own recorded voltage, the golgi panel):
    python scripts/record_2d_granular_activity.py --output-dir outputs/coupled
    python scripts/record_2d_granular_activity.py --output-dir outputs/uncoupled --disable-golgi-coupling

    e.g. (Not all 30 flags can be in one call meaningfully):

        python scripts/record_2d_granular_activity.py \
        --width-um 2000 --height-um 20000 --resolution-um 10 \
        --n-cells 50000 \
        --golgi-seed 3 --connectivity-seed 4 --golgi-ratio 0.005 \
        --disable-golgi-coupling \
        --duration-ms 300 --dt-ms 0.01 --record-every 10 \
        --output-dir outputs/full_param_run \
        --mossy-shape gradient --mossy-strength 0.08 --mossy-start-frac 0.1 --mossy-end-frac 0.4 \
        --climbing-shape random --climbing-strength 3000 --climbing-fraction 0.2 --climbing-seed 7 \
        --view
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts._granular_demo import (
    CLIMBING_INPUT_NA,
    DT_MS,
    DURATION_MS,
    HEIGHT_UM,
    MOSSY_INPUT_NA,
    OUTPUT_DIR,
    PATCH_FRACTION,
    RECORD_EVERY,
    RESOLUTION_UM,
    WIDTH_UM,
)
from src.simulation.activity_recording import record_grid_activity
from src.simulation.coupling_params import GridCouplingParams
from src.simulation.geometry import FlatGrid
from src.simulation.grid_node_batch import GridNodeBatch

_SHAPES = ("none", "uniform", "patch", "gradient", "sine", "random", "radius", "file")

# Defaults reproduce the classic demo pattern: mossy fiber drives the first
# PATCH_FRACTION of cells, climbing fiber drives the last PATCH_FRACTION.
_SHAPE_DEFAULTS = {
    "mossy": {"strength": MOSSY_INPUT_NA, "start_frac": 0.0, "end_frac": PATCH_FRACTION},
    "climbing": {"strength": CLIMBING_INPUT_NA, "start_frac": 1.0 - PATCH_FRACTION, "end_frac": 1.0},
}


def _add_stimulation_args(parser: argparse.ArgumentParser, name: str) -> None:
    """Registers one pathway's (mossy/climbing) full set of --{name}-* shape
    options, so both pathways get identical, independent controls."""
    defaults = _SHAPE_DEFAULTS[name]
    parser.add_argument(f"--{name}-shape", choices=_SHAPES, default="patch",
                         help=f"{name}-fiber drive shape (default: patch, matching the classic demo).")
    parser.add_argument(f"--{name}-strength", type=float, default=defaults["strength"],
                         help="Peak current [nA] (uniform/patch/gradient/sine/random).")
    parser.add_argument(f"--{name}-start-frac", type=float, default=defaults["start_frac"],
                         help="patch/gradient: start of driven index range, as a fraction [0-1] of n_cells.")
    parser.add_argument(f"--{name}-end-frac", type=float, default=defaults["end_frac"],
                         help="patch/gradient: end of driven index range, as a fraction [0-1] of n_cells.")
    parser.add_argument(f"--{name}-period-cells", type=float, default=100.0,
                         help="sine: period, in number of cells.")
    parser.add_argument(f"--{name}-fraction", type=float, default=0.15,
                         help="random: fraction of cells driven.")
    parser.add_argument(f"--{name}-seed", type=int, default=0,
                         help="random: RNG seed.")
    parser.add_argument(f"--{name}-center-x-um", type=float, default=None,
                         help="radius: stimulation center x [um] (default: --width-um / 2).")
    parser.add_argument(f"--{name}-center-y-um", type=float, default=None,
                         help="radius: stimulation center y [um] (default: --height-um / 2).")
    parser.add_argument(f"--{name}-radius-um", type=float, default=100.0,
                         help="radius: --*-strength on cells within this radius [um] of the "
                              "center, zero outside.")
    parser.add_argument(f"--{name}-file", type=str, default=None,
                         help="file: path to a .npy array of shape (n_cells,), literal per-cell "
                              "current [nA] -- overrides this pathway's other shape options.")
    parser.add_argument(f"--{name}-start-ms", type=float, default=None,
                         help="Time [ms] this pathway's drive switches on (default: 0, i.e. "
                              "start of run).")
    parser.add_argument(f"--{name}-end-ms", type=float, default=None,
                         help="Time [ms] this pathway's drive switches back off (default: "
                              "--duration-ms, i.e. stays on for the whole run).")


def _build_pattern(
    args: argparse.Namespace, name: str, n: int,
    pos_x: np.ndarray | None = None, pos_y: np.ndarray | None = None,
) -> np.ndarray | float:
    """Builds the --{name}-* pattern into either a scalar (none/uniform) or
    a per-target array (a "target" being a granule cell for --climbing-*, or
    a mossy fiber for --mossy-* -- DESIGN.md made mossy fiber its own
    population rather than driving granule/Golgi cells directly), as
    inject_mossy_fiber_input/inject_climbing_fiber_input already accept
    either (grid_node_batch.py). pos_x/pos_y are only needed for
    shape == "radius"."""
    shape = getattr(args, f"{name}_shape")
    strength = getattr(args, f"{name}_strength")

    if shape == "radius":
        center_x = getattr(args, f"{name}_center_x_um")
        center_x = args.width_um / 2.0 if center_x is None else center_x
        center_y = getattr(args, f"{name}_center_y_um")
        center_y = args.height_um / 2.0 if center_y is None else center_y
        radius = getattr(args, f"{name}_radius_um")
        pattern = np.zeros(n, dtype=np.float64)
        dist = np.hypot(pos_x - center_x, pos_y - center_y)
        pattern[dist <= radius] = strength
        return pattern

    if shape == "file":
        path = getattr(args, f"{name}_file")
        if path is None:
            raise ValueError(f"--{name}-shape file requires --{name}-file <path.npy>")
        pattern = np.load(path)
        if pattern.shape != (n,):
            raise ValueError(
                f"--{name}-file array has shape {pattern.shape}, expected ({n},)"
            )
        return pattern.astype(np.float64)

    if shape == "none":
        return 0.0
    if shape == "uniform":
        return strength

    start_frac = getattr(args, f"{name}_start_frac")
    end_frac = getattr(args, f"{name}_end_frac")
    i0 = round(start_frac * n)
    i1 = round(end_frac * n)

    if shape == "patch":
        pattern = np.zeros(n, dtype=np.float64)
        pattern[i0:i1] = strength
        return pattern

    if shape == "gradient":
        pattern = np.zeros(n, dtype=np.float64)
        pattern[i0:i1] = np.linspace(0.0, strength, max(1, i1 - i0))
        return pattern

    if shape == "sine":
        period = getattr(args, f"{name}_period_cells")
        idx = np.arange(n, dtype=np.float64)
        return strength * np.sin(2.0 * np.pi * idx / period)

    # shape == "random"
    fraction = getattr(args, f"{name}_fraction")
    seed = getattr(args, f"{name}_seed")
    rng = np.random.default_rng(seed)
    pattern = np.zeros(n, dtype=np.float64)
    n_on = max(1, round(fraction * n))
    chosen = rng.choice(n, size=n_on, replace=False)
    pattern[chosen] = strength
    return pattern


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duration-ms", type=float, default=DURATION_MS)
    parser.add_argument("--dt-ms", type=float, default=DT_MS)
    parser.add_argument("--record-every", type=int, default=RECORD_EVERY)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--width-um", type=float, default=WIDTH_UM)
    parser.add_argument("--height-um", type=float, default=HEIGHT_UM)
    parser.add_argument("--resolution-um", type=float, default=RESOLUTION_UM)
    parser.add_argument(
        "--n-cells", type=int, default=None,
        help="Granule/Purkinje/stellate count, independent of grid size "
             "(DESIGN.md). Default: match grid size.",
    )
    parser.add_argument("--golgi-seed", type=int, default=0)
    parser.add_argument("--connectivity-seed", type=int, default=0)
    parser.add_argument("--golgi-ratio", type=float, default=1.0 / 430.0)
    parser.add_argument(
        "--disable-golgi-coupling", action="store_true",
        help="Zero out Golgi<->granule synapses and Golgi<->Golgi diffusion (DESIGN.md), "
             "for a before/after comparison against a default (coupled) recording.",
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Open ActivityViewer on the recording immediately after it finishes.",
    )
    _add_stimulation_args(parser, "mossy")
    _add_stimulation_args(parser, "climbing")
    args = parser.parse_args()

    geometry = FlatGrid(
        width_um=args.width_um, height_um=args.height_um, resolution_um=args.resolution_um
    )
    n_cells = args.n_cells if args.n_cells is not None else geometry.build().n_nodes

    coupling = (
        GridCouplingParams(g_gap_nS=0.0, gmax_golgi_to_granule=0.0)
        if args.disable_golgi_coupling
        else None
    )
    node = GridNodeBatch(
        geometry,
        golgi_seed=args.golgi_seed,
        n_cells=n_cells,
        connectivity_seed=args.connectivity_seed,
        golgi_ratio=args.golgi_ratio,
        coupling=coupling,
    )

    output_dir = Path(args.output_dir)
    record_grid_activity(
        node, args.duration_ms, args.dt_ms, args.record_every, output_dir,
        # --mossy-* shapes are now defined over the mossy-fiber population,
        # not over granule cells directly (DESIGN.md) -- built once
        # against each of the two independent fiber pools (granule-facing,
        # Golgi-facing), since --mossy-shape radius/patch/etc. only take one
        # (n, x, y) target set at a time and the two pools differ in size.
        # "file" is the exception: one literal array can't match both pools'
        # sizes at once, so it drives the granule pool only and leaves
        # mossy_golgi_pattern None, falling back to inject_mossy_fiber_
        # input's own mean-collapse-to-uniform-Golgi-drive default.
        mossy_pattern=_build_pattern(
            args, "mossy", node.n_mossy_fibers_granule, node.mossy_granule_x, node.mossy_granule_y
        ),
        mossy_golgi_pattern=(
            None if args.mossy_shape == "file"
            else _build_pattern(
                args, "mossy", node.n_mossy_fibers_golgi, node.mossy_golgi_x, node.mossy_golgi_y
            )
        ),
        mossy_start_ms=args.mossy_start_ms,
        mossy_end_ms=args.mossy_end_ms,
        climbing_pattern=_build_pattern(args, "climbing", n_cells, node.node_x, node.node_y),
        climbing_start_ms=args.climbing_start_ms,
        climbing_end_ms=args.climbing_end_ms,
    )

    if args.view:
        from src.simulation.activity_viewer import ActivityViewer
        ActivityViewer(output_dir).show()


if __name__ == "__main__":
    main()
