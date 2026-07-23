"""
Static cell-type placement map -- shows where Golgi cells landed relative
to the dense granule/Purkinje/stellate layers (issue #7b). Unlike
record_2d_granular_activity.py / view_activity.py, this needs no recording
step at all: placement is fixed at GridNodeBatch construction, so this reads
straight off a freshly built node (via build_demo_node's fixed default
stimulation -- irrelevant here, since this script never steps the node).
Re-run with a different --golgi-seed to see how placement changes.

output_dir is unrelated to any recording (this script never reads one) --
if given, it's purely where to save the resulting cell_types.png instead of
showing it interactively.

Usage:
    python scripts/view_cell_types.py
    python scripts/view_cell_types.py --golgi-seed 7
    python scripts/view_cell_types.py --golgi-ratio 0.01
    python scripts/view_cell_types.py outputs/coupled/   # save PNG there instead of showing
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts._granular_demo import HEIGHT_UM, RESOLUTION_UM, WIDTH_UM, build_demo_node
from src.simulation.cell_type_viewer import plot_cell_types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir", type=str, nargs="?", default=None,
        help="If given, save cell_types.png here instead of showing interactively.",
    )
    parser.add_argument("--golgi-seed", type=int, default=0)
    parser.add_argument("--golgi-ratio", type=float, default=1.0 / 430.0)
    parser.add_argument("--width-um", type=float, default=WIDTH_UM)
    parser.add_argument("--height-um", type=float, default=HEIGHT_UM)
    parser.add_argument("--resolution-um", type=float, default=RESOLUTION_UM)
    parser.add_argument(
        "--n-cells", type=int, default=None,
        help="Granule/Purkinje/stellate count, independent of grid size "
             "(DESIGN.md). Default: match grid size (today's behavior).",
    )
    args = parser.parse_args()

    node = build_demo_node(
        args.width_um,
        args.height_um,
        args.resolution_um,
        n_cells=args.n_cells,
        golgi_seed=args.golgi_seed,
        golgi_ratio=args.golgi_ratio,
    )

    save_path = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "cell_types.png"

    plot_cell_types(node, save_path=save_path)

    if save_path is not None:
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
