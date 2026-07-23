"""scripts/record_2d_granular_activity.py's --{name}-shape radius: the only
stimulation shape defined by physical position (node_x/node_y) rather than
cell index -- unlike patch/gradient/sine/random, which select cells by
index and, since GridNodeBatch always places granule/Purkinje/stellate via
an independent random draw (DESIGN.md), have no relationship to where
those cells actually sit in the tissue."""
import argparse

import numpy as np

from scripts.record_2d_granular_activity import _build_pattern

_N_CELLS = 8


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        width_um=100.0, height_um=100.0,
        mossy_shape="radius", mossy_strength=0.08,
        mossy_center_x_um=None, mossy_center_y_um=None, mossy_radius_um=20.0,
        mossy_file=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_radius_selects_only_cells_within_radius_of_default_center():
    # Grid center defaults to (width/2, height/2) = (50, 50).
    node_x = np.array([50.0, 50.0, 60.0, 90.0, 10.0, 50.0, 55.0, 0.0])
    node_y = np.array([50.0, 60.0, 50.0, 90.0, 10.0, 50.0, 55.0, 0.0])
    args = _args()

    pattern = _build_pattern(args, "mossy", _N_CELLS, node_x, node_y)

    dist = np.hypot(node_x - 50.0, node_y - 50.0)
    expected_on = dist <= 20.0
    assert np.array_equal(pattern > 0, expected_on)
    assert np.all(pattern[expected_on] == 0.08)
    assert np.all(pattern[~expected_on] == 0.0)


def test_radius_center_is_overridable():
    node_x = np.array([0.0, 0.0, 100.0, 100.0])
    node_y = np.array([0.0, 100.0, 0.0, 100.0])
    args = _args(mossy_center_x_um=0.0, mossy_center_y_um=0.0, mossy_radius_um=5.0)

    pattern = _build_pattern(args, "mossy", 4, node_x, node_y)

    np.testing.assert_array_equal(pattern, [0.08, 0.0, 0.0, 0.0])


def test_patch_shape_selects_by_index_not_position():
    """Documents the exact problem "radius" was added to solve: "patch"
    picks a contiguous index range regardless of where those cells actually
    are, since node_x/node_y (independent random positions, DESIGN.md)
    aren't consulted at all."""
    args = _args(
        mossy_shape="patch", mossy_start_frac=0.0, mossy_end_frac=0.5,
    )
    node_x = np.array([90.0, 5.0, 50.0, 10.0, 80.0, 20.0, 60.0, 0.0])
    node_y = np.array([90.0, 5.0, 50.0, 10.0, 80.0, 20.0, 60.0, 0.0])

    pattern = _build_pattern(args, "mossy", _N_CELLS, node_x, node_y)

    # First half of the *index* range is driven, scattered across position.
    np.testing.assert_array_equal(pattern[:4], [0.08, 0.08, 0.08, 0.08])
    np.testing.assert_array_equal(pattern[4:], [0.0, 0.0, 0.0, 0.0])
