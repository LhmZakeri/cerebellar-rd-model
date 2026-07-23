"""scripts/_granular_demo.py: build_demo_node's n_cells defaults to grid size
(back-compat) but can be chosen independently (DESIGN.md), and the
index-based mossy-/climbing-fiber patch patterns work for any n_cells."""
from scripts._granular_demo import (
    _climbing_fiber_pattern,
    _mossy_fiber_pattern,
    build_demo_node,
)

_WIDTH_UM = 30.0
_HEIGHT_UM = 20.0
_RESOLUTION_UM = 10.0
_GRID_N_NODES = 6  # 3 cols x 2 rows


def test_default_n_cells_matches_grid_size():
    node = build_demo_node(_WIDTH_UM, _HEIGHT_UM, _RESOLUTION_UM)
    assert node.cells.n_nodes == _GRID_N_NODES


def test_explicit_n_cells_independent_of_grid_size():
    node = build_demo_node(_WIDTH_UM, _HEIGHT_UM, _RESOLUTION_UM, n_cells=50)
    assert node.cells.n_nodes == 50
    assert node.positions.n_nodes == _GRID_N_NODES


def test_patch_patterns_are_length_n_cells_and_non_uniform():
    for n_cells in (_GRID_N_NODES, 50):
        mossy = _mossy_fiber_pattern(n_cells)
        climbing = _climbing_fiber_pattern(n_cells)
        assert len(mossy) == n_cells
        assert len(climbing) == n_cells
        assert mossy.min() == 0.0 and mossy.max() > 0.0
        assert climbing.min() == 0.0 and climbing.max() > 0.0
