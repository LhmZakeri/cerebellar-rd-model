"""FlatGrid position generation, 4-connectivity neighbour lists (issue #7a),
and Poisson-disk Golgi placement (issue #7b).

Uses a small, hand-checkable 3-row x 4-col grid throughout the grid/neighbour
tests so indices can be verified by inspection rather than by construction-
mirrors-implementation logic. Golgi placement tests use a larger synthetic
grid instead, since "uniform, not patchy" is a statistical property that
needs enough placed cells to check meaningfully.
"""
import numpy as np

from src.simulation.geometry import (
    NO_NEIGHBOUR,
    FlatGrid,
    GridPositions,
    build_convergent_neighbours,
    build_golgi_diffusion_neighbours,
    build_golgi_granule_neighbours,
    build_grid_neighbours,
    place_golgi_cells,
    sample_uniform_positions,
)

_RESOLUTION_UM = 10.0
_N_COLS = 4  # x axis (width)
_N_ROWS = 3  # y axis (height)


def _small_grid() -> GridPositions:
    grid = FlatGrid(
        width_um=_N_COLS * _RESOLUTION_UM,
        height_um=_N_ROWS * _RESOLUTION_UM,
        resolution_um=_RESOLUTION_UM,
    )
    return grid.build()


def _nearest_neighbour_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Brute-force nearest-neighbour distance for each point -- fine at the
    scale (a few hundred points) these placement tests use."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


class TestFlatGridBuild:

    def test_rejects_nonpositive_dimensions(self):
        for kwargs in [
            dict(width_um=0, height_um=10, resolution_um=10),
            dict(width_um=10, height_um=-5, resolution_um=10),
            dict(width_um=10, height_um=10, resolution_um=0),
        ]:
            try:
                FlatGrid(**kwargs)
                assert False, f"expected ValueError for {kwargs}"
            except ValueError:
                pass

    def test_node_count_and_shape(self):
        positions = _small_grid()
        assert positions.n_rows == _N_ROWS
        assert positions.n_cols == _N_COLS
        assert positions.n_nodes == _N_ROWS * _N_COLS
        assert positions.x.shape == (_N_ROWS * _N_COLS,)
        assert positions.y.shape == (_N_ROWS * _N_COLS,)

    def test_row_major_node_ordering_and_axis_mapping(self):
        """node_id = row * n_cols + col; x runs along columns (width),
        y runs along rows (height)."""
        positions = _small_grid()

        node0 = 0  # row 0, col 0
        assert (positions.x[node0], positions.y[node0]) == (0.0, 0.0)

        node5 = 5  # row 1, col 1 (5 = 1*4 + 1)
        assert (positions.x[node5], positions.y[node5]) == (_RESOLUTION_UM, _RESOLUTION_UM)

        node11 = 11  # row 2, col 3 (11 = 2*4 + 3), last node
        assert (positions.x[node11], positions.y[node11]) == (3 * _RESOLUTION_UM, 2 * _RESOLUTION_UM)

    def test_full_scale_dimensions_match_issue_acceptance_criteria(self):
        grid = FlatGrid(width_um=2000.0, height_um=20000.0, resolution_um=10.0)
        positions = grid.build()
        assert positions.n_cols == 200
        assert positions.n_rows == 2000
        assert positions.n_nodes == 400_000


class TestBuildGridNeighbours:

    def test_shape_and_dtype(self):
        positions = _small_grid()
        neighbours = build_grid_neighbours(positions)
        assert neighbours.shape == (positions.n_nodes, 4)
        assert neighbours.dtype == np.int32

    def test_interior_node_has_all_four_neighbours(self):
        positions = _small_grid()
        neighbours = build_grid_neighbours(positions)

        node5 = 5  # row 1, col 1 -- interior for a 3x4 grid
        up, down, left, right = neighbours[node5]
        assert up == 1      # row 0, col 1
        assert down == 9    # row 2, col 1
        assert left == 4    # row 1, col 0
        assert right == 6   # row 1, col 2

    def test_top_left_corner_has_two_missing_neighbours(self):
        positions = _small_grid()
        neighbours = build_grid_neighbours(positions)

        up, down, left, right = neighbours[0]  # row 0, col 0
        assert up == NO_NEIGHBOUR
        assert left == NO_NEIGHBOUR
        assert down == 4    # row 1, col 0
        assert right == 1   # row 0, col 1

    def test_bottom_right_corner_has_two_missing_neighbours(self):
        positions = _small_grid()
        neighbours = build_grid_neighbours(positions)

        last = positions.n_nodes - 1  # row 2, col 3
        up, down, left, right = neighbours[last]
        assert down == NO_NEIGHBOUR
        assert right == NO_NEIGHBOUR
        assert up == 7      # row 1, col 3
        assert left == 10   # row 2, col 2

    def test_edge_node_has_three_neighbours(self):
        positions = _small_grid()
        neighbours = build_grid_neighbours(positions)

        node1 = 1  # row 0, col 1 -- top edge, not a corner
        up, down, left, right = neighbours[node1]
        assert up == NO_NEIGHBOUR
        assert down == 5
        assert left == 0
        assert right == 2

    def test_every_neighbour_relationship_is_symmetric(self):
        """If A's right neighbour is B, B's left neighbour must be A (and
        likewise for up/down) -- true for any node on a regular grid."""
        positions = _small_grid()
        neighbours = build_grid_neighbours(positions)
        UP, DOWN, LEFT, RIGHT = range(4)

        for node_id in range(positions.n_nodes):
            up, down, left, right = neighbours[node_id]
            if right != NO_NEIGHBOUR:
                assert neighbours[right][LEFT] == node_id
            if left != NO_NEIGHBOUR:
                assert neighbours[left][RIGHT] == node_id
            if up != NO_NEIGHBOUR:
                assert neighbours[up][DOWN] == node_id
            if down != NO_NEIGHBOUR:
                assert neighbours[down][UP] == node_id


class TestPlaceGolgiCells:
    """Poisson-disk (hard-core) Golgi placement -- see
    DESIGN.md for why this
    algorithm was chosen and how uniformity is validated."""

    _WIDTH_UM = 1000.0
    _HEIGHT_UM = 1000.0
    _RATIO = 1.0 / 50.0  # -> 200 Golgi cells on a 100x100 grid

    def _grid(self) -> GridPositions:
        return FlatGrid(
            width_um=self._WIDTH_UM, height_um=self._HEIGHT_UM, resolution_um=10.0
        ).build()

    def test_rejects_invalid_ratio(self):
        positions = self._grid()
        for bad_ratio in [0.0, 1.0, -0.1, 1.5]:
            try:
                place_golgi_cells(positions, ratio=bad_ratio, seed=1)
                assert False, f"expected ValueError for ratio={bad_ratio}"
            except ValueError:
                pass

    def test_seed_is_required(self):
        """No default seed -- see ADR 0004: reproducibility must be an
        explicit choice, since placement feeds #7c and the Scientific Goal
        A/B/C sweeps."""
        positions = self._grid()
        try:
            place_golgi_cells(positions, ratio=self._RATIO)
            assert False, "expected TypeError: seed has no default"
        except TypeError:
            pass

    def test_returns_sorted_unique_node_ids_at_target_count(self):
        positions = self._grid()
        ids = place_golgi_cells(positions, ratio=self._RATIO, seed=1)
        expected_count = round(self._RATIO * positions.n_nodes)
        assert len(ids) == expected_count
        assert len(set(ids.tolist())) == len(ids), "node IDs must be unique"
        assert np.array_equal(ids, np.sort(ids))
        assert ids.dtype == np.int64
        assert ids.min() >= 0
        assert ids.max() < positions.n_nodes

    def test_deterministic_given_same_seed(self):
        positions = self._grid()
        first = place_golgi_cells(positions, ratio=self._RATIO, seed=7)
        second = place_golgi_cells(positions, ratio=self._RATIO, seed=7)
        assert np.array_equal(first, second)

    def test_different_seeds_give_different_placement(self):
        positions = self._grid()
        first = place_golgi_cells(positions, ratio=self._RATIO, seed=1)
        second = place_golgi_cells(positions, ratio=self._RATIO, seed=2)
        assert not np.array_equal(first, second)

    def test_placement_is_uniform_not_patchy(self):
        """Nearest-neighbour distances between placed Golgi cells must sit
        in a tight band around the expected spacing: no near-zero distances
        (would indicate patchiness/clustering) and no outsized gaps (would
        indicate holes) -- the acceptance criterion ADR 0004 commits to."""
        positions = self._grid()
        ids = place_golgi_cells(positions, ratio=self._RATIO, seed=3)
        x, y = positions.x[ids], positions.y[ids]

        nn = _nearest_neighbour_distances(x, y)
        expected_spacing = np.sqrt(self._WIDTH_UM * self._HEIGHT_UM / len(ids))

        assert nn.min() > 0.3 * expected_spacing, (
            f"nearest-neighbour distance {nn.min():.1f} um too close to zero "
            f"relative to expected spacing {expected_spacing:.1f} um -- patchy"
        )
        assert nn.max() < 2.0 * expected_spacing, (
            f"nearest-neighbour distance {nn.max():.1f} um far above expected "
            f"spacing {expected_spacing:.1f} um -- indicates a hole"
        )
        # Coefficient of variation bounds how spread-out the distances are
        # overall -- a regular/uniform placement has low variance relative
        # to its mean; a patchy one (clusters + gaps) would not.
        cv = nn.std() / nn.mean()
        assert cv < 0.35, f"nearest-neighbour distance CV {cv:.2f} too high -- patchy"

    def test_full_scale_matches_1_to_430_density(self):
        """Full 2mm x 20mm / 10um grid (~400k nodes, issue #7a's scale) at
        the project's committed 1:430 Golgi:node density."""
        positions = FlatGrid(width_um=2000.0, height_um=20000.0, resolution_um=10.0).build()
        ids = place_golgi_cells(positions, ratio=1.0 / 430.0, seed=1)
        expected_count = round(positions.n_nodes / 430.0)
        assert len(ids) == expected_count


class TestSampleUniformPositions:
    """Transient granule positions for locality-biased Golgi<->granule
    connectivity (DESIGN.md) -- continuous, not snapped to
    any discrete grid."""

    def test_shape_and_bounds(self):
        x, y = sample_uniform_positions(width_um=200.0, height_um=100.0, n=500, seed=1)
        assert x.shape == (500,)
        assert y.shape == (500,)
        assert x.min() >= 0.0 and x.max() <= 200.0
        assert y.min() >= 0.0 and y.max() <= 100.0

    def test_deterministic_given_same_seed(self):
        x1, y1 = sample_uniform_positions(100.0, 100.0, 50, seed=7)
        x2, y2 = sample_uniform_positions(100.0, 100.0, 50, seed=7)
        assert np.array_equal(x1, x2) and np.array_equal(y1, y2)

    def test_different_seeds_give_different_positions(self):
        x1, y1 = sample_uniform_positions(100.0, 100.0, 50, seed=1)
        x2, y2 = sample_uniform_positions(100.0, 100.0, 50, seed=2)
        assert not (np.array_equal(x1, x2) and np.array_equal(y1, y2))


class TestBuildGolgiDiffusionNeighbours:
    """Golgi<->Golgi gap-junction proximity graph (issue #7c) -- built
    directly from placed Golgi (x, y) positions, per
    DESIGN.md SS4."""

    def test_exact_edges_at_hand_picked_radius(self):
        # 4 points on a line: 0-1 are 100um apart, 1-2 are 200um apart,
        # 2-3 are 700um apart -- radius=150 should connect only (0,1).
        x = np.array([0.0, 100.0, 300.0, 1000.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        edges = build_golgi_diffusion_neighbours(x, y, radius_um=150.0)
        assert np.array_equal(edges, np.array([[0, 1]]))

    def test_larger_radius_connects_more_pairs(self):
        x = np.array([0.0, 100.0, 300.0, 1000.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        edges = build_golgi_diffusion_neighbours(x, y, radius_um=250.0)
        assert np.array_equal(edges, np.array([[0, 1], [1, 2]]))

    def test_no_self_edges_and_i_less_than_j(self):
        x = np.array([0.0, 50.0, 100.0, 150.0])
        y = np.zeros(4)
        edges = build_golgi_diffusion_neighbours(x, y, radius_um=1000.0)
        assert np.all(edges[:, 0] < edges[:, 1])

    def test_fewer_than_two_golgi_cells_returns_empty(self):
        assert build_golgi_diffusion_neighbours(np.array([0.0]), np.array([0.0])).shape == (0, 2)
        assert build_golgi_diffusion_neighbours(np.empty(0), np.empty(0)).shape == (0, 2)

    def test_auto_radius_connects_evenly_spaced_nearest_neighbours(self):
        """With no radius_um, auto-calibration (1.5x median NN distance) on
        evenly-spaced points (all NN distances identical) must connect every
        adjacent pair, since 1.5x the (uniform) NN distance always exceeds
        the NN distance itself."""
        x = np.array([0.0, 100.0, 200.0, 300.0])
        y = np.zeros(4)
        edges = build_golgi_diffusion_neighbours(x, y)
        pairs = {tuple(e) for e in edges}
        assert (0, 1) in pairs and (1, 2) in pairs and (2, 3) in pairs


class TestBuildConvergentNeighbours:
    """Generic locality-biased convergent connectivity (DESIGN.md),
    extracted from build_golgi_granule_neighbours (DESIGN.md) once other
    pathways (granule/Purkinje/stellate) needed the same algorithm with
    different populations in the target/source roles. The Golgi-specific
    behaviors (shape, divergence count, locality bias, seed determinism) are
    already fully covered by TestBuildGolgiGranuleNeighbours below via the
    thin-wrapper delegation -- this class only checks the generic contract
    and that the wrapper truly delegates rather than reimplementing."""

    def test_shape_and_contact_count(self):
        target_x, target_y = np.array([50.0, 950.0]), np.array([50.0, 950.0])
        rng = np.random.default_rng(0)
        source_x = rng.uniform(0, 1000, size=100)
        source_y = rng.uniform(0, 1000, size=100)

        target_idx, source_idx = build_convergent_neighbours(
            target_x, target_y, source_x, source_y, n_contacts=10, seed=1
        )

        assert target_idx.shape == (20,)
        assert source_idx.shape == (20,)
        assert set(target_idx.tolist()) == {0, 1}
        assert np.sum(target_idx == 0) == 10
        assert np.sum(target_idx == 1) == 10

    def test_rejects_n_contacts_exceeding_source_population(self):
        target_x, target_y = np.array([0.0]), np.array([0.0])
        source_x, source_y = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        try:
            build_convergent_neighbours(target_x, target_y, source_x, source_y, n_contacts=5, seed=0)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_golgi_granule_wrapper_delegates_exactly(self):
        """build_golgi_granule_neighbours must produce bit-identical results
        to calling build_convergent_neighbours directly with the same
        positional mapping -- proving it's a true thin wrapper, not a
        parallel reimplementation that could drift out of sync."""
        golgi_x, golgi_y = np.array([50.0, 950.0]), np.array([50.0, 950.0])
        rng = np.random.default_rng(0)
        granule_x = rng.uniform(0, 1000, size=100)
        granule_y = rng.uniform(0, 1000, size=100)

        via_wrapper = build_golgi_granule_neighbours(
            golgi_x, golgi_y, granule_x, granule_y, divergence=10, seed=1
        )
        via_generic = build_convergent_neighbours(
            golgi_x, golgi_y, granule_x, granule_y, n_contacts=10, seed=1
        )

        np.testing.assert_array_equal(via_wrapper[0], via_generic[0])
        np.testing.assert_array_equal(via_wrapper[1], via_generic[1])


class TestBuildGolgiGranuleNeighbours:
    """Locality-biased Golgi<->granule connectivity (DESIGN.md): one
    shared edge list serving both golgiToGranuleNeighbors (inhibitory,
    Golgi->granule) and granuleToGolgiNeighbors (excitatory, granule->Golgi)."""

    def test_shape_and_divergence_count(self):
        golgi_x, golgi_y = np.array([50.0, 950.0]), np.array([50.0, 950.0])
        granule_x, granule_y = sample_uniform_positions(1000.0, 1000.0, 200, seed=1)
        golgi_idx, granule_idx = build_golgi_granule_neighbours(
            golgi_x, golgi_y, granule_x, granule_y, divergence=10, seed=2
        )
        assert golgi_idx.shape == (20,)
        assert granule_idx.shape == (20,)
        assert set(golgi_idx.tolist()) == {0, 1}
        assert np.sum(golgi_idx == 0) == 10
        assert np.sum(golgi_idx == 1) == 10

    def test_no_duplicate_targets_within_one_golgi_cells_group(self):
        golgi_x, golgi_y = np.array([500.0]), np.array([500.0])
        granule_x, granule_y = sample_uniform_positions(1000.0, 1000.0, 100, seed=1)
        golgi_idx, granule_idx = build_golgi_granule_neighbours(
            golgi_x, golgi_y, granule_x, granule_y, divergence=20, seed=2
        )
        assert len(set(granule_idx.tolist())) == 20

    def test_rejects_divergence_exceeding_granule_population(self):
        golgi_x, golgi_y = np.array([0.0]), np.array([0.0])
        granule_x, granule_y = np.array([0.0, 1.0]), np.array([0.0, 0.0])
        try:
            build_golgi_granule_neighbours(golgi_x, golgi_y, granule_x, granule_y, divergence=5, seed=0)
            assert False, "expected ValueError: divergence exceeds granule population"
        except ValueError:
            pass

    def test_deterministic_given_same_seed(self):
        golgi_x, golgi_y = np.array([50.0, 950.0]), np.array([50.0, 950.0])
        granule_x, granule_y = sample_uniform_positions(1000.0, 1000.0, 200, seed=1)
        first = build_golgi_granule_neighbours(golgi_x, golgi_y, granule_x, granule_y, divergence=10, seed=5)
        second = build_golgi_granule_neighbours(golgi_x, golgi_y, granule_x, granule_y, divergence=10, seed=5)
        assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])

    def test_selection_is_locality_biased_not_uniform_random(self):
        """Chosen granule targets must sit meaningfully closer to their Golgi
        cell, on average, than the granule population as a whole -- the
        acceptance criterion that selection is locality-biased, not
        uniform-random. Golgi placed near a domain corner (not the center)
        and locality_sigma_um << domain size, so the population's mean
        distance is dominated by far-away cells and the bias signal is
        unambiguous rather than borderline."""
        golgi_x, golgi_y = np.array([100.0]), np.array([100.0])
        granule_x, granule_y = sample_uniform_positions(2000.0, 2000.0, 3000, seed=1)
        golgi_idx, granule_idx = build_golgi_granule_neighbours(
            golgi_x, golgi_y, granule_x, granule_y,
            divergence=100, seed=2, locality_sigma_um=150.0,
        )

        chosen_dist = np.sqrt(
            (granule_x[granule_idx] - golgi_x[0]) ** 2 + (granule_y[granule_idx] - golgi_y[0]) ** 2
        )
        all_dist = np.sqrt((granule_x - golgi_x[0]) ** 2 + (granule_y - golgi_y[0]) ** 2)

        assert chosen_dist.mean() < 0.3 * all_dist.mean(), (
            f"mean chosen-target distance {chosen_dist.mean():.1f} um not meaningfully "
            f"smaller than population mean {all_dist.mean():.1f} um -- selection doesn't look locality-biased"
        )
