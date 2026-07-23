import numpy as np
import pytest

from src.simulation.divergence_metrics import log_divergence_growth_rate, trajectory_distance


class TestTrajectoryDistance:
    def test_identical_trajectories_give_zero(self):
        V = np.random.default_rng(0).uniform(-80, 40, size=(50, 10))
        D = trajectory_distance(V, V.copy())
        np.testing.assert_array_almost_equal(D, np.zeros(50))

    def test_hand_checkable_l2_norm(self):
        V_A = np.array([[0.0, 0.0], [1.0, 1.0]])
        V_B = np.array([[3.0, 4.0], [1.0, 1.0]])
        D = trajectory_distance(V_A, V_B)
        np.testing.assert_array_almost_equal(D, [5.0, 0.0])

    def test_shape(self):
        V_A = np.zeros((100, 5))
        V_B = np.ones((100, 5))
        D = trajectory_distance(V_A, V_B)
        assert D.shape == (100,)


class TestLogDivergenceGrowthRate:
    def test_recovers_known_exponential_growth_rate(self):
        t_ms = np.arange(0.0, 100.0, 0.1)
        true_rate = 0.05  # per ms
        D_t = 1e-4 * np.exp(true_rate * t_ms)
        slope, log_D = log_divergence_growth_rate(D_t, t_ms, fit_start_ms=0.0, fit_end_ms=100.0)
        assert slope == pytest.approx(true_rate, rel=1e-6)
        assert len(log_D) == len(t_ms)

    def test_nan_with_too_few_points_in_window(self):
        t_ms = np.arange(0.0, 10.0, 1.0)
        D_t = np.ones(10)
        slope, log_D = log_divergence_growth_rate(D_t, t_ms, fit_start_ms=100.0, fit_end_ms=200.0)
        assert np.isnan(slope)
        assert len(log_D) == 0

    def test_zero_distance_entries_dropped(self):
        t_ms = np.array([0.0, 1.0, 2.0, 3.0])
        D_t = np.array([0.0, 0.0, 1e-3, 1e-2])
        slope, log_D = log_divergence_growth_rate(D_t, t_ms, fit_start_ms=0.0, fit_end_ms=4.0)
        assert len(log_D) == 2  # only the two nonzero entries
        assert np.isfinite(slope)

    def test_flat_distance_gives_near_zero_slope(self):
        t_ms = np.arange(0.0, 100.0, 1.0)
        D_t = np.full(100, 0.5)  # constant separation -- no growth
        slope, _ = log_divergence_growth_rate(D_t, t_ms, fit_start_ms=0.0, fit_end_ms=100.0)
        assert slope == pytest.approx(0.0, abs=1e-9)
