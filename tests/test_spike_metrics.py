import numpy as np
import pytest

from src.simulation.spike_metrics import (
    binned_spike_counts,
    mean_offdiag_correlation,
    pairwise_correlation,
    pooled_isi_cv,
    population_power_spectrum,
    population_spike_count,
    population_spike_times_ms,
    psth,
    spike_count,
    spike_times_ms,
)


def _square_pulse_trace(n_frames: int, spike_frames: list[int], base: float = -70.0, peak: float = 30.0):
    """A synthetic (n_frames,) trace that's `peak` at exactly the given
    frames and `base` elsewhere -- so threshold=0.0 crossings land exactly
    at spike_frames."""
    V = np.full(n_frames, base, dtype=np.float64)
    V[spike_frames] = peak
    return V


class TestSpikeCountAndPopulationSpikeCount:
    def test_spike_count_matches_hand_count(self):
        V = _square_pulse_trace(20, [3, 7, 12])
        assert spike_count(V, threshold=0.0) == 3

    def test_population_spike_count_pools_across_cells(self):
        n_frames = 20
        V = np.stack(
            [_square_pulse_trace(n_frames, [3, 7]), _square_pulse_trace(n_frames, [5])], axis=1
        )
        assert population_spike_count(V, threshold=0.0) == 3

    def test_parity_with_old_sim_count_spikes_logic(self):
        """sim.py::count_spikes is being refactored to delegate to
        spike_count -- confirm they agree on random data."""
        rng = np.random.default_rng(0)
        V = rng.uniform(-80, 40, size=500)
        above = V > 0.0
        expected = int(np.sum(above[1:] & ~above[:-1]))
        assert spike_count(V, threshold=0.0) == expected


class TestPopulationSpikeTimesMs:
    def test_matches_single_cell_spike_times_ms_per_column(self):
        n_frames = 50
        t_ms = np.arange(n_frames) * 0.1
        V = np.stack(
            [_square_pulse_trace(n_frames, [5, 15, 30]), _square_pulse_trace(n_frames, [10])],
            axis=1,
        )
        result = population_spike_times_ms(t_ms, V, threshold=0.0)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], spike_times_ms(t_ms, V[:, 0], threshold=0.0))
        np.testing.assert_array_equal(result[1], spike_times_ms(t_ms, V[:, 1], threshold=0.0))

    def test_empty_for_cell_with_no_spikes(self):
        n_frames = 20
        t_ms = np.arange(n_frames) * 0.1
        V = np.stack([_square_pulse_trace(n_frames, []), _square_pulse_trace(n_frames, [5])], axis=1)
        result = population_spike_times_ms(t_ms, V, threshold=0.0)
        assert len(result[0]) == 0


class TestPooledIsiCv:
    def test_parity_with_pilot_probe_coupling_original(self):
        """Bit-for-bit parity with the pre-refactor local implementation in
        scripts/pilot_probe_coupling.py, on the same synthetic input."""

        def _original_pooled_isi_cv(V, dt_ms, threshold_mV):
            above = V > threshold_mV
            rising = above[1:] & ~above[:-1]
            all_isis = []
            for c in range(V.shape[1]):
                spike_steps = np.nonzero(rising[:, c])[0]
                if len(spike_steps) >= 2:
                    all_isis.append(np.diff(spike_steps) * dt_ms)
            if not all_isis:
                return float("nan"), 0
            isis = np.concatenate(all_isis)
            if len(isis) < 2 or isis.mean() == 0:
                return float("nan"), len(isis)
            return float(isis.std() / isis.mean()), len(isis)

        rng = np.random.default_rng(3)
        V = rng.uniform(-80, 40, size=(2000, 15))
        expected_cv, expected_n = _original_pooled_isi_cv(V, 0.1, -20.0)
        got_cv, got_n = pooled_isi_cv(V, 0.1, -20.0)
        assert got_n == expected_n
        if np.isnan(expected_cv):
            assert np.isnan(got_cv)
        else:
            assert got_cv == pytest.approx(expected_cv)

    def test_nan_with_too_few_isis(self):
        V = _square_pulse_trace(20, [5]).reshape(-1, 1)  # only one spike, no ISI possible
        cv, n = pooled_isi_cv(V, 0.1, 0.0)
        assert np.isnan(cv)
        assert n == 0

    def test_regular_isis_give_near_zero_cv(self):
        n_frames = 100
        V = _square_pulse_trace(n_frames, list(range(0, n_frames, 10))).reshape(-1, 1)
        cv, n = pooled_isi_cv(V, 1.0, 0.0)
        assert cv == pytest.approx(0.0, abs=1e-9)


class TestBinnedSpikeCountsAndPsth:
    def test_binned_counts_hand_checkable(self):
        spike_times_list = [np.array([1.0, 5.0, 9.0]), np.array([2.0])]
        counts = binned_spike_counts(spike_times_list, t_start_ms=0.0, t_end_ms=10.0, bin_ms=5.0)
        assert counts.shape == (2, 2)
        # bins are [0,5), [5,10) (numpy histogram convention: an edge value
        # belongs to the bin on its right) -- 1.0 -> bin0; 5.0, 9.0 -> bin1
        np.testing.assert_array_equal(counts[0], [1, 2])
        np.testing.assert_array_equal(counts[1], [1, 0])

    def test_psth_rate_units(self):
        # 2 cells, each spiking once per 5ms bin -> pooled pop rate = (1+1)/2 cells / 0.005s = 200 Hz
        spike_times_list = [np.array([1.0]), np.array([2.0])]
        bin_centers, rate_hz = psth(spike_times_list, t_start_ms=0.0, t_end_ms=5.0, bin_ms=5.0)
        assert len(bin_centers) == 1
        assert rate_hz[0] == pytest.approx(200.0)


class TestPopulationPowerSpectrum:
    def test_peak_near_injected_frequency(self):
        # A perfectly regular spike train's PSTH is a narrow pulse train
        # (Dirac-comb-like), whose spectrum has real energy at every harmonic
        # of the fundamental, not only the fundamental itself -- so this
        # checks for a genuine LOCAL peak at the injected frequency (clearly
        # elevated above the noise floor), not that it's the single global
        # maximum across the whole spectrum (a harmonic can legitimately
        # win that for a narrow-pulse signal).
        period_ms = 10.0
        duration_ms = 2000.0
        spike_times = np.arange(0.0, duration_ms, period_ms)
        freqs_hz, power = population_power_spectrum(
            [spike_times], t_start_ms=0.0, t_end_ms=duration_ms, bin_ms=1.0, nperseg=256
        )
        injected_freq_hz = 1000.0 / period_ms
        idx = np.argmin(np.abs(freqs_hz - injected_freq_hz))
        assert power[idx] > 5.0 * np.median(power)


class TestPairwiseCorrelation:
    def test_identical_spike_trains_correlate_near_one(self):
        # Uneven counts per bin (3, 1, 5) so each cell's binned series has
        # nonzero variance -- a constant per-bin count would make the
        # correlation genuinely undefined (NaN), not a test bug.
        times = np.array([1.0, 2.0, 3.0, 11.0, 21.0, 22.0, 23.0, 24.0, 25.0])
        spike_times_list = [times, times.copy()]
        counts = binned_spike_counts(spike_times_list, t_start_ms=0.0, t_end_ms=30.0, bin_ms=10.0)
        corr = pairwise_correlation(counts)
        assert corr.shape == (2, 2)
        assert corr[0, 1] == pytest.approx(1.0)

    def test_diagonal_is_one(self):
        spike_times_list = [np.array([1.0, 5.0, 15.0]), np.array([2.0, 12.0])]
        counts = binned_spike_counts(spike_times_list, t_start_ms=0.0, t_end_ms=20.0, bin_ms=5.0)
        corr = pairwise_correlation(counts)
        assert corr[0, 0] == pytest.approx(1.0)
        assert corr[1, 1] == pytest.approx(1.0)

    def test_single_cell_stays_2d(self):
        """Regression: np.corrcoef collapses a 1-row input to a 0-d scalar,
        which broke exp1_synchronization.py's imshow() on a sparse (n_golgi=1)
        discovery-scale run -- must stay a (1, 1) matrix."""
        spike_times_list = [np.array([1.0, 5.0, 15.0])]
        counts = binned_spike_counts(spike_times_list, t_start_ms=0.0, t_end_ms=20.0, bin_ms=5.0)
        corr = pairwise_correlation(counts)
        assert corr.shape == (1, 1)
        assert corr[0, 0] == pytest.approx(1.0)


class TestMeanOffdiagCorrelation:
    def test_hand_checkable_mean(self):
        corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]])
        # off-diagonal entries: 0.5, 0.3, 0.5, 0.7, 0.3, 0.7 -> mean = 0.5
        assert mean_offdiag_correlation(corr) == pytest.approx(0.5)

    def test_nan_entries_dropped_not_zeroed(self):
        corr = np.array([[1.0, np.nan, 0.5], [np.nan, 1.0, 0.5], [0.5, 0.5, 1.0]])
        # only the two real 0.5 entries should count
        assert mean_offdiag_correlation(corr) == pytest.approx(0.5)

    def test_single_cell_returns_nan(self):
        assert np.isnan(mean_offdiag_correlation(np.array([[1.0]])))

    def test_all_nan_offdiag_returns_nan(self):
        corr = np.array([[1.0, np.nan], [np.nan, 1.0]])
        assert np.isnan(mean_offdiag_correlation(corr))
