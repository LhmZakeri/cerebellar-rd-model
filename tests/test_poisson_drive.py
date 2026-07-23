import numpy as np
import pytest

from src.simulation.poisson_drive import ExponentialCurrentFilter, generate_poisson_spike_train


class TestGeneratePoissonSpikeTrain:
    def test_shape_and_dtype(self):
        spikes = generate_poisson_spike_train(
            n_fibers=5, n_steps=100, dt_ms=0.01, rate_hz=20.0, seed=0
        )
        assert spikes.shape == (100, 5)
        assert spikes.dtype == bool

    def test_zero_outside_active_window(self):
        spikes = generate_poisson_spike_train(
            n_fibers=10, n_steps=1000, dt_ms=0.1, rate_hz=70.0, seed=1,
            on_step=200, off_step=800,
        )
        assert not spikes[:200].any()
        assert not spikes[800:].any()

    def test_off_step_none_means_active_to_end(self):
        spikes = generate_poisson_spike_train(
            n_fibers=10, n_steps=500, dt_ms=0.1, rate_hz=70.0, seed=1, on_step=100,
        )
        # some activity must appear right up to the end at this rate/dt/n_fibers
        assert spikes[400:500].any()

    def test_empirical_rate_within_tolerance(self):
        n_fibers = 200
        n_steps = 200_000
        dt_ms = 0.01
        rate_hz = 40.0
        spikes = generate_poisson_spike_train(
            n_fibers=n_fibers, n_steps=n_steps, dt_ms=dt_ms, rate_hz=rate_hz, seed=2
        )
        duration_s = n_steps * dt_ms / 1000.0
        empirical_rate_hz = spikes.sum() / (n_fibers * duration_s)
        assert abs(empirical_rate_hz - rate_hz) / rate_hz < 0.05

    def test_seed_reproducibility(self):
        a = generate_poisson_spike_train(n_fibers=8, n_steps=500, dt_ms=0.01, rate_hz=20.0, seed=7)
        b = generate_poisson_spike_train(n_fibers=8, n_steps=500, dt_ms=0.01, rate_hz=20.0, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = generate_poisson_spike_train(n_fibers=8, n_steps=2000, dt_ms=0.01, rate_hz=40.0, seed=1)
        b = generate_poisson_spike_train(n_fibers=8, n_steps=2000, dt_ms=0.01, rate_hz=40.0, seed=2)
        assert not np.array_equal(a, b)

    def test_raises_when_p_spike_too_high(self):
        with pytest.raises(ValueError):
            generate_poisson_spike_train(n_fibers=1, n_steps=10, dt_ms=10.0, rate_hz=200.0, seed=0)


class TestExponentialCurrentFilter:
    def test_zero_input_stays_zero(self):
        f = ExponentialCurrentFilter(n_channels=3)
        spike = np.zeros(3, dtype=bool)
        for _ in range(10):
            I = f.step(0.01, spike)
        np.testing.assert_array_equal(I, np.zeros(3))

    def test_single_spike_peak_equals_amplitude(self):
        f = ExponentialCurrentFilter(n_channels=1, amplitude_nA=0.05, tau_ms=3.0)
        I = f.step(0.01, np.array([True]))
        assert I[0] == pytest.approx(0.05)

    def test_decay_half_life_matches_tau(self):
        tau_ms = 3.0
        f = ExponentialCurrentFilter(n_channels=1, amplitude_nA=1.0, tau_ms=tau_ms)
        dt_ms = 0.5
        f.step(dt_ms, np.array([True]))  # I = amplitude, no decay applied yet
        n_decay_steps = round(tau_ms / dt_ms)
        for _ in range(n_decay_steps):
            I = f.step(dt_ms, np.array([False]))
        # after ~tau_ms of pure decay, I should be ~1/e of its peak
        assert I[0] == pytest.approx(np.exp(-1.0), rel=0.05)

    def test_reset_zeroes_state(self):
        f = ExponentialCurrentFilter(n_channels=2, amplitude_nA=0.05)
        f.step(0.01, np.array([True, True]))
        f.reset()
        I = f.step(0.01, np.array([False, False]))
        np.testing.assert_array_equal(I, np.zeros(2))

    def test_independent_channels(self):
        f = ExponentialCurrentFilter(n_channels=2, amplitude_nA=0.05)
        I = f.step(0.01, np.array([True, False]))
        assert I[0] == pytest.approx(0.05)
        assert I[1] == 0.0
