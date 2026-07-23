from src.models.dangelo_cell import DAngelo2001CellModel
from tests.helpers import _DT, run_cell, count_spikes_cell, max_voltage_cell, max_calcium_cell


class TestDAngelo2001CellModel:

    def test_instantiation_returns_float_voltage(self):
        """Model can be constructed and exposes a float voltage."""
        cell = DAngelo2001CellModel()
        assert isinstance(cell.get_voltage(), float)

    # --- Resting state --------------------------------------------------------

    def test_initial_voltage_is_v_init(self):
        """DAngelo2001 : Resting membrane potential in the model settled at
        –80 mV."""
        cell = DAngelo2001CellModel()
        assert cell.get_voltage() == -80.0

    def test_resting_potential_in_physiological_range(self):
        """After 200 ms at zero input, V must settle in [-85, -70] mV."""
        cell = DAngelo2001CellModel()
        run_cell(cell, 200.0, 0.0)
        V = cell.get_voltage()
        assert -85.0 < V < -70.0, f"Resting V = {V:.1f} mV, expected [-85, -70]."

    def test_resting_calcium_at_baseline(self):
        """Ca_i at rest must equal cai0 = 1e-4 mM."""
        cell = DAngelo2001CellModel()
        assert abs(cell.get_calcium() - 1e-4) < 1e-7

    # --- Stepping mechanisms --------------------------------------------------

    def test_step_changes_voltage_under_applied_current(self):
        """step() must alter V when depolarising current is injected."""
        cell = DAngelo2001CellModel()
        V0 = cell.get_voltage()
        for _ in range(100):
            cell.step(_DT, 0.02)
        assert cell.get_voltage() != V0

    def test_reset_restores_exact_initial_conditions(self):
        """reset() must return state to original values exactly."""
        cell = DAngelo2001CellModel()
        V_init = cell.get_voltage()
        Ca_init = cell.get_calcium()
        run_cell(cell, 500.0, 0.05)
        cell.reset()
        assert abs(cell.get_voltage() - V_init) < 1e-10
        assert abs(cell.get_calcium() - Ca_init) < 1e-10

    # --- Subthreshold ---------------------------------------------------------

    def test_subthreshold_input_does_not_spike(self):
        """0.005 nA (below threshold) must not produce spikes."""
        cell = DAngelo2001CellModel()
        spikes = count_spikes_cell(cell, 1000.0, 0.005)
        assert spikes == 0, f"Expected no spikes, got {spikes}"

    # --- Firing protocols (Based on Figure 6A) --------------------------------

    def test_threshold_current_fires(self):
        """0.016 nA (Fig 6A protocol) must produce at least one spike."""
        cell = DAngelo2001CellModel()
        spikes = count_spikes_cell(cell, 1000.0, 0.016)
        assert spikes >= 1, f"Expected spikes at 0.016 nA, got {spikes}"

    def test_spike_amplitude_exceeds_0_mV(self):
        """Spikes must cross 0 mV (conservative lower bound on amplitude)."""
        cell = DAngelo2001CellModel()
        V_max = max_voltage_cell(cell, 500.0, 0.02)
        assert V_max > 0.0, f"Peak V = {V_max:.1f} mV, expected > 0 mV"

    def test_high_input_fires_repetitively(self):
        """0.05 nA -> repetitive firing, at least 10 spikes in 500 ms."""
        cell = DAngelo2001CellModel()
        run_cell(cell, 100.0, 0.0)  # settle
        spikes = count_spikes_cell(cell, 500.0, 0.05)
        assert spikes >= 10, f"Expected repetitive firing, got {spikes} spikes"

    def test_firing_rate_increases_with_current(self):
        """Higher input must produce higher firing rate."""
        cell = DAngelo2001CellModel()
        run_cell(cell, 100.0, 0.0)
        spikes_low = count_spikes_cell(cell, 500.0, 0.02)
        cell.reset()
        run_cell(cell, 100.0, 0.0)
        spikes_high = count_spikes_cell(cell, 500.0, 0.05)
        assert (
            spikes_high > spikes_low
        ), f"Expected rate to increase: low={spikes_low}, high={spikes_high}"

    # --- Calcium dynamics -----------------------------------------------------

    def test_calcium_rises_during_spiking(self):
        """Ca_i must at least double above resting during spiking."""
        cell = DAngelo2001CellModel()
        Ca_init = cell.get_calcium()
        Ca_max = max_calcium_cell(cell, 500.0, 0.05)
        assert Ca_max > Ca_init * 2.0, (
            f"Ca_max = {Ca_max:.2e} mM, baseline = {Ca_init:.2e} mM"
        )

    def test_calcium_returns_toward_baseline_after_stimulus(self):
        """Ca_i must decay back toward cai0 after stimulus ends."""
        cell = DAngelo2001CellModel()
        run_cell(cell, 200.0, 0.05)
        Ca_peak = cell.get_calcium()
        run_cell(cell, 500.0, 0.0)
        Ca_after = cell.get_calcium()
        assert Ca_after < Ca_peak, (
            f"Ca did not decay: peak={Ca_peak:.2e}, after={Ca_after:.2e}"
        )
