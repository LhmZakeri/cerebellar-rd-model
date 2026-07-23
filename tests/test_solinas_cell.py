from src.models.solinas_cell import Solinas2007CellModel
from tests.helpers import _DT, run_cell, count_spikes_cell, max_voltage_cell, max_calcium_cell


class TestSolinas2007CellModel:

    def test_instantiation_returns_float_voltage(self):
        """Model can be constructed and exposes a float voltage."""
        cell = Solinas2007CellModel()
        assert isinstance(cell.get_voltage(), float)

    # --- Resting state ---------------------------------------------------------

    def test_initial_voltage_is_v_init(self):
        """initial voltage for Solinas model"""
        cell = Solinas2007CellModel()
        assert cell.get_voltage() == -60

    def test_resting_calcium_at_baseline(self):
        """Ca_i at rest must equal cai0 = 50e-6 mM."""
        cell = Solinas2007CellModel()
        assert abs(cell.get_calcium() - 50e-6) < 1e-7

    # --- Stepping mechanisms ---------------------------------------------------

    def test_step_changes_voltage_under_current(self):
        """step() must alter V when depolarising current is injected."""
        cell = Solinas2007CellModel()
        V0 = cell.get_voltage()
        for _ in range(100):
            cell.step(_DT, 0.02)
        assert cell.get_voltage() != V0

    def test_reset_restores_exact_initial_conditions(self):
        """reset() must return state to original values exactly."""
        cell = Solinas2007CellModel()
        V_init = cell.get_voltage()
        Ca_init = cell.get_calcium()
        run_cell(cell, 500.0, 0.05)
        cell.reset()
        assert abs(cell.get_voltage() - V_init) < 1e-10
        assert abs(cell.get_calcium() - Ca_init) < 1e-10

    # --- Firing protocol -------------------------------------------------------

    def test_spontaneous_pacemaker_firing(self):
        """Golgi cell fires spontaneously (0 nA) after a 500ms settle."""
        cell = Solinas2007CellModel()
        run_cell(cell, 1000.0, 0.0)  # 1s to prestimulate

        spikes = count_spikes_cell(cell, 1000.0, 0.0)
        freq = spikes / 1.0  # spikes per second
        assert 1 <= freq <= 8, f"Pacemaker freq = {freq:.1f} Hz, expected 1-8 Hz"

    def test_spike_amplitude_exceeds_0_mV(self):
        """Spikes must cross 0 mV."""
        cell = Solinas2007CellModel()
        run_cell(cell, 1000.0, 0.0)  # 1s to prestimulate
        V_max = max_voltage_cell(cell, 1000.0, 0.0)
        assert V_max > 0.0, f"Peak V = {V_max:.1f} mV, expected > 0 mV"

    def test_firing_rate_increases_with_current(self):
        """Higher depolarising input must produce higher firing rate."""
        cell = Solinas2007CellModel()
        run_cell(cell, 500.0, 0.0)
        spikes_low = count_spikes_cell(cell, 500.0, 0.0)
        cell.reset()
        run_cell(cell, 500.0, 0.05)
        spikes_high = count_spikes_cell(cell, 500.0, 0.05)
        assert (
            spikes_high > spikes_low
        ), f"Rate must increase: low={spikes_low}, high={spikes_high}"

    # --- Calcium dynamics ------------------------------------------------------

    def test_calcium_rises_during_spiking(self):
        """Ca_i must exceed resting level during spiking."""
        cell = Solinas2007CellModel()
        Ca_init = cell.get_calcium()
        Ca_max = max_calcium_cell(cell, 400.0, 0.05)
        assert (
            Ca_max > Ca_init * 1.5
        ), f"Ca_max = {Ca_max:.2e} mM, baseline = {Ca_init:.2e} mM"
