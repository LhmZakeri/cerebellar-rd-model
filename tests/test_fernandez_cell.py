import pytest

from src.models.fernandez_cell import Fernandez2007CellModel, Fernandez2007ReducedCellModel
from tests.helpers import _DT, run_cell, count_spikes_cell, max_voltage_cell

CELL_CLASSES = [Fernandez2007CellModel, Fernandez2007ReducedCellModel]


# --- Standard property tests: shared across both models ----------------------

@pytest.mark.parametrize("cell_cls", CELL_CLASSES)
class TestFernandezCellModels:

    def test_instantiation_returns_float_voltage(self, cell_cls):
        """Model can be constructed and exposes a float voltage."""
        cell = cell_cls()
        assert isinstance(cell.get_voltage(), float)

    def test_initial_voltage_is_v_init(self, cell_cls):
        """Voltage initialises at -70 mV."""
        cell = cell_cls()
        assert cell.get_voltage() == -70.0

    def test_step_changes_voltage_under_applied_current(self, cell_cls):
        """step() must alter V when current is injected."""
        cell = cell_cls()
        V0 = cell.get_voltage()
        for _ in range(100):
            cell.step(_DT, 1000.0)
        assert cell.get_voltage() != V0

    def test_reset_restores_exact_initial_conditions(self, cell_cls):
        """reset() must return voltage to its original value exactly."""
        cell = cell_cls()
        V_init = cell.get_voltage()
        run_cell(cell, 200.0, 1000.0)
        cell.reset()
        assert abs(cell.get_voltage() - V_init) < 1e-10

    def test_suprathreshold_fires(self, cell_cls):
        """1000 nA (~1.0 µA/cm²) must produce at least one spike in 200 ms."""
        cell = cell_cls()
        spikes = count_spikes_cell(cell, 200.0, 1000.0)
        assert spikes >= 1, f"Expected spikes at 1000 nA, got {spikes}"

    def test_spike_amplitude_exceeds_0_mV(self, cell_cls):
        """Spikes must cross 0 mV."""
        cell = cell_cls()
        V_max = max_voltage_cell(cell, 200.0, 1000.0)
        assert V_max > 0.0, f"Peak V = {V_max:.1f} mV, expected > 0 mV"


# --- Five-equation-model-only tests: dendritic compartment, f-I sweep --------

class TestFernandez2007CellModelExtra:

    def test_initial_dendritic_voltage_is_v_init(self):
        """Dendritic voltage initialises at -70 mV."""
        cell = Fernandez2007CellModel()
        assert cell.get_dendritic_voltage() == -70.0

    def test_reset_restores_dendritic_initial_condition(self):
        """reset() must return the dendritic compartment to its original value exactly."""
        cell = Fernandez2007CellModel()
        Vd_init = cell.get_dendritic_voltage()
        run_cell(cell, 200.0, 1000.0)
        cell.reset()
        assert abs(cell.get_dendritic_voltage() - Vd_init) < 1e-10

    def test_firing_rate_increases_with_current(self):
        """Higher current must produce higher firing rate."""
        cell = Fernandez2007CellModel()
        run_cell(cell, 100.0, 0.0)
        spikes_low = count_spikes_cell(cell, 300.0, 500.0)
        cell.reset()
        run_cell(cell, 100.0, 0.0)
        spikes_high = count_spikes_cell(cell, 300.0, 1500.0)
        assert spikes_high > spikes_low, (
            f"Rate must increase: low={spikes_low}, high={spikes_high}"
        )
