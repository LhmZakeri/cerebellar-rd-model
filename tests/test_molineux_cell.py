from src.models.molineux_cell import MolineuxStellateCellModel
from tests.helpers import _DT, run_cell, count_spikes_cell, max_voltage_cell

_SUBTHRESHOLD_NA = 0.0003
_SUPRATHRESHOLD_NA = 0.01


class TestMolineuxStellateCellModel:

    def test_instantiation_returns_float_voltage(self):
        """Model can be constructed and exposes a float voltage"""
        cell = MolineuxStellateCellModel()
        assert isinstance(cell.get_voltage(), float)

    def test_initial_voltage_is_v_init(self):
        """Voltage intialises at -65 mV (the source doc's quiescent range)."""
        cell = MolineuxStellateCellModel()
        assert cell.get_voltage() == -65.0

    def test_step_changes_voltage_under_applied_current(self):
        """step() must alter V when current is injected."""
        cell = MolineuxStellateCellModel()
        V0 = cell.get_voltage()
        for _ in range(100):
            cell.step(_DT, _SUPRATHRESHOLD_NA)
        assert cell.get_voltage() != V0

    def test_reset_restores_exact_initial_conditions(self):
        """reset() must return voltage to its original value exactly."""
        cell = MolineuxStellateCellModel()
        V_init = cell.get_voltage()
        run_cell(cell, 200.0, _SUPRATHRESHOLD_NA)
        cell.reset()
        assert abs(cell.get_voltage() - V_init) < 1e-10

    def test_subthreashold_current_does_not_spike(self):
        """A weak current must not cross 0 mV."""
        cell = MolineuxStellateCellModel()
        V_max = max_voltage_cell(cell, 200.0, _SUBTHRESHOLD_NA)
        assert V_max < 0.0, f"Peak V = {V_max:.1f} mV, expected < 0 mV"

    def test_suprathreshold_fires(self):
        """Strong current must produce at least one spike."""
        cell = MolineuxStellateCellModel()
        spikes = count_spikes_cell(cell, 200.0, _SUPRATHRESHOLD_NA)
        assert spikes >= 1, f"Expected spikes at {_SUPRATHRESHOLD_NA} nA, git {spikes}"

    def test_spike_amplitude_exceeds_0_mV(self):
        """Spikes must cross 0 mV."""
        cell = MolineuxStellateCellModel()
        V_max = max_voltage_cell(cell, 200.0, _SUPRATHRESHOLD_NA)
        assert V_max > 0.0, f"Peak V = {V_max:.1f} mV, expected > 0mV"


# --- A-type/T-type latency-voltage relationship ------------------------------


class TestLatencyVoltageRelationship:
    """More hyperpolarised holding potentials leave more h_A/h_T available,
    which delays the first spike after a depolarising step -- the paper's
    central result (A-type and T-type currents interact to set spike
    latency)."""

    def _latency_ms(
        self, V_hold: float, I_test: float, test_ms: float = 100.0
    ) -> float | None:
        cell = MolineuxStellateCellModel()
        # Equilibrate gates at the holding potential (steady state of a long
        # bias current at V_hold), matching the paper's holding-potential
        # protocol without needing to search for the exact bias current.
        cell._state[0] = V_hold
        cell._state[1] = cell._h_inf(V_hold)
        cell._state[2] = cell._n_inf(V_hold)
        cell._state[3] = cell._hT_inf(V_hold)
        cell._state[4] = cell._hA_inf(V_hold)

        n_steps = round(test_ms / _DT)
        for i in range(n_steps):
            cell.step(_DT, I_test)
            if cell.get_voltage() > 0.0:
                return i * _DT
        return None

    def test_hyperpolarised_hold_delays_first_spike(self):
        I_test = 0.3
        latency_hyperpolarised = self._latency_ms(-80.0, I_test)
        latency_depolarised = self._latency_ms(-60.0, I_test)

        assert latency_hyperpolarised is not None and latency_depolarised is not None
        assert latency_hyperpolarised > latency_depolarised, (
            f"Hold=-80mV latency={latency_hyperpolarised} must exceed"
            f"hold=-60mV latency={latency_depolarised}"
        )
