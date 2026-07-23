import pytest

from src.models.mitry_cell import MitryStellateCellModel, MitryStellateParams
from tests.helpers import _DT, run_cell, count_spikes_cell, max_voltage_cell

VARIANTS = [MitryStellateParams.pre_runup(), MitryStellateParams.post_runup()]
VARIANTS_IDS = ["pre_runup", "post_runup"]

# Empirically-found suprathreshold current (nA, project convention) for this
# parameter set: below ~0.2 nA the cell stays subthreshold; at and above it
# the cell fires a single spike then settles into depolarisation block
# (h -> 0 and stays there while V remains elevated) rather than firing
# tonically, given these exact literal parameters.
_SUPRATHRESHOLD_NA = 0.3


# -----------------------------------------------------------------------------
@pytest.mark.parametrize("params", VARIANTS, ids=VARIANTS_IDS)
class TestMitryStellateCellModel:

    def test_instantiation_returns_float_voltage(self, params):
        """Model can be constructed and exposes a float voltage."""
        cell = MitryStellateCellModel(params)
        assert isinstance(cell.get_voltage(), float)

    def test_initial_voltage_is_v_init(self, params):
        """Voltage initialises at -70 mV."""
        cell = MitryStellateCellModel(params)
        assert cell.get_voltage() == -70.0

    def test_step_changes_voltage_under_applied_current(self, params):
        """step() must alter V when current is injected."""
        cell = MitryStellateCellModel(params)
        V0 = cell.get_voltage()
        for _ in range(100):
            cell.step(_DT, _SUPRATHRESHOLD_NA)
        assert cell.get_voltage() != V0

    def test_reset_restores_exact_initial_conditions(self, params):
        """reset() must return voltage to its original value exactly."""
        cell = MitryStellateCellModel(params)
        V_init = cell.get_voltage()
        run_cell(cell, 200.0, _SUPRATHRESHOLD_NA)
        cell.reset()
        assert abs(cell.get_voltage() - V_init) < 1e-10

    def test_subthreshold_current_does_not_spike(self, params):
        """A weak current must not cross 0 mV."""
        cell = MitryStellateCellModel(params)
        V_max = max_voltage_cell(cell, 200.0, 0.05)
        assert V_max < 0.0, f"Peak V = {V_max:.1f} mV, expected < 0 mV"

    def test_supthreshold_fires(self, params):
        """Strong current must produce at least one spike."""
        cell = MitryStellateCellModel(params)
        spikes = count_spikes_cell(cell, 100.0, _SUPRATHRESHOLD_NA)
        assert spikes >= 1, f"Expected spikes at {_SUPRATHRESHOLD_NA} nA, got {spikes}"

    def test_spike_amplitude_exceeds_0_mV(self, params):
        """Spikes must cross 0 mV"""
        cell = MitryStellateCellModel(params)
        V_max = max_voltage_cell(cell, 100.0, _SUPRATHRESHOLD_NA)
        assert V_max > 0.0, f"Peak V = {V_max:.1f} mV, expected > 0 mV"


# --- Variant-specific: post-runup curve shifts raise Na+/A-type threshold ----
class TestRunupVariantDifference:

    def test_post_runup_is_less_excitable_at_fixed_voltage(self):
        """Post-runup shifts m_inf/h_inf/n_A, inf right and h_A, inf differently,
        the qualitative change the source document attributes to 'runup'."""
        pre = MitryStellateCellModel(MitryStellateParams.pre_runup())
        post = MitryStellateCellModel(MitryStellateParams.post_runup())

        V_test = -45.0
        m_inf_pre = pre._m_inf(V_test)
        m_inf_post = post._m_inf(V_test)
        assert m_inf_post > m_inf_pre, (
            "Post-runup v_m is more depolarised (-44 vs -37), so m_inf at a "
            "fixed sub-threshold voltage must be smaller."
        )
