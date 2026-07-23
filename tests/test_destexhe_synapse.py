import pytest

from src.models.destexhe_synapse import TwoStateDestexhe

_DT = 0.025

REST_V = -70.0  # mV -- presynaptic voltage with no spike
SPIKE_V = 30.0  # mV -- presynaptic voltage during spike

SYNAPSE_FACTORIES = [
    ("excitatory", lambda: TwoStateDestexhe.excitatory(gmax=1.0)),
    ("Inhibitory", lambda: TwoStateDestexhe.inhibitory(gmax=1.0)),
]

SYNAPSE_IDS = [name for name, _ in SYNAPSE_FACTORIES]

# ------------------------------------------------------------------------------


def _drive_pulse(syn, dt=_DT, pulse_ms=1.0, post_ms=60.0):
    """
    Hold V_pre above threshold for pulse_ms, then back at rest for post_ms.
    Returns the list of R sampled after every step.
    """
    trace = []
    for _ in range(round(pulse_ms / dt)):
        syn.step(dt, SPIKE_V)
        trace.append(syn.R)
    for _ in range(round(post_ms / dt)):
        syn.step(dt, REST_V)
        trace.append(syn.R)
    return trace


# --- Standard property tests: shared across both presets ----------------------
@pytest.mark.parametrize("name,make_synapse", SYNAPSE_FACTORIES, ids=SYNAPSE_IDS)
class TestTwoStateDestexhe:

    def test_initial_state_is_fully_closed(self, name, make_synapse):
        """R = 0 and current = 0 before any release."""
        syn = make_synapse()
        assert syn.R == 0.0
        assert syn.get_conductance() == 0.0
        assert syn.get_current(REST_V) == 0.0

    def test_rinf_and_rtau_match_closed_form(self, name, make_synapse):
        """Rinf and Rtau must equal the published closed-form
        expression exactly."""
        syn = make_synapse()
        p = syn._p
        expected_rinf = p.Cmax * p.Alpha / (p.Cmax * p.Alpha + p.Beta)
        excepted_rtau = 1.0 / (p.Alpha * p.Cmax + p.Beta)
        assert syn.Rinf == pytest.approx(expected_rinf)
        assert syn.Rtau == pytest.approx(excepted_rtau)

    def test_pulse_opens_channels_then_decays(self, name, make_synapse):
        """A single presynaptic pulse must raise R, then let it decay
        back toward 0."""
        syn = make_synapse()
        trace = _drive_pulse(syn)
        peak = max(trace)
        assert peak > 0.0
        assert trace[-1] < peak
        assert trace[-1] == pytest.approx(0.0, abs=1e-3)

    def test_peak_r_stays_within_rinf_bound(self, name, make_synapse):
        """R must never exceed the steady-state bound Rinf (physical:
        r in [0, 1])."""
        syn = make_synapse()
        trace = _drive_pulse(syn)
        assert max(trace) <= syn.Rinf + 1e-9

    def test_reset_restores_exact_initial_conditions(self, name, make_synapse):
        """reset() must return all kinetics state to its original
        values exactly."""
        syn = make_synapse()
        _drive_pulse(syn)
        syn.reset()
        assert syn.R == 0.0
        assert syn.C == 0.0
        assert syn.TimeCount == -1.0
        assert syn.lastrelease == -1000.0

    def test_current_direction_matches_reversal_potential(self, name, make_synapse):
        """I = g*(V-Erev): sign must flip either side of Erev."""
        syn = make_synapse()
        _drive_pulse(syn, post_ms=0.0)  # leave channels open (R > 0)
        assert syn.R > 0.0
        p = syn._p
        assert syn.get_current(p.Erev + 10.0) > 0.0
        assert syn.get_current(p.Erev - 10.0) < 0.0


# --- Preset-specific aparmeter checks -------------------------------------


def test_excitatory_preset_matches_ampa_mod_defaults():
    p = TwoStateDestexhe.excitatory(gmax=1.0)._p
    assert p.Alpha == 1.1
    assert p.Beta == 0.19
    assert p.Erev == 0.0
    assert p.Cmax == 1.0
    assert p.Cdur == 1.0
    assert p.Deadtime == 1.0


def test_inhibitory_preset_matches_gabaa_mod_defaults():
    p = TwoStateDestexhe.inhibitory(gmax=1.0)._p
    assert p.Alpha == 5.0
    assert p.Beta == 0.18
    assert p.Erev == -80.0
    assert p.Cmax == 1.0
    assert p.Cdur == 1.0
    assert p.Deadtime == 1.0
