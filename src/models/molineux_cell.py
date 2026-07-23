"""
Molineux, Fernandez, Mehaffey & Turner (2005) cerebellar stellate-cell
model — Phase 1 Python prototype.

Five-dimensional Hodgkin-Huxley-type system: membrane voltage V plus four
dynamic gates h, n, h_T, h_A. The Na+ activation (m), T-type Ca2+
activation (m_T), and A-type K+ activation (n_A) variables are
instantaneous and set to their steady-state values, per the source
document's implementation notes. This is the earlier, first-order-gated
model that Mitry et al. (2020) (see mitry_cell.py) later revised to
higher-order (m^3 h, n^4) gating.

Units (internal, matching the source document):
  V [mV], t [ms], C [uF/cm^2], g [uS/cm^2], E [mV], tau [ms]

  step() accepts I_ext in nA, matching the project-wide units contract
  (see DESIGN.md). The source document states its explicit test current
  as a current density (0.9 uA/cm^2) with no membrane area defined, and
  flags this current/current-density mixing as an open ambiguity itself.
  Following the same-lab Fernandez2007 Purkinje-cell prototype (which
  shares this model's C/g/E magnitudes) and the mitry_cell.py precedent,
  step() converts nA -> pA by multiplying by 1000 (i.e. treats the model
  as a notional 1 cm^2 patch). This is an interface choice, not a paper
  value, and does not touch any equation or parameter below.

One documented typo is carried from the source note: the printed
I_A driving force uses E_Na, but since I_A is textually identified as an
A-type K+ current, the physically consistent (and implemented) driving
force is (V - E_K).

Integration: 4th-order Runge-Kutta, matching the codebase's other
standard-HH-gate models (see fernandez_cell.py, mitry_cell.py).

Reference:
  M. L. Molineux, F. Fernandez, W. H. Mehaffey, R. W. Turner (2005).
  "A-Type and T-Type Currents Interact to Produce a Novel Spike
  Latency-Voltage Relationship in Cerebellar Stellate Cells." J.
  Neurosci. 25(46):10863-10873.
"""

import numpy as np
from dataclasses import dataclass

from src.models.cell_model import CellModel

# --- Helpers ------------------------------------------------------------------


def _sigmoid_inf(V: float, V_half: float, k: float) -> float:
    """x_inf = 1 / (1 + exp[(V - V_half) / -k]) (source doc's general sigmoid form)."""
    return 1.0 / (1.0 + np.exp((V - V_half) / (-k)))


def _rk4_step(rhs, y: np.ndarray, dt: float) -> np.ndarray:
    """4th-order Runge-Kutta for a time-autonomous RHS (I_ext fixed over step)."""
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * dt * k1)
    k3 = rhs(y + 0.5 * dt * k2)
    k4 = rhs(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# nA (project-wide I_ext unit) -> pA (see module docstring for rationale).
_NA_TO_PA = 1000.0


# --- Parameters ---------------------------------------------------------------


@dataclass
class MolineuxStellateParams:
    """Molineux et al. (2005) stellate-cell parameters.

    Units: C [uF/cm^2], g [uS/cm^2], E [mV], v/s [mV], tau [ms].
    """

    # Membrane capacitance [uF/cm^2]
    C: float = 1.5

    # I_Na (first-order: I_Na = g_Na * m_inf * h * (V - E_Na))
    g_Na: float = 30.0
    E_Na: float = 45.0
    v_m: float = -35.0
    s_m: float = 4.0
    v_h: float = -35.0
    s_h: float = -4.0

    # tau_h(V) = y0 + 2*A*w / (4*pi*(V - V_c)^2 + w^2)  (Lorentzian fit)
    y0: float = 0.15
    A_th: float = 232.0
    w_th: float = 28.0
    V_c: float = -74.0

    # I_K (delayed rectifier, first-order: I_K = g_K * n * (V - E_K))
    g_K: float = 7.0
    E_K: float = -90.0
    v_n: float = -35.0
    s_n: float = 4.0
    tau_n: float = 0.5

    # I_leak
    g_leak: float = 0.1
    E_leak: float = -70.0

    # I_A (A-type K+; driving force V - E_K, correcting the source doc's
    # printed E_Na typo -- see module docstring)
    g_A: float = 16.0
    v_nA: float = -27.0
    s_nA: float = 8.8
    v_hA: float = -68.0
    s_hA: float = -6.6
    tau_hA: float = 15.0

    # I_T (T-type Ca2+)
    g_T: float = 0.55
    E_Ca: float = 22.0
    v_mT: float = -60.0
    s_mT: float = 3.0
    v_hT: float = -78.0
    s_hT: float = -3.75
    tau_hT: float = 15.0


# --- Model ----------------------------------------------------------------


class MolineuxStellateCellModel(CellModel):
    """Five-dimensional Molineux et al. (2005) cerebellar stellate-cell model.

    State: [V, h, n, h_T, h_A]
      V   — membrane voltage [mV]
      h   — Na+ inactivation gate
      n   — delayed-rectifier K+ activation gate
      h_T — T-type Ca2+ inactivation gate
      h_A — A-type K+ inactivation gate

    m (Na+ activation), m_T (T-type Ca2+ activation), and n_A (A-type K+
    activation) are instantaneous and computed from V at each evaluation,
    not carried as state.

    I_ext [nA] injected as I_E; converted internally to the source
    document's pA convention (see module docstring).
    """

    def __init__(self, params: MolineuxStellateParams | None = None) -> None:
        self._p = params or MolineuxStellateParams()
        self._state: np.ndarray = self._make_initial_state()

    # --- CellModel interface --------------------------------------------------

    def step(self, dt: float, I_ext: float = 0.0) -> None:
        I_E = I_ext * _NA_TO_PA

        def _rhs(y: np.ndarray) -> np.ndarray:
            return self._compute_rhs(y, I_E)
        self._state = _rk4_step(_rhs, self._state, dt)

    def get_voltage(self) -> float:
        return float(self._state[0])

    def reset(self) -> None:
        self._state = self._make_initial_state()

    # --- Kinetics (private) ---------------------------------------------------

    def _m_inf(self, V: float) -> float:
        return _sigmoid_inf(V, self._p.v_m, self._p.s_m)

    def _h_inf(self, V: float) -> float:
        return _sigmoid_inf(V, self._p.v_h, self._p.s_h)

    def _n_inf(self, V: float) -> float:
        return _sigmoid_inf(V, self._p.v_n, self._p.s_n)

    def _nA_inf(self, V: float) -> float:
        return _sigmoid_inf(V, self._p.v_nA, self._p.s_nA)

    def _hA_inf(self, V: float) -> float:
        return _sigmoid_inf(V, self._p.v_hA, self._p.s_hA)

    def _mT_inf(self, V: float) -> float:
        return _sigmoid_inf(V, self._p.v_mT, self._p.s_mT)

    def _hT_inf(self, V: float) -> float:
        return _sigmoid_inf(V, self._p.v_hT, self._p.s_hT)

    def _tau_h(self, V: float) -> float:
        p = self._p
        return p.y0 + (2.0 * p.A_th * p.w_th) / (4.0 * np.pi * (V - p.V_c) ** 2 + p.w_th ** 2)

    # --- Internal -------------------------------------------------------------

    def _make_initial_state(self, V0: float = -65.0) -> np.ndarray:
        return np.array([
            V0,
            self._h_inf(V0),
            self._n_inf(V0),
            self._hT_inf(V0),
            self._hA_inf(V0),
        ], dtype=float)

    def _compute_rhs(self, y: np.ndarray, I_E: float) -> np.ndarray:
        p = self._p
        V, h, n, h_T, h_A = y

        m_inf = self._m_inf(V)
        h_inf = self._h_inf(V)
        n_inf = self._n_inf(V)
        nA_inf = self._nA_inf(V)
        hA_inf = self._hA_inf(V)
        mT_inf = self._mT_inf(V)
        hT_inf = self._hT_inf(V)

        tau_h = self._tau_h(V)

        # I_Na   = g_Na * m_inf * h * (V - E_Na)
        # I_K    = g_K * n * (V - E_K)
        # I_leak = g_leak * (V - E_leak)
        # I_A    = g_A * n_A,inf * h_A * (V - E_K)
        # I_T    = g_T * m_T,inf * h_T * (V - E_Ca)
        I_Na = p.g_Na * m_inf * h * (V - p.E_Na)
        I_K = p.g_K * n * (V - p.E_K)
        I_leak = p.g_leak * (V - p.E_leak)
        I_A = p.g_A * nA_inf * h_A * (V - p.E_K)
        I_T = p.g_T * mT_inf * h_T * (V - p.E_Ca)

        # C dV/dt = I_E - I_Na - I_K - I_leak - I_A - I_T
        dV = (I_E - I_Na - I_K - I_leak - I_A - I_T) / p.C
        dh = (h_inf - h) / tau_h
        dn = (n_inf - n) / p.tau_n
        dhT = (hT_inf - h_T) / p.tau_hT
        dhA = (hA_inf - h_A) / p.tau_hA

        return np.array([dV, dh, dn, dhT, dhA], dtype=float)
