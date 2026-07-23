"""
Mitry, Alexander, Farjami, Bowie & Khadra (2020) cerebellar stellate-cell
model — Phase 1 Python prototype.

Six-dimensional Hodgkin-Huxley-type system: membrane voltage V plus five
dynamic gates h, n, n_A, h_A, h_T. The Na+ and T-type Ca2+ activation
variables (m, m_T) are instantaneous and set to their steady-state values,
per the source document's implementation notes.

Two curve variants, differing only in the steady-state activation /
inactivation curves for I_Na and I_A (all conductances, reversal
potentials, and remaining kinetics are shared):
  pre_runup()  — t = 0 min fit
  post_runup() — t = 25 min fit

Units (internal, matching the source document):
  V [mV], t [ms], C_m [uF/cm^2], g [uS/cm^2], E [mV], tau [ms]

  step() accepts I_ext in nA, matching the project-wide units contract
  (see DESIGN.md). The source document's I_app/I_test/I_bias are given in
  pA directly (not a current density — no membrane area is defined), so
  step() converts nA -> pA by multiplying by 1000.

Integration: 4th-order Runge-Kutta, matching the codebase's other
standard-HH-gate models (see fernandez_cell.py).

Reference:
  J. Mitry, R. P. D. Alexander, S. Farjami, D. Bowie, A. Khadra (2020).
  "Modeling excitability in cerebellar stellate cells: Temporal changes
  in threshold, latency and frequency of firing." Commun. Nonlinear Sci.
  Numer. Simul. 82:105014.
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


# nA (project-wide I_ext unit) -> pA (the source document's I_app unit).
_NA_TO_PA = 1000.0


# --- Parameters ---------------------------------------------------------------


@dataclass
class MitryStellateParams:
    """Mitry et al. (2020) stellate-cell parameters (pre-runup by default).

    Units: C_m [uF/cm^2], g [uS/cm^2], E [mV], v/s [mV], tau [ms].
    """

    # Membrane capacitance [uF/cm^2]
    C_m: float = 15.0148

    # I_Na
    g_Na: float = 3.4
    E_Na: float = 55.0
    v_m: float = -37.0
    s_m: float = 3.0
    v_h: float = -40.0
    s_h: float = -4.0

    # tau_h(V) = y0 + 2*A_th*w_th / (4*pi*(V - V_c)^2 + w_th^2)
    y0: float = 0.1
    A_th: float = 322.0
    w_th: float = 46.0
    V_c: float = -74.0

    # I_K (delayed rectifier)
    g_K: float = 9.0556
    E_K: float = -80.0
    v_n: float = -23.0
    s_n: float = 5.0

    # I_L (leak)
    g_L: float = 0.07407
    E_L: float = -38.0

    # I_A (A-type K+)
    g_A: float = 15.0159
    v_nA: float = -27.0
    s_nA: float = 13.2
    tau_nA: float = 5.0
    v_hA: float = -80.0
    s_hA: float = -6.5
    tau_hA: float = 10.0

    # I_T (T-type Ca2+)
    g_T: float = 0.45045
    E_Ca: float = 22.0
    v_mT: float = -50.0
    s_mT: float = 3.0
    v_hT: float = -68.0
    s_hT: float = -3.75
    tau_hT: float = 15.0

    @classmethod
    def pre_runup(cls) -> "MitryStellateParams":
        """t = 0 min fit (the dataclass defaults)."""
        return cls()

    @classmethod
    def post_runup(cls) -> "MitryStellateParams":
        """t = 25 min fit: shifted I_Na and I_A steady-state curves only."""
        return cls(
            v_m=-44.0,
            v_h=-48.5,
            v_nA=-41.0,
            v_hA=-96.0,
            s_hA=-9.2,
        )


# --- Model ----------------------------------------------------------------


class MitryStellateCellModel(CellModel):
    """Six-dimensional Mitry et al. (2020) cerebellar stellate-cell model.

    State: [V, h, n, n_A, h_A, h_T]
      V   — membrane voltage [mV]
      h   — Na+ inactivation gate
      n   — delayed-rectifier K+ activation gate
      n_A — A-type K+ activation gate
      h_A — A-type K+ inactivation gate
      h_T — T-type Ca2+ inactivation gate

    m (Na+ activation) and m_T (T-type Ca2+ activation) are instantaneous
    and computed from V at each evaluation, not carried as state.

    I_ext [nA] injected as I_app; converted internally to the source
    document's pA convention (see module docstring).
    """

    def __init__(self, params: MitryStellateParams | None = None) -> None:
        self._p = params or MitryStellateParams()
        self._state: np.ndarray = self._make_initial_state()

    # --- CellModel interface --------------------------------------------------

    def step(self, dt: float, I_ext: float = 0.0) -> None:
        I_app = I_ext * _NA_TO_PA

        def _rhs(y: np.ndarray) -> np.ndarray:
            return self._compute_rhs(y, I_app)
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

    def _tau_n(self, V: float) -> float:
        return 6.0 / (1.0 + np.exp((V + 23.0) / 15.0))

    # --- Internal -------------------------------------------------------------

    def _make_initial_state(self, V0: float = -70.0) -> np.ndarray:
        return np.array([
            V0,
            self._h_inf(V0),
            self._n_inf(V0),
            self._nA_inf(V0),
            self._hA_inf(V0),
            self._hT_inf(V0),
        ], dtype=float)

    def _compute_rhs(self, y: np.ndarray, I_app: float) -> np.ndarray:
        p = self._p
        V, h, n, n_A, h_A, h_T = y

        m_inf = self._m_inf(V)
        h_inf = self._h_inf(V)
        n_inf = self._n_inf(V)
        nA_inf = self._nA_inf(V)
        hA_inf = self._hA_inf(V)
        mT_inf = self._mT_inf(V)
        hT_inf = self._hT_inf(V)

        tau_h = self._tau_h(V)
        tau_n = self._tau_n(V)

        # I_Na = g_Na * m_inf^3 * h * (V - E_Na)
        # I_K  = g_K * n^4 * (V - E_K)
        # I_L  = g_L * (V - E_L)
        # I_A  = g_A * n_A * h_A * (V - E_K)
        # I_T  = g_T * m_T,inf * h_T * (V - E_Ca)
        I_Na = p.g_Na * m_inf ** 3 * h * (V - p.E_Na)
        I_K = p.g_K * n ** 4 * (V - p.E_K)
        I_L = p.g_L * (V - p.E_L)
        I_A = p.g_A * n_A * h_A * (V - p.E_K)
        I_T = p.g_T * mT_inf * h_T * (V - p.E_Ca)

        # C_m dV/dt = -I_Na - I_K - I_L - I_A - I_T + I_app
        dV = (-I_Na - I_K - I_L - I_A - I_T + I_app) / p.C_m
        dh = (h_inf - h) / tau_h
        dn = (n_inf - n) / tau_n
        dnA = (nA_inf - n_A) / p.tau_nA
        dhA = (hA_inf - h_A) / p.tau_hA
        dhT = (hT_inf - h_T) / p.tau_hT

        return np.array([dV, dh, dn, dnA, dhA, dhT], dtype=float)
