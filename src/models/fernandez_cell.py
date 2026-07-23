"""
Fernandez, Engbers & Turner (2007) Purkinje-cell models — Phase 1 Python prototype.

Two models from the paper:
  Fernandez2007CellModel        — five-equation two-compartment (somatic + dendritic)
  Fernandez2007ReducedCellModel — reduced two-equation single-compartment

Units (internal, matching the paper):
  V [mV], t [ms], C [µF/cm²], g [mS/cm²], I_ext [µA/cm²], R [kΩ·cm²]

  step() accepts I_ext in nA, matching the project-wide units contract
  (see DESIGN.md). The paper's equations are written in current density
  (µA/cm²) with no defined cell area, so step() converts nA -> µA/cm²
  by dividing by 1000 (i.e. treating the model as if it had a 1 cm²
  membrane patch) before evaluating the RHS.

Integration: 4th-order Runge-Kutta as specified in the paper (dt ≤ 0.001 ms
recommended for full accuracy; RK4 remains stable at dt = 0.025 ms).

References:
  Fernandez et al. (2007) J Neurophysiol — Purkinje cell electroresponsiveness
"""

import numpy as np
from dataclasses import dataclass

from src.models.cell_model import CellModel

# --- Helpers ------------------------------------------------------------------


def _sigmoid_inf(V: float, V_half: float, k: float) -> float:
    """x_inf = 1 / (1 + exp[(V - V_half) / -k])  (paper eq.)"""
    return 1.0 / (1.0 + np.exp((V - V_half) / (-k)))


def _rk4_step(rhs, y: np.ndarray, dt: float) -> np.ndarray:
    """4th-order Runge-Kutta for a time-autonomous RHS (I_ext fixed over step)."""
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * dt * k1)
    k3 = rhs(y + 0.5 * dt * k2)
    k4 = rhs(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# nA (project-wide I_ext unit) -> µA/cm² (paper's current-density unit).
# The paper defines no cell area, so this is a 1000x scale-down rather
# than a physiological area conversion.
_NA_TO_UA_PER_CM2 = 1e-3


# --- Parameters ---------------------------------------------------------------


@dataclass
class Fernandez2007Params:
    """Five-equation two-compartment Purkinje-cell parameters.

    Units: C [µF/cm²], R [kΩ·cm²], g [mS/cm²], E [mV].
    """
    # Capacitances [µF/cm²]
    C_s: float = 1.5
    C_d: float = 1.5

    # Coupling resistance [kΩ·cm²]
    R: float = 0.75

    # Max conductances [mS/cm²]
    g_Na: float = 40.0
    g_Ks: float = 8.75
    g_IH: float = 0.03
    g_Kd_slow: float = 12.0
    g_leak: float = 0.032

    # Reversal potentials [mV]
    E_Na: float = 45.0
    E_K: float = -95.0
    E_ih: float = -20.0
    E_leak: float = -77.0


@dataclass
class Fernandez2007ReducedParams:
    """Reduced two-equation single-compartment Purkinje-cell parameters.

    Units: C [µF/cm²], g [mS/cm²], E [mV], tau [ms].
    """
    # Capacitance [µF/cm²]
    C: float = 1.5

    # Max conductances [mS/cm²]
    g_Na: float = 20.0
    g_K: float = 4.2
    g_leak: float = 0.05

    # Reversal potentials [mV]
    E_Na: float = 45.0
    E_K: float = -95.0
    E_leak: float = -77.0

    # Gating time constant [ms]
    tau_n: float = 0.6


# --- Five-equation model ------------------------------------------------------


class Fernandez2007CellModel(CellModel):
    """Five-equation two-compartment Purkinje-cell model.

    State: [V_s, V_d, h, ih, n_d]
      V_s — somatic membrane voltage [mV]
      V_d — dendritic membrane voltage [mV]
      h   — Na inactivation gate
      ih  — I_H activation gate
      n_d — slow dendritic K activation gate

    I_ext [nA] injected into the soma; converted internally to the
    paper's µA/cm² current density (see module docstring).
    """

    def __init__(self, params: Fernandez2007Params | None = None) -> None:
        self._p = params or Fernandez2007Params()
        self._state: np.ndarray = self._make_initial_state()

    # --- CellModel interface --------------------------------------------------

    def step(self, dt: float, I_ext: float = 0.0) -> None:
        I_ext_density = I_ext * _NA_TO_UA_PER_CM2

        def _rhs(y: np.ndarray) -> np.ndarray:
            return self._compute_rhs(y, I_ext_density)
        self._state = _rk4_step(_rhs, self._state, dt)

    def get_voltage(self) -> float:
        return float(self._state[0])  # V_s

    def reset(self) -> None:
        self._state = self._make_initial_state()

    # --- Extra accessors ------------------------------------------------------

    def get_dendritic_voltage(self) -> float:
        return float(self._state[1])  # V_d

    def get_slow_k_current(self) -> float:
        """Slow dendritic K+ current I_Kd,slow = g_Kd_slow * n_d * (V_d - E_K) [uA/cm^2]."""
        p = self._p
        V_d, n_d = self._state[1], self._state[4]
        return float(p.g_Kd_slow * n_d * (V_d - p.E_K))

    # --- Kinetics (private) ---------------------------------------------------

    def _m_inf(self, V_s: float) -> float:
        return _sigmoid_inf(V_s, V_half=-40.0, k=3.0)

    def _h_inf(self, V_s: float) -> float:
        return _sigmoid_inf(V_s, V_half=-40.0, k=-3.0)

    def _ih_inf(self, V_s: float) -> float:
        return _sigmoid_inf(V_s, V_half=-80.0, k=-3.0)

    def _nd_inf(self, V_d: float) -> float:
        return _sigmoid_inf(V_d, V_half=-35.0, k=3.0)

    def _tau_h(self, V_s: float) -> float:
        return 295.4 / (4.0 * (V_s + 50.0) ** 2 + 400.0) + 0.012

    def _tau_ih(self, V_s: float) -> float:
        return 100.0

    def _tau_nd(self, V_d: float) -> float:
        return 15.0

    # --- Internal -------------------------------------------------------------

    def _make_initial_state(self, V_s0: float = -70.0, V_d0: float = -70.0) -> np.ndarray:
        return np.array([
            V_s0,
            V_d0,
            self._h_inf(V_s0),
            self._ih_inf(V_s0),
            self._nd_inf(V_d0),
        ], dtype=float)

    def _compute_rhs(self, y: np.ndarray, I_ext: float) -> np.ndarray:
        p = self._p
        V_s, V_d, h, ih, n_d = y

        m_inf = self._m_inf(V_s)
        h_inf = self._h_inf(V_s)
        ih_inf = self._ih_inf(V_s)
        nd_inf = self._nd_inf(V_d)

        tau_h = self._tau_h(V_s)
        tau_ih = self._tau_ih(V_s)
        tau_nd = self._tau_nd(V_d)

        # Somatic voltage:
        # C_s dV_s/dt = (V_d - V_s)/R + I_E
        #              - g_Na m_inf h (V_s - E_Na)
        #              - g_Ks (1 - h) (V_s - E_K)
        #              - g_leak (V_s - E_leak)
        #              - g_IH ih (V_s - E_ih)
        dV_s = (
            (V_d - V_s) / p.R
            + I_ext
            - p.g_Na * m_inf * h * (V_s - p.E_Na)
            - p.g_Ks * (1.0 - h) * (V_s - p.E_K)
            - p.g_leak * (V_s - p.E_leak)
            - p.g_IH * ih * (V_s - p.E_ih)
        ) / p.C_s

        # Dendritic voltage:
        # C_d dV_d/dt = (V_s - V_d)/R
        #              - g_leak (V_d - E_leak)
        #              - g_Kd(slow) n_d (V_d - E_K)
        dV_d = (
            (V_s - V_d) / p.R
            - p.g_leak * (V_d - p.E_leak)
            - p.g_Kd_slow * n_d * (V_d - p.E_K)
        ) / p.C_d

        dh = (h_inf - h) / tau_h
        dih = (ih_inf - ih) / tau_ih
        dnd = (nd_inf - n_d) / tau_nd

        return np.array([dV_s, dV_d, dh, dih, dnd], dtype=float)


# --- Reduced two-equation model -----------------------------------------------


class Fernandez2007ReducedCellModel(CellModel):
    """Reduced two-equation single-compartment Purkinje-cell model.

    State: [V, n]
      V — membrane voltage [mV]
      n — combined activation / inactivation gate

    I_ext [nA]; converted internally to the paper's µA/cm² current
    density (see module docstring).
    """

    def __init__(self, params: Fernandez2007ReducedParams | None = None) -> None:
        self._p = params or Fernandez2007ReducedParams()
        self._state: np.ndarray = self._make_initial_state()

    # --- CellModel interface --------------------------------------------------

    def step(self, dt: float, I_ext: float = 0.0) -> None:
        I_ext_density = I_ext * _NA_TO_UA_PER_CM2

        def _rhs(y: np.ndarray) -> np.ndarray:
            return self._compute_rhs(y, I_ext_density)
        self._state = _rk4_step(_rhs, self._state, dt)

    def get_voltage(self) -> float:
        return float(self._state[0])

    def reset(self) -> None:
        self._state = self._make_initial_state()

    # --- Kinetics (private) ---------------------------------------------------

    def _m_inf(self, V: float) -> float:
        return _sigmoid_inf(V, V_half=-35.0, k=5.0)

    def _n_inf(self, V: float) -> float:
        return _sigmoid_inf(V, V_half=-36.0, k=5.0)

    # --- Internal -------------------------------------------------------------

    def _make_initial_state(self, V0: float = -70.0) -> np.ndarray:
        return np.array([V0, self._n_inf(V0)], dtype=float)

    def _compute_rhs(self, y: np.ndarray, I_ext: float) -> np.ndarray:
        p = self._p
        V, n = y

        m_inf = self._m_inf(V)
        n_inf = self._n_inf(V)

        # C dV/dt = I_E
        #           - g_Na m_inf (1 - n) (V - E_Na)
        #           - g_K n (V - E_K)
        #           - g_leak (V - E_leak)
        dV = (
            I_ext
            - p.g_Na * m_inf * (1.0 - n) * (V - p.E_Na)
            - p.g_K * n * (V - p.E_K)
            - p.g_leak * (V - p.E_leak)
        ) / p.C

        dn = (n_inf - n) / p.tau_n

        return np.array([dV, dn], dtype=float)
