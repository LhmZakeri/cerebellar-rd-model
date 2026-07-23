"""
Numba-accelerated batch kernel for the Fernandez et al. (2007) five-equation
two-compartment Purkinje cell.

Operates on arrays of N nodes at once (RK4, same as Fernandez2007CellModel in
fernandez_cell.py). Parameters are read once from that module's own
Fernandez2007Params dataclass -- no duplicated numbers. See DESIGN.md
("Performance strategy: Numba") for why this replaced the earlier planned
C++ port, and src/models/dangelo_numba.py for the first model on this path.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from src.models.fernandez_cell import Fernandez2007CellModel, Fernandez2007Params

_NA_TO_UA_PER_CM2 = 1e-3


@njit(cache=True, inline="always")
def _sigmoid_inf(V, V_half, k):
    return 1.0 / (1.0 + np.exp((V - V_half) / (-k)))


@njit(cache=True, inline="always")
def _tau_h_fn(V_s):
    return 295.4 / (4.0 * (V_s + 50.0) ** 2 + 400.0) + 0.012


@njit(cache=True, inline="always")
def _rhs(V_s, V_d, h, ih, n_d, I_ext,
         g_Na, g_Ks, g_IH, g_Kd_slow, g_leak,
         E_Na, E_K, E_ih, E_leak, R, C_s, C_d, tau_ih_const, tau_nd_const):
    m_inf = _sigmoid_inf(V_s, -40.0, 3.0)
    h_inf = _sigmoid_inf(V_s, -40.0, -3.0)
    ih_inf = _sigmoid_inf(V_s, -80.0, -3.0)
    nd_inf = _sigmoid_inf(V_d, -35.0, 3.0)
    tau_h = _tau_h_fn(V_s)

    dV_s = (
        (V_d - V_s) / R
        + I_ext
        - g_Na * m_inf * h * (V_s - E_Na)
        - g_Ks * (1.0 - h) * (V_s - E_K)
        - g_leak * (V_s - E_leak)
        - g_IH * ih * (V_s - E_ih)
    ) / C_s
    dV_d = (
        (V_s - V_d) / R
        - g_leak * (V_d - E_leak)
        - g_Kd_slow * n_d * (V_d - E_K)
    ) / C_d
    dh = (h_inf - h) / tau_h
    dih = (ih_inf - ih) / tau_ih_const
    dnd = (nd_inf - n_d) / tau_nd_const
    return dV_s, dV_d, dh, dih, dnd


@njit(parallel=True, cache=True, fastmath=False)
def _step_all(V_s, V_d, h, ih, n_d, I_ext, dt,
              g_Na, g_Ks, g_IH, g_Kd_slow, g_leak_arr,
              E_Na, E_K, E_ih, E_leak, R, C_s, C_d, tau_ih_const, tau_nd_const):
    """Advance every node by one dt via RK4. Mutates all state arrays in place."""
    n = V_s.shape[0]
    for i in prange(n):
        g_leak = g_leak_arr[i]
        vs, vd, hh, iih, nd = V_s[i], V_d[i], h[i], ih[i], n_d[i]
        Ie = I_ext[i]

        k1 = _rhs(vs, vd, hh, iih, nd, Ie, g_Na, g_Ks, g_IH, g_Kd_slow,
                  g_leak, E_Na, E_K, E_ih, E_leak, R, C_s, C_d, tau_ih_const, tau_nd_const)
        k2 = _rhs(vs + 0.5 * dt * k1[0], vd + 0.5 * dt * k1[1], hh + 0.5 * dt * k1[2],
                  iih + 0.5 * dt * k1[3], nd + 0.5 * dt * k1[4], Ie,
                  g_Na, g_Ks, g_IH, g_Kd_slow, g_leak, E_Na, E_K, E_ih, E_leak,
                  R, C_s, C_d, tau_ih_const, tau_nd_const)
        k3 = _rhs(vs + 0.5 * dt * k2[0], vd + 0.5 * dt * k2[1], hh + 0.5 * dt * k2[2],
                  iih + 0.5 * dt * k2[3], nd + 0.5 * dt * k2[4], Ie,
                  g_Na, g_Ks, g_IH, g_Kd_slow, g_leak, E_Na, E_K, E_ih, E_leak,
                  R, C_s, C_d, tau_ih_const, tau_nd_const)
        k4 = _rhs(vs + dt * k3[0], vd + dt * k3[1], hh + dt * k3[2],
                  iih + dt * k3[3], nd + dt * k3[4], Ie,
                  g_Na, g_Ks, g_IH, g_Kd_slow, g_leak, E_Na, E_K, E_ih, E_leak,
                  R, C_s, C_d, tau_ih_const, tau_nd_const)

        V_s[i] = vs + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        V_d[i] = vd + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        h[i] = hh + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        ih[i] = iih + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        n_d[i] = nd + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])


class FernandezBatch:
    """N independent Fernandez2007 Purkinje cells, stepped together via one Numba kernel.

    Same math as Fernandez2007CellModel (fernandez_cell.py) -- this is not a
    reimplementation, it reads that module's own Fernandez2007Params values
    directly. Use this when N is large enough that the per-node Numba
    speedup matters; use Fernandez2007CellModel for single-node work.
    """

    def __init__(
        self,
        n_nodes: int,
        params: Fernandez2007Params | None = None,
        heterogeneity_seed: int | None = None,
    ) -> None:
        """heterogeneity_seed: None (default) -> every node identical, exact
        prior behavior. An int -> Sou11-style per-node heterogeneity (DESIGN.md): g_leak and initial V_s/V_d each independently drawn
        uniform +/-20% around their base value, one draw per node. NO
        membrane-area heterogeneity here -- this model's equations are
        written directly in current-density (uA/cm^2) with "no defined cell
        area" (this module's own docstring), so there is no area parameter
        to jitter."""
        self.n_nodes = n_nodes
        self._p = params or Fernandez2007Params()
        self._heterogeneity_seed = heterogeneity_seed
        if heterogeneity_seed is None:
            self._g_leak = np.full(n_nodes, self._p.g_leak, dtype=np.float64)
        else:
            rng = np.random.default_rng(heterogeneity_seed)
            self._g_leak = self._p.g_leak * rng.uniform(0.8, 1.2, size=n_nodes)
        self.reset()

    def reset(self) -> None:
        # Single scalar instantiation gives us the exact validated initial
        # state (fernandez_cell.py's own _make_initial_state()); broadcast it
        # to every node, EXCEPT V_s/V_d, which each get their own independent
        # +/-20% draw per node when heterogeneity_seed is set (DESIGN.md).
        initial = Fernandez2007CellModel(self._p)._state
        self.V_s = np.full(self.n_nodes, initial[0], dtype=np.float64)
        self.V_d = np.full(self.n_nodes, initial[1], dtype=np.float64)
        self.h = np.full(self.n_nodes, initial[2], dtype=np.float64)
        self.ih = np.full(self.n_nodes, initial[3], dtype=np.float64)
        self.n_d = np.full(self.n_nodes, initial[4], dtype=np.float64)
        if self._heterogeneity_seed is not None:
            rng = np.random.default_rng(self._heterogeneity_seed + 1)
            self.V_s = self.V_s * rng.uniform(0.8, 1.2, size=self.n_nodes)
            self.V_d = self.V_d * rng.uniform(0.8, 1.2, size=self.n_nodes)

    def step(self, dt: float, I_ext_nA) -> None:
        """Advance all nodes by dt [ms]. I_ext_nA: scalar or per-node array [nA]."""
        if np.isscalar(I_ext_nA):
            I_ext = np.full(self.n_nodes, I_ext_nA * _NA_TO_UA_PER_CM2, dtype=np.float64)
        else:
            I_ext = np.asarray(I_ext_nA, dtype=np.float64) * _NA_TO_UA_PER_CM2

        p = self._p
        _step_all(
            self.V_s, self.V_d, self.h, self.ih, self.n_d, I_ext, dt,
            p.g_Na, p.g_Ks, p.g_IH, p.g_Kd_slow, self._g_leak,
            p.E_Na, p.E_K, p.E_ih, p.E_leak, p.R, p.C_s, p.C_d,
            100.0,  # tau_ih (constant in fernandez_cell.py's _tau_ih)
            15.0,   # tau_nd (constant in fernandez_cell.py's _tau_nd)
        )

    def get_voltage(self) -> np.ndarray:
        return self.V_s

    def get_dendritic_voltage(self) -> np.ndarray:
        return self.V_d

    def get_slow_k_current(self) -> np.ndarray:
        p = self._p
        return p.g_Kd_slow * self.n_d * (self.V_d - p.E_K)
