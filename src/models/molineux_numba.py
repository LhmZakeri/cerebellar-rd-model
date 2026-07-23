"""
Numba-accelerated batch kernel for the Molineux et al. (2005) cerebellar
stellate cell -- the model CONTEXT.md's "ReducedHHCellModel" maps to (not
the later Mitry et al. 2020 revision, which stays Python-only).

Operates on arrays of N nodes at once (RK4, same as
MolineuxStellateCellModel in molineux_cell.py). Parameters are read once
from that module's own MolineuxStellateParams dataclass -- no duplicated
numbers. See DESIGN.md ("Performance strategy: Numba") and
src/models/dangelo_numba.py / fernandez_numba.py for the same pattern
applied to the other two models.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from src.models.molineux_cell import MolineuxStellateCellModel, MolineuxStellateParams

_NA_TO_PA = 1000.0


@njit(cache=True)
def _sigmoid_inf(V, V_half, k):
    return 1.0 / (1.0 + np.exp((V - V_half) / (-k)))


@njit(cache=True)
def _tau_h_fn(V, y0, A_th, w_th, V_c):
    return y0 + (2.0 * A_th * w_th) / (4.0 * np.pi * (V - V_c) ** 2 + w_th ** 2)


@njit(cache=True)
def _rhs(V, h, n, h_T, h_A, I_E,
         C, g_Na, E_Na, v_m, s_m, v_h, s_h, y0, A_th, w_th, V_c,
         g_K, E_K, v_n, s_n, tau_n,
         g_leak, E_leak,
         g_A, v_nA, s_nA, v_hA, s_hA, tau_hA,
         g_T, E_Ca, v_mT, s_mT, v_hT, s_hT, tau_hT):
    m_inf = _sigmoid_inf(V, v_m, s_m)
    h_inf = _sigmoid_inf(V, v_h, s_h)
    n_inf = _sigmoid_inf(V, v_n, s_n)
    nA_inf = _sigmoid_inf(V, v_nA, s_nA)
    hA_inf = _sigmoid_inf(V, v_hA, s_hA)
    mT_inf = _sigmoid_inf(V, v_mT, s_mT)
    hT_inf = _sigmoid_inf(V, v_hT, s_hT)
    tau_h = _tau_h_fn(V, y0, A_th, w_th, V_c)

    I_Na = g_Na * m_inf * h * (V - E_Na)
    I_K = g_K * n * (V - E_K)
    I_leak = g_leak * (V - E_leak)
    I_A = g_A * nA_inf * h_A * (V - E_K)
    I_T = g_T * mT_inf * h_T * (V - E_Ca)

    dV = (I_E - I_Na - I_K - I_leak - I_A - I_T) / C
    dh = (h_inf - h) / tau_h
    dn = (n_inf - n) / tau_n
    dhT = (hT_inf - h_T) / tau_hT
    dhA = (hA_inf - h_A) / tau_hA
    return dV, dh, dn, dhT, dhA


@njit(parallel=True, cache=True, fastmath=False)
def _step_all(V, h, n, h_T, h_A, I_ext, dt,
              C, g_Na, E_Na, v_m, s_m, v_h, s_h, y0, A_th, w_th, V_c,
              g_K, E_K, v_n, s_n, tau_n,
              g_leak, E_leak,
              g_A, v_nA, s_nA, v_hA, s_hA, tau_hA,
              g_T, E_Ca, v_mT, s_mT, v_hT, s_hT, tau_hT):
    """Advance every node by one dt via RK4. Mutates all state arrays in place."""
    nn = V.shape[0]
    for i in prange(nn):
        v, hh, nnn, hT, hA = V[i], h[i], n[i], h_T[i], h_A[i]
        Ie = I_ext[i]

        k1 = _rhs(v, hh, nnn, hT, hA, Ie,
                  C, g_Na, E_Na, v_m, s_m, v_h, s_h, y0, A_th, w_th, V_c,
                  g_K, E_K, v_n, s_n, tau_n, g_leak, E_leak,
                  g_A, v_nA, s_nA, v_hA, s_hA, tau_hA,
                  g_T, E_Ca, v_mT, s_mT, v_hT, s_hT, tau_hT)
        k2 = _rhs(v + 0.5 * dt * k1[0], hh + 0.5 * dt * k1[1], nnn + 0.5 * dt * k1[2],
                  hT + 0.5 * dt * k1[3], hA + 0.5 * dt * k1[4], Ie,
                  C, g_Na, E_Na, v_m, s_m, v_h, s_h, y0, A_th, w_th, V_c,
                  g_K, E_K, v_n, s_n, tau_n, g_leak, E_leak,
                  g_A, v_nA, s_nA, v_hA, s_hA, tau_hA,
                  g_T, E_Ca, v_mT, s_mT, v_hT, s_hT, tau_hT)
        k3 = _rhs(v + 0.5 * dt * k2[0], hh + 0.5 * dt * k2[1], nnn + 0.5 * dt * k2[2],
                  hT + 0.5 * dt * k2[3], hA + 0.5 * dt * k2[4], Ie,
                  C, g_Na, E_Na, v_m, s_m, v_h, s_h, y0, A_th, w_th, V_c,
                  g_K, E_K, v_n, s_n, tau_n, g_leak, E_leak,
                  g_A, v_nA, s_nA, v_hA, s_hA, tau_hA,
                  g_T, E_Ca, v_mT, s_mT, v_hT, s_hT, tau_hT)
        k4 = _rhs(v + dt * k3[0], hh + dt * k3[1], nnn + dt * k3[2],
                  hT + dt * k3[3], hA + dt * k3[4], Ie,
                  C, g_Na, E_Na, v_m, s_m, v_h, s_h, y0, A_th, w_th, V_c,
                  g_K, E_K, v_n, s_n, tau_n, g_leak, E_leak,
                  g_A, v_nA, s_nA, v_hA, s_hA, tau_hA,
                  g_T, E_Ca, v_mT, s_mT, v_hT, s_hT, tau_hT)

        V[i] = v + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        h[i] = hh + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        n[i] = nnn + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        h_T[i] = hT + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        h_A[i] = hA + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])


class MolineuxBatch:
    """N independent Molineux2005 stellate cells, stepped via one Numba kernel.

    Same math as MolineuxStellateCellModel (molineux_cell.py) -- this is
    not a reimplementation, it reads that module's own
    MolineuxStellateParams values directly. Use this when N is large
    enough that the per-node Numba speedup matters; use
    MolineuxStellateCellModel for single-node work.
    """

    def __init__(self, n_nodes: int, params: MolineuxStellateParams | None = None) -> None:
        self.n_nodes = n_nodes
        self._p = params or MolineuxStellateParams()
        self.reset()

    def reset(self) -> None:
        # Single scalar instantiation gives us the exact validated initial
        # state (molineux_cell.py's own _make_initial_state()); broadcast it.
        initial = MolineuxStellateCellModel(self._p)._state
        self.V = np.full(self.n_nodes, initial[0], dtype=np.float64)
        self.h = np.full(self.n_nodes, initial[1], dtype=np.float64)
        self.n = np.full(self.n_nodes, initial[2], dtype=np.float64)
        self.h_T = np.full(self.n_nodes, initial[3], dtype=np.float64)
        self.h_A = np.full(self.n_nodes, initial[4], dtype=np.float64)

    def step(self, dt: float, I_ext_nA) -> None:
        """Advance all nodes by dt [ms]. I_ext_nA: scalar or per-node array [nA]."""
        if np.isscalar(I_ext_nA):
            I_ext = np.full(self.n_nodes, I_ext_nA * _NA_TO_PA, dtype=np.float64)
        else:
            I_ext = np.asarray(I_ext_nA, dtype=np.float64) * _NA_TO_PA

        p = self._p
        _step_all(
            self.V, self.h, self.n, self.h_T, self.h_A, I_ext, dt,
            p.C, p.g_Na, p.E_Na, p.v_m, p.s_m, p.v_h, p.s_h,
            p.y0, p.A_th, p.w_th, p.V_c,
            p.g_K, p.E_K, p.v_n, p.s_n, p.tau_n,
            p.g_leak, p.E_leak,
            p.g_A, p.v_nA, p.s_nA, p.v_hA, p.s_hA, p.tau_hA,
            p.g_T, p.E_Ca, p.v_mT, p.s_mT, p.v_hT, p.s_hT, p.tau_hT,
        )

    def get_voltage(self) -> np.ndarray:
        return self.V
