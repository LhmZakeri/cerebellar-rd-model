"""
Numba-accelerated batch kernel for the Solinas et al. (2007) Golgi cell.

Operates on arrays of N nodes at once, reusing the exact LUTs and parameters
already built by solinas_cell.py -- no duplicated numerics, so there is no
second implementation to drift out of sync with the validated Python
prototype. Same pattern as dangelo_numba.py (LUT + Rush-Larsen); see
DESIGN.md ("Performance strategy: Numba") for why this replaced the earlier
planned C++ port.

One wrinkle vs. the other three *Batch kernels: Golgi_SK2 is a 6-state
Markov chain (c1/c2/c3/c4/o1/o2), not a simple two-state Rush-Larsen gate.
solinas_cell.py integrates it with a "diagonal-exact per state + renormalise"
scheme (each state's ODE is treated as dx/dt = src - lam*x with sources
frozen at step start, giving an exact exponential-decay-to-equilibrium
solution per state, then rescaled if the frozen-source approximation ever
pushes the five tracked states' sum above 1). This kernel replicates that
scheme exactly, not the standard two-state Rush-Larsen pattern.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from src.models.solinas_cell import (
    Solinas2007CellModel,
    Solinas2007Params,
    _LUTS,
    _LUT_V_MIN,
    _LUT_INV_DV,
    _LUT_N,
    _Q10_30,
    _Q10_23,
    _NERNST_CA,
    _CA_FACTOR,
)


@njit(cache=True)
def _rl(x: float, x_inf: float, tau: float, dt: float) -> float:
    """Rush-Larsen exact update -- mirrors solinas_cell.py's `_rl` exactly,
    including its two guards (near-zero tau, and the -700 exponent clamp to
    avoid overflow for very large dt/tau)."""
    if tau < 1e-9:
        return x_inf
    exponent = -dt / tau
    if exponent < -700.0:
        exponent = -700.0
    return x_inf + (x - x_inf) * math.exp(exponent)


@njit(parallel=True, cache=True, fastmath=False)
def _step_all(
    V, m_Na, h_Na, s_NaR, f_NaR, m_NaP, n_KV, a_KA, b_KA, n_KM, c_BK,
    c2_SK2, c3_SK2, c4_SK2, o1_SK2, o2_SK2,
    s_HVA, u_HVA, m_LVA, h_LVA, of_hcn1, os_hcn1, of_hcn2, os_hcn2,
    cai, ca2i, I_ext, dt,
    m_inf_Na_t, m_tau_Na_t, h_inf_Na_t, h_tau_Na_t,
    s_inf_NaR_t, s_tau_NaR_t, f_inf_NaR_t, f_tau_NaR_t,
    m_inf_NaP_t, m_tau_NaP_t,
    n_inf_KV_t, n_tau_KV_t,
    a_inf_KA_t, a_tau_KA_t, b_inf_KA_t, b_tau_KA_t,
    n_inf_KM_t, n_tau_KM_t,
    s_inf_HVA_t, s_tau_HVA_t, u_inf_HVA_t, u_tau_HVA_t,
    m_inf_LVA_t, m_tau_LVA_t, h_inf_LVA_t, h_tau_LVA_t,
    of_inf_hcn1_t, of_tau_hcn1_t, os_inf_hcn1_t, os_tau_hcn1_t,
    of_inf_hcn2_t, of_tau_hcn2_t, os_inf_hcn2_t, os_tau_hcn2_t,
    lut_vmin, lut_invdv, lut_n,
    gnabar_Na, gnabar_NaR, gnabar_NaP, gkbar_KV, gkbar_KA, gkbar_KM,
    gkbar_BK, gkbar_SK2, gcabar_HVA, gca2bar_LVA, gbar_hcn1, gbar_hcn2,
    glbar_lkg_arr,
    ena, ek, eca, el, Erev_hcn1, Erev_hcn2,
    Aalpha_c_BK, Balpha_c_BK, Kalpha_c_BK, Abeta_c_BK, Bbeta_c_BK, Kbeta_c_BK,
    invc1_SK2, invc2_SK2, invc3_SK2, invo1_SK2, invo2_SK2,
    diro1_SK2, diro2_SK2, dirc2_SK2, dirc3_SK2, dirc4_SK2, diff_SK2,
    ca_d, ca_beta, cai0, cao,
    q10_30, q10_23, nernst_ca, ca_factor,
    cm, area_arr, i_ext_scale,
):
    """Advance every node by one dt. Mutates all state arrays in place."""
    n = V.shape[0]
    for i in prange(n):
        glbar_lkg = glbar_lkg_arr[i]
        area = area_arr[i]
        v = V[i]
        idxf = (v - lut_vmin) * lut_invdv
        idx = int(idxf)
        if idx < 0:
            idx = 0
        elif idx >= lut_n - 1:
            idx = lut_n - 2
        frac = idxf - idx
        i1 = idx + 1

        m_inf_Na = m_inf_Na_t[idx] + frac * (m_inf_Na_t[i1] - m_inf_Na_t[idx])
        m_tau_Na = m_tau_Na_t[idx] + frac * (m_tau_Na_t[i1] - m_tau_Na_t[idx])
        h_inf_Na = h_inf_Na_t[idx] + frac * (h_inf_Na_t[i1] - h_inf_Na_t[idx])
        h_tau_Na = h_tau_Na_t[idx] + frac * (h_tau_Na_t[i1] - h_tau_Na_t[idx])
        s_inf_NaR = s_inf_NaR_t[idx] + frac * (s_inf_NaR_t[i1] - s_inf_NaR_t[idx])
        s_tau_NaR = s_tau_NaR_t[idx] + frac * (s_tau_NaR_t[i1] - s_tau_NaR_t[idx])
        f_inf_NaR = f_inf_NaR_t[idx] + frac * (f_inf_NaR_t[i1] - f_inf_NaR_t[idx])
        f_tau_NaR = f_tau_NaR_t[idx] + frac * (f_tau_NaR_t[i1] - f_tau_NaR_t[idx])
        m_inf_NaP = m_inf_NaP_t[idx] + frac * (m_inf_NaP_t[i1] - m_inf_NaP_t[idx])
        m_tau_NaP = m_tau_NaP_t[idx] + frac * (m_tau_NaP_t[i1] - m_tau_NaP_t[idx])
        n_inf_KV = n_inf_KV_t[idx] + frac * (n_inf_KV_t[i1] - n_inf_KV_t[idx])
        n_tau_KV = n_tau_KV_t[idx] + frac * (n_tau_KV_t[i1] - n_tau_KV_t[idx])
        a_inf_KA = a_inf_KA_t[idx] + frac * (a_inf_KA_t[i1] - a_inf_KA_t[idx])
        a_tau_KA = a_tau_KA_t[idx] + frac * (a_tau_KA_t[i1] - a_tau_KA_t[idx])
        b_inf_KA = b_inf_KA_t[idx] + frac * (b_inf_KA_t[i1] - b_inf_KA_t[idx])
        b_tau_KA = b_tau_KA_t[idx] + frac * (b_tau_KA_t[i1] - b_tau_KA_t[idx])
        n_inf_KM = n_inf_KM_t[idx] + frac * (n_inf_KM_t[i1] - n_inf_KM_t[idx])
        n_tau_KM = n_tau_KM_t[idx] + frac * (n_tau_KM_t[i1] - n_tau_KM_t[idx])
        s_inf_HVA = s_inf_HVA_t[idx] + frac * (s_inf_HVA_t[i1] - s_inf_HVA_t[idx])
        s_tau_HVA = s_tau_HVA_t[idx] + frac * (s_tau_HVA_t[i1] - s_tau_HVA_t[idx])
        u_inf_HVA = u_inf_HVA_t[idx] + frac * (u_inf_HVA_t[i1] - u_inf_HVA_t[idx])
        u_tau_HVA = u_tau_HVA_t[idx] + frac * (u_tau_HVA_t[i1] - u_tau_HVA_t[idx])
        m_inf_LVA = m_inf_LVA_t[idx] + frac * (m_inf_LVA_t[i1] - m_inf_LVA_t[idx])
        m_tau_LVA = m_tau_LVA_t[idx] + frac * (m_tau_LVA_t[i1] - m_tau_LVA_t[idx])
        h_inf_LVA = h_inf_LVA_t[idx] + frac * (h_inf_LVA_t[i1] - h_inf_LVA_t[idx])
        h_tau_LVA = h_tau_LVA_t[idx] + frac * (h_tau_LVA_t[i1] - h_tau_LVA_t[idx])
        of_inf_hcn1 = of_inf_hcn1_t[idx] + frac * (of_inf_hcn1_t[i1] - of_inf_hcn1_t[idx])
        of_tau_hcn1 = of_tau_hcn1_t[idx] + frac * (of_tau_hcn1_t[i1] - of_tau_hcn1_t[idx])
        os_inf_hcn1 = os_inf_hcn1_t[idx] + frac * (os_inf_hcn1_t[i1] - os_inf_hcn1_t[idx])
        os_tau_hcn1 = os_tau_hcn1_t[idx] + frac * (os_tau_hcn1_t[i1] - os_tau_hcn1_t[idx])
        of_inf_hcn2 = of_inf_hcn2_t[idx] + frac * (of_inf_hcn2_t[i1] - of_inf_hcn2_t[idx])
        of_tau_hcn2 = of_tau_hcn2_t[idx] + frac * (of_tau_hcn2_t[i1] - of_tau_hcn2_t[idx])
        os_inf_hcn2 = os_inf_hcn2_t[idx] + frac * (os_inf_hcn2_t[i1] - os_inf_hcn2_t[idx])
        os_tau_hcn2 = os_tau_hcn2_t[idx] + frac * (os_tau_hcn2_t[i1] - os_tau_hcn2_t[idx])

        # --- BK: Ca- and V-dependent (computed per step, not LUT'd) --------
        cai_safe = cai[i] if cai[i] > 1e-10 else 1e-10
        alp_c_BK = q10_30 * Aalpha_c_BK / (
            1.0 + Balpha_c_BK * math.exp(v / Kalpha_c_BK) / cai_safe
        )
        bet_c_BK = q10_30 * Abeta_c_BK / (
            1.0 + cai_safe / (Bbeta_c_BK * math.exp(v / Kbeta_c_BK))
        )
        c_inf_BK = alp_c_BK / (alp_c_BK + bet_c_BK)
        tau_c_BK = 1.0 / (alp_c_BK + bet_c_BK)

        # --- SK2: 6-state Markov chain (diagonal-exact per state + renormalise) ---
        tcorr_SK2 = q10_23
        invc1_t = invc1_SK2 * tcorr_SK2
        invc2_t = invc2_SK2 * tcorr_SK2
        invc3_t = invc3_SK2 * tcorr_SK2
        invo1_t = invo1_SK2 * tcorr_SK2
        invo2_t = invo2_SK2 * tcorr_SK2
        diro1_t = diro1_SK2 * tcorr_SK2
        diro2_t = diro2_SK2 * tcorr_SK2
        dirc2_t = dirc2_SK2 * cai_safe * tcorr_SK2 / diff_SK2
        dirc3_t = dirc3_SK2 * cai_safe * tcorr_SK2 / diff_SK2
        dirc4_t = dirc4_SK2 * cai_safe * tcorr_SK2 / diff_SK2

        c2, c3, c4, o1, o2 = c2_SK2[i], c3_SK2[i], c4_SK2[i], o1_SK2[i], o2_SK2[i]
        c1 = 1.0 - c2 - c3 - c4 - o1 - o2
        if c1 < 0.0:
            c1 = 0.0

        lam2 = invc1_t + dirc3_t
        eq2 = (dirc2_t * c1 + invc2_t * c3) / lam2
        exp2 = -lam2 * dt
        if exp2 < -700.0:
            exp2 = -700.0
        c2_new = eq2 + (c2 - eq2) * math.exp(exp2)

        lam3 = invc2_t + dirc4_t + diro1_t
        eq3 = (dirc3_t * c2 + invc3_t * c4 + invo1_t * o1) / lam3
        exp3 = -lam3 * dt
        if exp3 < -700.0:
            exp3 = -700.0
        c3_new = eq3 + (c3 - eq3) * math.exp(exp3)

        lam4 = invc3_t + diro2_t
        eq4 = (dirc4_t * c3 + invo2_t * o2) / lam4
        exp4 = -lam4 * dt
        if exp4 < -700.0:
            exp4 = -700.0
        c4_new = eq4 + (c4 - eq4) * math.exp(exp4)

        eq_o1 = diro1_t * c3 / invo1_t
        expo1 = -invo1_t * dt
        if expo1 < -700.0:
            expo1 = -700.0
        o1_new = eq_o1 + (o1 - eq_o1) * math.exp(expo1)

        eq_o2 = diro2_t * c4 / invo2_t
        expo2 = -invo2_t * dt
        if expo2 < -700.0:
            expo2 = -700.0
        o2_new = eq_o2 + (o2 - eq_o2) * math.exp(expo2)

        sk2_sum = c2_new + c3_new + c4_new + o1_new + o2_new
        if sk2_sum > 1.0:
            inv_sum = 1.0 / sk2_sum
            c2_new *= inv_sum
            c3_new *= inv_sum
            c4_new *= inv_sum
            o1_new *= inv_sum
            o2_new *= inv_sum

        # --- LVA Ca reversal potential (Nernst, uses previous ca2i) --------
        ca2i_safe = ca2i[i] if ca2i[i] > 1e-10 else 1e-10
        eca2 = nernst_ca * math.log(cao / ca2i_safe)

        # --- Conductances ----------------------------------------------------
        g_Na = gnabar_Na * m_Na[i] * m_Na[i] * m_Na[i] * h_Na[i]
        g_NaR = gnabar_NaR * s_NaR[i] * f_NaR[i]
        g_NaP = gnabar_NaP * m_NaP[i]
        g_KV = gkbar_KV * n_KV[i] * n_KV[i] * n_KV[i] * n_KV[i]
        g_KA = gkbar_KA * a_KA[i] * a_KA[i] * a_KA[i] * b_KA[i]
        g_KM = gkbar_KM * n_KM[i]
        g_BK = gkbar_BK * c_BK[i]
        g_SK2 = gkbar_SK2 * (o1_SK2[i] + o2_SK2[i])
        g_HVA = gcabar_HVA * s_HVA[i] * s_HVA[i] * u_HVA[i]
        g_LVA = gca2bar_LVA * m_LVA[i] * m_LVA[i] * h_LVA[i]
        g_hcn1 = gbar_hcn1 * (of_hcn1[i] + os_hcn1[i])
        g_hcn2 = gbar_hcn2 * (of_hcn2[i] + os_hcn2[i])
        g_lkg = glbar_lkg

        # --- Voltage: Rush-Larsen via conductance sum -----------------------
        g_tot = (
            g_Na + g_NaR + g_NaP + g_KV + g_KA + g_KM + g_BK + g_SK2
            + g_HVA + g_LVA + g_hcn1 + g_hcn2 + g_lkg
        )
        I_gE = (
            g_Na * ena + g_NaR * ena + g_NaP * ena
            + g_KV * ek + g_KA * ek + g_KM * ek + g_BK * ek + g_SK2 * ek
            + g_HVA * eca + g_LVA * eca2
            + g_hcn1 * Erev_hcn1 + g_hcn2 * Erev_hcn2
            + g_lkg * el
        )
        I_ext_mA = I_ext[i] * i_ext_scale / area
        v_inf = (I_ext_mA + I_gE) / g_tot
        v_new = v_inf + (v - v_inf) * math.exp(-g_tot * dt / cm)

        # --- Ca2+: semi-implicit Backward Euler (HVA pool) ------------------
        ica_HVA = g_HVA * (v_new - eca)
        ca_src_HVA = -ica_HVA * ca_factor / ca_d
        cai_new = (cai[i] + dt * (ca_src_HVA + ca_beta * cai0)) / (1.0 + dt * ca_beta)
        if cai_new < 1e-10:
            cai_new = 1e-10

        # --- Ca2+: semi-implicit Backward Euler (LVA pool) ------------------
        ica_LVA = g_LVA * (v_new - eca2)
        ca_src_LVA = -ica_LVA * ca_factor / ca_d
        ca2i_new = (ca2i[i] + dt * (ca_src_LVA + ca_beta * cai0)) / (1.0 + dt * ca_beta)
        if ca2i_new < 1e-10:
            ca2i_new = 1e-10

        # --- Gate variables: Rush-Larsen -------------------------------------
        m_Na[i] = _rl(m_Na[i], m_inf_Na, m_tau_Na, dt)
        h_Na[i] = _rl(h_Na[i], h_inf_Na, h_tau_Na, dt)
        s_NaR[i] = _rl(s_NaR[i], s_inf_NaR, s_tau_NaR, dt)
        f_NaR[i] = _rl(f_NaR[i], f_inf_NaR, f_tau_NaR, dt)
        m_NaP[i] = _rl(m_NaP[i], m_inf_NaP, m_tau_NaP, dt)
        n_KV[i] = _rl(n_KV[i], n_inf_KV, n_tau_KV, dt)
        a_KA[i] = _rl(a_KA[i], a_inf_KA, a_tau_KA, dt)
        b_KA[i] = _rl(b_KA[i], b_inf_KA, b_tau_KA, dt)
        n_KM[i] = _rl(n_KM[i], n_inf_KM, n_tau_KM, dt)
        c_BK[i] = _rl(c_BK[i], c_inf_BK, tau_c_BK, dt)
        s_HVA[i] = _rl(s_HVA[i], s_inf_HVA, s_tau_HVA, dt)
        u_HVA[i] = _rl(u_HVA[i], u_inf_HVA, u_tau_HVA, dt)
        m_LVA[i] = _rl(m_LVA[i], m_inf_LVA, m_tau_LVA, dt)
        h_LVA[i] = _rl(h_LVA[i], h_inf_LVA, h_tau_LVA, dt)
        of_hcn1[i] = _rl(of_hcn1[i], of_inf_hcn1, of_tau_hcn1, dt)
        os_hcn1[i] = _rl(os_hcn1[i], os_inf_hcn1, os_tau_hcn1, dt)
        of_hcn2[i] = _rl(of_hcn2[i], of_inf_hcn2, of_tau_hcn2, dt)
        os_hcn2[i] = _rl(os_hcn2[i], os_inf_hcn2, os_tau_hcn2, dt)

        c2_SK2[i] = c2_new
        c3_SK2[i] = c3_new
        c4_SK2[i] = c4_new
        o1_SK2[i] = o1_new
        o2_SK2[i] = o2_new

        V[i] = v_new
        cai[i] = cai_new
        ca2i[i] = ca2i_new


class SolinasBatch:
    """N independent Solinas 2007 Golgi cells, stepped together via one
    Numba kernel.

    Same math as Solinas2007CellModel (solinas_cell.py) -- this is not a
    reimplementation, it consumes that module's own LUTs and parameters
    directly. Sparse-sized: construct with n_golgi (the number of
    Golgi-hosting nodes), not n_nodes -- see
    DESIGN.md for why Golgi state
    is not allocated one-per-node like the other three cell types.
    """

    _STATE_NAMES = (
        "V", "m_Na", "h_Na", "s_NaR", "f_NaR", "m_NaP", "n_KV", "a_KA",
        "b_KA", "n_KM", "c_BK", "c2_SK2", "c3_SK2", "c4_SK2", "o1_SK2",
        "o2_SK2", "s_HVA", "u_HVA", "m_LVA", "h_LVA", "of_hcn1", "os_hcn1",
        "of_hcn2", "os_hcn2", "cai", "ca2i",
    )

    def __init__(
        self,
        n_golgi: int,
        params: Solinas2007Params | None = None,
        heterogeneity_seed: int | None = None,
    ) -> None:
        """heterogeneity_seed: None (default) -> every Golgi cell identical,
        exact prior behavior. An int -> Sou11-style per-cell heterogeneity
        (DESIGN.md): glbar_lkg, membrane area, and initial V each
        independently drawn uniform +/-20% around their base value, one
        draw per cell."""
        self.n_golgi = n_golgi
        self._p = params or Solinas2007Params()
        self._Cm = self._p.cm_spec * 1e-3
        base_area = math.pi * self._p.diam * 1e-4 * self._p.L * 1e-4
        self._heterogeneity_seed = heterogeneity_seed
        if heterogeneity_seed is None:
            self._area_cm2 = np.full(n_golgi, base_area, dtype=np.float64)
            self._glbar_lkg = np.full(n_golgi, self._p.glbar_lkg, dtype=np.float64)
        else:
            rng = np.random.default_rng(heterogeneity_seed)
            self._area_cm2 = base_area * rng.uniform(0.8, 1.2, size=n_golgi)
            self._glbar_lkg = self._p.glbar_lkg * rng.uniform(0.8, 1.2, size=n_golgi)
        self._lut = {k: np.asarray(v, dtype=np.float64) for k, v in _LUTS.items()}
        self.reset()

    def reset(self) -> None:
        # Single scalar instantiation gives us the exact validated initial
        # state (solinas_cell.py's own _make_initial_state()); broadcast it
        # to every cell, EXCEPT V, which gets its own independent +/-20%
        # draw per cell when heterogeneity_seed is set (DESIGN.md).
        initial = Solinas2007CellModel(self._p)._state
        for name, value in zip(self._STATE_NAMES, initial):
            setattr(self, name, np.full(self.n_golgi, value, dtype=np.float64))
        if self._heterogeneity_seed is not None:
            rng = np.random.default_rng(self._heterogeneity_seed + 1)
            self.V = self.V * rng.uniform(0.8, 1.2, size=self.n_golgi)

    def step(self, dt: float, I_ext_nA) -> None:
        """Advance all Golgi cells by dt [ms]. I_ext_nA: scalar or per-cell array [nA]."""
        if np.isscalar(I_ext_nA):
            I_ext = np.full(self.n_golgi, I_ext_nA, dtype=np.float64)
        else:
            I_ext = np.asarray(I_ext_nA, dtype=np.float64)

        L = self._lut
        p = self._p
        _step_all(
            self.V, self.m_Na, self.h_Na, self.s_NaR, self.f_NaR, self.m_NaP,
            self.n_KV, self.a_KA, self.b_KA, self.n_KM, self.c_BK,
            self.c2_SK2, self.c3_SK2, self.c4_SK2, self.o1_SK2, self.o2_SK2,
            self.s_HVA, self.u_HVA, self.m_LVA, self.h_LVA,
            self.of_hcn1, self.os_hcn1, self.of_hcn2, self.os_hcn2,
            self.cai, self.ca2i, I_ext, dt,
            L["m_inf_Na"], L["m_tau_Na"], L["h_inf_Na"], L["h_tau_Na"],
            L["s_inf_NaR"], L["s_tau_NaR"], L["f_inf_NaR"], L["f_tau_NaR"],
            L["m_inf_NaP"], L["m_tau_NaP"],
            L["n_inf_KV"], L["n_tau_KV"],
            L["a_inf_KA"], L["a_tau_KA"], L["b_inf_KA"], L["b_tau_KA"],
            L["n_inf_KM"], L["n_tau_KM"],
            L["s_inf_HVA"], L["s_tau_HVA"], L["u_inf_HVA"], L["u_tau_HVA"],
            L["m_inf_LVA"], L["m_tau_LVA"], L["h_inf_LVA"], L["h_tau_LVA"],
            L["of_inf_hcn1"], L["of_tau_hcn1"], L["os_inf_hcn1"], L["os_tau_hcn1"],
            L["of_inf_hcn2"], L["of_tau_hcn2"], L["os_inf_hcn2"], L["os_tau_hcn2"],
            _LUT_V_MIN, _LUT_INV_DV, _LUT_N,
            p.gnabar_Na, p.gnabar_NaR, p.gnabar_NaP, p.gkbar_KV, p.gkbar_KA,
            p.gkbar_KM, p.gkbar_BK, p.gkbar_SK2, p.gcabar_HVA, p.gca2bar_LVA,
            p.gbar_hcn1, p.gbar_hcn2, self._glbar_lkg,
            p.ena, p.ek, p.eca, p.el, p.Erev_hcn1, p.Erev_hcn2,
            p.Aalpha_c_BK, p.Balpha_c_BK, p.Kalpha_c_BK,
            p.Abeta_c_BK, p.Bbeta_c_BK, p.Kbeta_c_BK,
            p.invc1_SK2, p.invc2_SK2, p.invc3_SK2, p.invo1_SK2, p.invo2_SK2,
            p.diro1_SK2, p.diro2_SK2, p.dirc2_SK2, p.dirc3_SK2, p.dirc4_SK2,
            p.diff_SK2,
            p.ca_d, p.ca_beta, p.cai0, p.cao,
            _Q10_30, _Q10_23, _NERNST_CA, _CA_FACTOR,
            self._Cm, self._area_cm2, 1e-6,
        )

    def get_voltage(self) -> np.ndarray:
        return self.V

    def get_calcium(self) -> np.ndarray:
        return self.cai
