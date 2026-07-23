"""
Numba-accelerated batch kernel for the D'Angelo et al. (2001) granule cell.

Operates on arrays of N nodes at once (the shape needed at 1.2M-node scale),
reusing the exact LUTs and parameters already built by dangelo_cell.py -- no
duplicated numerics, so there is no second implementation to drift out of
sync with the validated Python prototype. See DESIGN.md ("Performance
strategy: Numba") for why this replaced the earlier planned C++ port.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from src.models.dangelo_cell import (
    DAngelo2001CellModel,
    DAngelo2001Params,
    _LUTS_2001,
    _LUT_V_MIN,
    _LUT_INV_DV,
    _LUT_N,
)


@njit(parallel=True, cache=True, fastmath=False)
def _step_all(
    V, m_Na, h_Na, s_Nar, f_Nar, m_pNa, n_KV, n_KM, a_KA, b_KA,
    d_Kir, c_KCa, s_Ca, u_Ca, cai, I_ext, dt,
    m_inf_Na_t, m_tau_Na_t, h_inf_Na_t, h_tau_Na_t,
    s_inf_Nar_t, s_tau_Nar_t, f_inf_Nar_t, f_tau_Nar_t,
    m_inf_pNa_t, m_tau_pNa_t, n_inf_KV_t, n_tau_KV_t,
    n_inf_KM_t, n_tau_KM_t, a_inf_KA_t, a_tau_KA_t,
    b_inf_KA_t, b_tau_KA_t, d_inf_Kir_t, d_tau_Kir_t,
    s_inf_Ca_t, s_tau_Ca_t, u_inf_Ca_t, u_tau_Ca_t,
    lut_vmin, lut_invdv, lut_n,
    gnabar_Na, gnabar_Nar, gnabar_pNa, gkbar_KV, gkbar_KM, gkbar_KA,
    gkbar_Kir, gkbar_KCa, gcabar_Ca, gl_Lkg1_arr, ggaba_Lkg2_arr,
    ena, ek, eca, el_Lkg1, egaba_Lkg2,
    Aalpha_c_KCa, Balpha_c_KCa, Kalpha_c_KCa,
    Abeta_c_KCa, Bbeta_c_KCa, Kbeta_c_KCa,
    ca_factor, ca_d, ca_beta, cai0, cm, area_arr, i_ext_scale,
):
    """Advance every node by one dt. Mutates all state arrays in place."""
    n = V.shape[0]
    for i in prange(n):
        gl_Lkg1 = gl_Lkg1_arr[i]
        ggaba_Lkg2 = ggaba_Lkg2_arr[i]
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

        m_inf = m_inf_Na_t[idx] + frac * (m_inf_Na_t[i1] - m_inf_Na_t[idx])
        m_tau = m_tau_Na_t[idx] + frac * (m_tau_Na_t[i1] - m_tau_Na_t[idx])
        h_inf = h_inf_Na_t[idx] + frac * (h_inf_Na_t[i1] - h_inf_Na_t[idx])
        h_tau = h_tau_Na_t[idx] + frac * (h_tau_Na_t[i1] - h_tau_Na_t[idx])
        s_inf = s_inf_Nar_t[idx] + frac * (s_inf_Nar_t[i1] - s_inf_Nar_t[idx])
        s_tau = s_tau_Nar_t[idx] + frac * (s_tau_Nar_t[i1] - s_tau_Nar_t[idx])
        f_inf = f_inf_Nar_t[idx] + frac * (f_inf_Nar_t[i1] - f_inf_Nar_t[idx])
        f_tau = f_tau_Nar_t[idx] + frac * (f_tau_Nar_t[i1] - f_tau_Nar_t[idx])
        mp_inf = m_inf_pNa_t[idx] + frac * (m_inf_pNa_t[i1] - m_inf_pNa_t[idx])
        mp_tau = m_tau_pNa_t[idx] + frac * (m_tau_pNa_t[i1] - m_tau_pNa_t[idx])
        nv_inf = n_inf_KV_t[idx] + frac * (n_inf_KV_t[i1] - n_inf_KV_t[idx])
        nv_tau = n_tau_KV_t[idx] + frac * (n_tau_KV_t[i1] - n_tau_KV_t[idx])
        nm_inf = n_inf_KM_t[idx] + frac * (n_inf_KM_t[i1] - n_inf_KM_t[idx])
        nm_tau = n_tau_KM_t[idx] + frac * (n_tau_KM_t[i1] - n_tau_KM_t[idx])
        a_inf = a_inf_KA_t[idx] + frac * (a_inf_KA_t[i1] - a_inf_KA_t[idx])
        a_tau = a_tau_KA_t[idx] + frac * (a_tau_KA_t[i1] - a_tau_KA_t[idx])
        b_inf = b_inf_KA_t[idx] + frac * (b_inf_KA_t[i1] - b_inf_KA_t[idx])
        b_tau = b_tau_KA_t[idx] + frac * (b_tau_KA_t[i1] - b_tau_KA_t[idx])
        d_inf = d_inf_Kir_t[idx] + frac * (d_inf_Kir_t[i1] - d_inf_Kir_t[idx])
        d_tau = d_tau_Kir_t[idx] + frac * (d_tau_Kir_t[i1] - d_tau_Kir_t[idx])
        sc_inf = s_inf_Ca_t[idx] + frac * (s_inf_Ca_t[i1] - s_inf_Ca_t[idx])
        sc_tau = s_tau_Ca_t[idx] + frac * (s_tau_Ca_t[i1] - s_tau_Ca_t[idx])
        u_inf = u_inf_Ca_t[idx] + frac * (u_inf_Ca_t[i1] - u_inf_Ca_t[idx])
        u_tau = u_tau_Ca_t[idx] + frac * (u_tau_Ca_t[i1] - u_tau_Ca_t[idx])

        cai_safe = cai[i] if cai[i] > 1e-10 else 1e-10
        alpha_c = Aalpha_c_KCa / (1.0 + Balpha_c_KCa * np.exp(v / Kalpha_c_KCa) / cai_safe)
        beta_c = Abeta_c_KCa / (1.0 + cai_safe / (Bbeta_c_KCa * np.exp(v / Kbeta_c_KCa)))
        c_inf = alpha_c / (alpha_c + beta_c)
        tau_c = 1.0 / (alpha_c + beta_c)

        g_Na = gnabar_Na * m_Na[i] * m_Na[i] * m_Na[i] * h_Na[i]
        g_Nar = gnabar_Nar * s_Nar[i] * f_Nar[i]
        g_pNa = gnabar_pNa * m_pNa[i]
        g_KV = gkbar_KV * n_KV[i] * n_KV[i] * n_KV[i] * n_KV[i]
        g_KM = gkbar_KM * n_KM[i]
        g_KA = gkbar_KA * a_KA[i] * a_KA[i] * a_KA[i] * b_KA[i]
        g_Kir = gkbar_Kir * d_Kir[i]
        g_KCa = gkbar_KCa * c_KCa[i]
        g_Ca = gcabar_Ca * s_Ca[i] * s_Ca[i] * u_Ca[i]

        g_tot = (g_Na + g_Nar + g_pNa + g_KV + g_KM + g_KA + g_Kir +
                 g_KCa + g_Ca + gl_Lkg1 + ggaba_Lkg2)
        I_gE = (g_Na * ena + g_Nar * ena + g_pNa * ena + g_KV * ek +
                g_KM * ek + g_KA * ek + g_Kir * ek + g_KCa * ek +
                g_Ca * eca + gl_Lkg1 * el_Lkg1 + ggaba_Lkg2 * egaba_Lkg2)

        I_ext_mA = I_ext[i] * i_ext_scale / area
        v_inf = (I_ext_mA + I_gE) / g_tot
        v_new = v_inf + (v - v_inf) * np.exp(-g_tot * dt / cm)

        ica_at_vnew = g_Ca * (v_new - eca)
        ca_source = -ica_at_vnew * 1e4 * ca_factor / ca_d
        cai_new = (cai[i] + dt * (ca_source + ca_beta * cai0)) / (1.0 + dt * ca_beta)
        if cai_new < 1e-10:
            cai_new = 1e-10

        m_Na[i] = m_inf + (m_Na[i] - m_inf) * np.exp(-dt / m_tau)
        h_Na[i] = h_inf + (h_Na[i] - h_inf) * np.exp(-dt / h_tau)
        s_Nar[i] = s_inf + (s_Nar[i] - s_inf) * np.exp(-dt / s_tau)
        f_Nar[i] = f_inf + (f_Nar[i] - f_inf) * np.exp(-dt / f_tau)
        m_pNa[i] = mp_inf + (m_pNa[i] - mp_inf) * np.exp(-dt / mp_tau)
        n_KV[i] = nv_inf + (n_KV[i] - nv_inf) * np.exp(-dt / nv_tau)
        n_KM[i] = nm_inf + (n_KM[i] - nm_inf) * np.exp(-dt / nm_tau)
        a_KA[i] = a_inf + (a_KA[i] - a_inf) * np.exp(-dt / a_tau)
        b_KA[i] = b_inf + (b_KA[i] - b_inf) * np.exp(-dt / b_tau)
        d_Kir[i] = d_inf + (d_Kir[i] - d_inf) * np.exp(-dt / d_tau)
        c_KCa[i] = c_inf + (c_KCa[i] - c_inf) * np.exp(-dt / tau_c)
        s_Ca[i] = sc_inf + (s_Ca[i] - sc_inf) * np.exp(-dt / sc_tau)
        u_Ca[i] = u_inf + (u_Ca[i] - u_inf) * np.exp(-dt / u_tau)

        V[i] = v_new
        cai[i] = cai_new


class DAngeloBatch:
    """N independent D'Angelo granule cells, stepped together via one Numba kernel.

    Same math as DAngelo2001CellModel (dangelo_cell.py) -- this is not a
    reimplementation, it consumes that module's own LUTs and parameters
    directly. Use this when N is large enough that the per-node Numba
    speedup matters; use DAngelo2001CellModel for single-node work.
    """

    _STATE_NAMES = (
        "V", "m_Na", "h_Na", "s_Nar", "f_Nar", "m_pNa", "n_KV", "n_KM",
        "a_KA", "b_KA", "d_Kir", "c_KCa", "s_Ca", "u_Ca", "cai",
    )

    def __init__(
        self,
        n_nodes: int,
        params: DAngelo2001Params | None = None,
        heterogeneity_seed: int | None = None,
    ) -> None:
        """heterogeneity_seed: None (default) -> every node identical, exact
        prior behavior. An int -> Sou11-style per-node heterogeneity (DESIGN.md): gl_Lkg1, ggaba_Lkg2, membrane area, and initial V each
        independently drawn uniform +/-20% around their base value, one
        draw per node. Required, explicit (no silent default) at the
        experiment-script level -- same reproducibility rationale as
        place_golgi_cells/sample_uniform_positions elsewhere in this repo."""
        self.n_nodes = n_nodes
        self._p = params or DAngelo2001Params()
        self._Cm = self._p.cm_spec * 1e-3
        base_area = np.pi * self._p.diam * 1e-4 * self._p.L * 1e-4
        self._heterogeneity_seed = heterogeneity_seed
        if heterogeneity_seed is None:
            self._area_cm2 = np.full(n_nodes, base_area, dtype=np.float64)
            self._gl_Lkg1 = np.full(n_nodes, self._p.gl_Lkg1, dtype=np.float64)
            self._ggaba_Lkg2 = np.full(n_nodes, self._p.ggaba_Lkg2, dtype=np.float64)
        else:
            rng = np.random.default_rng(heterogeneity_seed)
            self._area_cm2 = base_area * rng.uniform(0.8, 1.2, size=n_nodes)
            self._gl_Lkg1 = self._p.gl_Lkg1 * rng.uniform(0.8, 1.2, size=n_nodes)
            self._ggaba_Lkg2 = self._p.ggaba_Lkg2 * rng.uniform(0.8, 1.2, size=n_nodes)
        self._lut = {k: np.asarray(v, dtype=np.float64) for k, v in _LUTS_2001.items()}
        self.reset()

    def reset(self) -> None:
        # Single scalar instantiation gives us the exact validated initial
        # state (dangelo_cell.py's own _make_initial_state()); broadcast it
        # to every node, EXCEPT V, which gets its own independent +/-20%
        # draw per node when heterogeneity_seed is set (DESIGN.md) --
        # only the initial membrane potential is randomized per Sou11, not
        # the gating-variable initial conditions.
        initial = DAngelo2001CellModel(self._p)._state
        for name, value in zip(self._STATE_NAMES, initial):
            setattr(self, name, np.full(self.n_nodes, value, dtype=np.float64))
        if self._heterogeneity_seed is not None:
            # Offset from the constructor's rng stream (which drew 3 arrays
            # of size n_nodes already) so reset() calls after construction
            # reproduce the identical initial-V draw every time, independent
            # of how many times reset() itself is called.
            rng = np.random.default_rng(self._heterogeneity_seed + 1)
            self.V = self.V * rng.uniform(0.8, 1.2, size=self.n_nodes)

    def step(self, dt: float, I_ext_nA) -> None:
        """Advance all nodes by dt [ms]. I_ext_nA: scalar or per-node array [nA]."""
        if np.isscalar(I_ext_nA):
            I_ext = np.full(self.n_nodes, I_ext_nA, dtype=np.float64)
        else:
            I_ext = np.asarray(I_ext_nA, dtype=np.float64)

        L = self._lut
        p = self._p
        _step_all(
            self.V, self.m_Na, self.h_Na, self.s_Nar, self.f_Nar, self.m_pNa,
            self.n_KV, self.n_KM, self.a_KA, self.b_KA, self.d_Kir, self.c_KCa,
            self.s_Ca, self.u_Ca, self.cai, I_ext, dt,
            L["m_inf_Na"], L["m_tau_Na"], L["h_inf_Na"], L["h_tau_Na"],
            L["s_inf_Nar"], L["s_tau_Nar"], L["f_inf_Nar"], L["f_tau_Nar"],
            L["m_inf_pNa"], L["m_tau_pNa"], L["n_inf_KV"], L["n_tau_KV"],
            L["n_inf_KM"], L["n_tau_KM"], L["a_inf_KA"], L["a_tau_KA"],
            L["b_inf_KA"], L["b_tau_KA"], L["d_inf_Kir"], L["d_tau_Kir"],
            L["s_inf_Ca"], L["s_tau_Ca"], L["u_inf_Ca"], L["u_tau_Ca"],
            _LUT_V_MIN, _LUT_INV_DV, _LUT_N,
            p.gnabar_Na, p.gnabar_Nar, p.gnabar_pNa, p.gkbar_KV, p.gkbar_KM,
            p.gkbar_KA, p.gkbar_Kir, p.gkbar_KCa, p.gcabar_Ca, self._gl_Lkg1,
            self._ggaba_Lkg2, p.ena, p.ek, p.eca, p.el_Lkg1, p.egaba_Lkg2,
            p.Aalpha_c_KCa, p.Balpha_c_KCa, p.Kalpha_c_KCa,
            p.Abeta_c_KCa, p.Bbeta_c_KCa, p.Kbeta_c_KCa,
            p.ca_factor, p.ca_d, p.ca_beta, p.cai0,
            self._Cm, self._area_cm2, 1e-6,
        )

    def get_voltage(self) -> np.ndarray:
        return self.V

    def get_calcium(self) -> np.ndarray:
        return self.cai
