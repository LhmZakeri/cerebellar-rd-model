"""
Solinas et al. (2007) cerebellar Golgi cell ionic model — Phase 1 Python prototype.

Single compartment (soma: diam = L = 27 um), 26 state variables.

References:
  Solinas et al. (2007) Front Neurosci — Golgi cell pacemaking and electroresponsiveness
  ModelDB: 112685

Channels (soma):
  Golgi_Na    fast transient Na      m^3 h       gnabar = 0.048
  Golgi_NaR   resurgent Na           s f         gnabar = 0.0017
  Golgi_NaP   persistent Na          m           gbar   = 0.00019
  Golgi_KV    delayed rectifier K    n^4         gkbar  = 0.032
  Golgi_KA    A-type K               a^3 b       gkbar  = 0.008
  Golgi_KM    M-current K            n           gkbar  = 0.001
  Golgi_BK    BK Ca-activated K      c           gkbar  = 0.003
  Golgi_SK2   SK2 Ca-activated K     6-state     gkbar  = 0.038
  Golgi_Ca_HVA HVA Ca               s^2 u       gcabar = 460e-6
  Golgi_Ca_LVA LVA Ca               m^2 h       gcabar = 2.5e-4
  Golgi_hcn1  HCN1                   of os       gbar   = 5e-5
  Golgi_hcn2  HCN2                   of os       gbar   = 8e-5
  Golgi_lkg   leak                               glbar  = 21e-6

Ca dynamics: two pools
  cai  (HVA pool) — read by BK and SK2
  ca2i (LVA pool) — used for LVA reversal potential only
"""

import math
from dataclasses import dataclass

from src.models.cell_model import CellModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linoid(x: float, y: float) -> float:
    """x / (exp(x/y) - 1)  — standard linoid used by most channels."""
    ratio = x / y
    if abs(ratio) < 1e-6:
        return y * (1.0 - ratio / 2.0)
    if ratio > 500.0:
        return 0.0
    if ratio < -500.0:
        return -x
    return x / (math.exp(ratio) - 1.0)


def _linoid_na(x: float, y: float) -> float:
    """x / (1 - exp(x/y)) — Golgi_Na linoid (sign-flipped vs standard)."""
    ratio = x / y
    if abs(ratio) < 1e-6:
        return y * (1.0 - ratio / 2.0)
    if ratio > 500.0:
        return 0.0
    if ratio < -500.0:
        return x
    return x / (1.0 - math.exp(ratio))


def _sigm(x: float, y: float) -> float:
    """1 / (exp(x/y) + 1)"""
    return 1.0 / (math.exp(x / y) + 1.0)


# ---------------------------------------------------------------------------
# LUT constants
# ---------------------------------------------------------------------------
_LUT_V_MIN: float = -100.0
_LUT_V_MAX: float = 80.0
_LUT_DV: float = 0.1
_LUT_INV_DV: float = 10.0
_LUT_N: int = 1801

# ---------------------------------------------------------------------------
# Pre-computed Q10 factors at celsius = 30
# ---------------------------------------------------------------------------
_CELSIUS: float = 23.0
_Q10_20: float = 3.0 ** ((_CELSIUS - 20.0) / 10.0)  # = 3.0
_Q10_30: float = 3.0 ** ((_CELSIUS - 30.0) / 10.0)  # = 1.0 (NaP, BK)
_Q10_63: float = 3.0 ** ((_CELSIUS - 6.3) / 10.0)  # ≈ 13.97 (KV)
_Q10_255: float = 3.0 ** ((_CELSIUS - 25.5) / 10.0)  # ≈ 1.616 (KA)
_Q10_22: float = 3.0 ** ((_CELSIUS - 22.0) / 10.0)  # ≈ 2.408 (KM)
_Q10_23: float = 3.0 ** ((_CELSIUS - 23.0) / 10.0)  # ≈ 2.157 (SK2)

# Ca_LVA phi factors
_PHI_M_LVA: float = 5.0 ** ((_CELSIUS - 24.0) / 10.0)  # ≈ 2.627
_PHI_H_LVA: float = 3.0 ** ((_CELSIUS - 24.0) / 10.0)  # ≈ 1.933

# Nernst constant at 30°C: (R * T * 1000) / (2 * F) [mV]
_NERNST_CA: float = (8.314 * (_CELSIUS + 273.15) * 1000.0) / (2.0 * 96485.0)

# Ca dynamics factor: 1e4 / (2 * F)  [mol / (mA·cm²·um·ms)]
_CA_FACTOR: float = 1e4 / (2.0 * 96485.0)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

_p_default = None  # module-level default instance, set after class definition


@dataclass
class Solinas2007Params:
    """All conductances and kinetic parameters for the Solinas 2007 Golgi cell.
    Units: g [S/cm²], E [mV], diam/L [um], cm [uF/cm²].
    """

    celsius: float = 23.0

    # --- Reversal potentials --------------------------------------------------
    ena: float = 87.39
    ek: float = -84.69
    eca: float = 138.6  # HVA Ca (Nernst at cai0=50e-6, cao=2.0 mM, 30°C)
    el: float = -55.0  # leak
    Erev_hcn1: float = -20.0
    Erev_hcn2: float = -20.0

    # --- Cell geometry (soma) -------------------------------------------------
    diam: float = 27.0  # um
    L: float = 27.0  # um
    cm_spec: float = 1.0  # uF/cm²

    # --- Golgi_Na: fast transient Na (m^3 h), Q10 ref 20°C ------------------
    gnabar_Na: float = 0.048
    Aalpha_m_Na: float = 0.3  # uses _linoid_na
    Kalpha_m_Na: float = -10.0
    V0alpha_m_Na: float = -25.0
    Abeta_m_Na: float = 12.0
    Kbeta_m_Na: float = -18.182
    V0beta_m_Na: float = -50.0
    Aalpha_h_Na: float = 0.21
    Kalpha_h_Na: float = -3.333
    V0alpha_h_Na: float = -50.0
    Abeta_h_Na: float = 3.0
    Kbeta_h_Na: float = -5.0
    V0beta_h_Na: float = -17.0

    # --- Golgi_NaR: resurgent Na (s f), Q10 ref 20°C ------------------------
    gnabar_NaR: float = 0.0017
    Aalpha_s_NaR: float = -0.00493
    V0alpha_s_NaR: float = -4.48754
    Kalpha_s_NaR: float = -6.81881
    Shiftalpha_s_NaR: float = 0.00008
    Abeta_s_NaR: float = 0.01558
    V0beta_s_NaR: float = 43.97494
    Kbeta_s_NaR: float = 0.10818
    Shiftbeta_s_NaR: float = 0.04752
    Aalpha_f_NaR: float = 0.31836
    V0alpha_f_NaR: float = -80.0
    Kalpha_f_NaR: float = -62.52621
    Abeta_f_NaR: float = 0.01014
    V0beta_f_NaR: float = -83.3332
    Kbeta_f_NaR: float = 16.05379

    # --- Golgi_NaP: persistent Na (m), Q10 ref 30°C (=1) --------------------
    gnabar_NaP: float = 0.00019
    Aalpha_m_NaP: float = -0.91  # uses standard _linoid
    Kalpha_m_NaP: float = -5.0
    V0alpha_m_NaP: float = -40.0
    Abeta_m_NaP: float = 0.62
    Kbeta_m_NaP: float = 5.0
    V0beta_m_NaP: float = -40.0
    V0_minf_NaP: float = -43.0
    B_minf_NaP: float = 5.0

    # --- Golgi_KV: delayed rectifier K (n^4), Q10 ref 6.3°C -----------------
    gkbar_KV: float = 0.032
    Aalpha_n_KV: float = -0.01  # uses standard _linoid
    Kalpha_n_KV: float = -10.0
    V0alpha_n_KV: float = -26.0
    Abeta_n_KV: float = 0.125
    Kbeta_n_KV: float = -80.0
    V0beta_n_KV: float = -36.0

    # --- Golgi_KA: A-type K (a^3 b), Q10 ref 25.5°C -------------------------
    gkbar_KA: float = 0.008
    Aalpha_a_KA: float = 0.8147  # uses _sigm
    Kalpha_a_KA: float = -23.32708
    V0alpha_a_KA: float = -9.17203
    Abeta_a_KA: float = 0.1655
    Kbeta_a_KA: float = 19.47175
    V0beta_a_KA: float = -18.27914
    Aalpha_b_KA: float = 0.0368  # uses _sigm
    Kalpha_b_KA: float = 12.8433
    V0alpha_b_KA: float = -111.33209
    Abeta_b_KA: float = 0.0345
    Kbeta_b_KA: float = -8.90123
    V0beta_b_KA: float = -49.9537
    V0_ainf_KA: float = -38.0
    K_ainf_KA: float = -17.0
    V0_binf_KA: float = -78.8
    K_binf_KA: float = 8.4

    # --- Golgi_KM: M-current K (n), Q10 ref 22°C ----------------------------
    gkbar_KM: float = 0.001
    Aalpha_n_KM: float = 0.0033
    Kalpha_n_KM: float = 40.0
    V0alpha_n_KM: float = -30.0
    Abeta_n_KM: float = 0.0033
    Kbeta_n_KM: float = -20.0
    V0beta_n_KM: float = -30.0
    V0_ninf_KM: float = -35.0
    B_ninf_KM: float = 6.0

    # --- Golgi_BK: BK Ca-activated K (c), Q10 ref 30°C (=1) -----------------
    gkbar_BK: float = 0.003
    Aalpha_c_BK: float = 7.0
    Balpha_c_BK: float = 1.5e-3  # mM
    Kalpha_c_BK: float = -11.765
    Abeta_c_BK: float = 1.0
    Bbeta_c_BK: float = 0.15e-3  # mM
    Kbeta_c_BK: float = -11.765

    # --- Golgi_SK2: SK2 Ca-activated K (6-state Markov), Q10 ref 23°C -------
    gkbar_SK2: float = 0.038
    invc1_SK2: float = 80e-3  # /ms
    invc2_SK2: float = 80e-3
    invc3_SK2: float = 200e-3
    invo1_SK2: float = 1.0
    invo2_SK2: float = 100e-3
    diro1_SK2: float = 160e-3
    diro2_SK2: float = 1.2
    dirc2_SK2: float = 200.0  # /ms-mM
    dirc3_SK2: float = 160.0
    dirc4_SK2: float = 80.0
    diff_SK2: float = 3.0  # Ca diffusion/buffer factor

    # --- Golgi_Ca_HVA: HVA Ca (s^2 u), Q10 ref 20°C — same as D'Angelo -----
    gcabar_HVA: float = 460e-6
    Aalpha_s_HVA: float = 0.04944
    Kalpha_s_HVA: float = 15.87301587302
    V0alpha_s_HVA: float = -29.06
    Abeta_s_HVA: float = 0.08298
    Kbeta_s_HVA: float = -25.641
    V0beta_s_HVA: float = -18.66
    Aalpha_u_HVA: float = 0.0013
    Kalpha_u_HVA: float = -18.183
    V0alpha_u_HVA: float = -48.0
    Abeta_u_HVA: float = 0.0013
    Kbeta_u_HVA: float = 83.33
    V0beta_u_HVA: float = -48.0

    # --- Golgi_Ca_LVA: LVA Ca (m^2 h), phi_m=5^0.6, phi_h=3^0.6 at 30°C ---
    gca2bar_LVA: float = 2.5e-4
    shift_LVA: float = 2.0
    v0_m_inf_LVA: float = -50.0
    v0_h_inf_LVA: float = -78.0
    k_m_inf_LVA: float = -7.4
    k_h_inf_LVA: float = 5.0
    C_tau_m_LVA: float = 3.0
    A_tau_m_LVA: float = 1.0
    v0_tau_m1_LVA: float = -25.0
    v0_tau_m2_LVA: float = -100.0
    k_tau_m1_LVA: float = 10.0
    k_tau_m2_LVA: float = -15.0
    C_tau_h_LVA: float = 85.0
    A_tau_h_LVA: float = 1.0
    v0_tau_h1_LVA: float = -46.0
    v0_tau_h2_LVA: float = -405.0
    k_tau_h1_LVA: float = 4.0
    k_tau_h2_LVA: float = -50.0

    # --- Golgi_hcn1 ----------------------------------------------------------
    gbar_hcn1: float = 5e-5
    Ehalf_hcn1: float = -72.49
    c_hcn1: float = 0.11305
    rA_hcn1: float = 0.002096
    rB_hcn1: float = 0.97596
    tCf_hcn1: float = 0.01371
    tDf_hcn1: float = -3.368
    tEf_hcn1: float = 2.302585092
    tCs_hcn1: float = 0.01451
    tDs_hcn1: float = -4.056
    tEs_hcn1: float = 2.302585092

    # --- Golgi_hcn2 ----------------------------------------------------------
    gbar_hcn2: float = 8e-5
    Ehalf_hcn2: float = -81.95
    c_hcn2: float = 0.1661
    rA_hcn2: float = -0.0227
    rB_hcn2: float = -1.4694
    tCf_hcn2: float = 0.0269
    tDf_hcn2: float = -5.6111
    tEf_hcn2: float = 2.3026
    tCs_hcn2: float = 0.0152
    tDs_hcn2: float = -5.2944
    tEs_hcn2: float = 2.3026

    # --- Golgi_lkg: leak -----------------------------------------------------
    glbar_lkg: float = 21e-6

    # --- Ca dynamics (Golgi_CALC) --------------------------------------------
    ca_d: float = 0.2  # um, shell depth
    ca_beta: float = 1.3  # /ms, removal rate
    cai0: float = 50e-6   # mM, resting [Ca²⁺]
    cao: float = 2.0  # mM, extracellular


# ---------------------------------------------------------------------------
# Rate functions (used only during LUT construction — not called from step())
# ---------------------------------------------------------------------------

_p = Solinas2007Params()

# --- Golgi_Na ----------------------------------------------------------------


def _alp_m_Na(V: float) -> float:
    return _Q10_20 * _p.Aalpha_m_Na * _linoid_na(V - _p.V0alpha_m_Na, _p.Kalpha_m_Na)


def _bet_m_Na(V: float) -> float:
    return _Q10_20 * _p.Abeta_m_Na * math.exp((V - _p.V0beta_m_Na) / _p.Kbeta_m_Na)


def _m_Na_inf(V: float) -> float:
    a = _alp_m_Na(V)
    b = _bet_m_Na(V)
    return a / (a + b)


def _m_Na_tau(V: float) -> float:
    a = _alp_m_Na(V)
    b = _bet_m_Na(V)
    return 1.0 / (a + b)


def _alp_h_Na(V: float) -> float:
    return _Q10_20 * _p.Aalpha_h_Na * math.exp((V - _p.V0alpha_h_Na) / _p.Kalpha_h_Na)


def _bet_h_Na(V: float) -> float:
    return (
        _Q10_20 * _p.Abeta_h_Na / (1.0 + math.exp((V - _p.V0beta_h_Na) / _p.Kbeta_h_Na))
    )


def _h_Na_inf(V: float) -> float:
    a = _alp_h_Na(V)
    b = _bet_h_Na(V)
    return a / (a + b)


def _h_Na_tau(V: float) -> float:
    a = _alp_h_Na(V)
    b = _bet_h_Na(V)
    return 1.0 / (a + b)


# --- Golgi_NaR ---------------------------------------------------------------


def _alp_s_NaR(V: float) -> float:
    return _Q10_20 * (
        _p.Shiftalpha_s_NaR
        + _p.Aalpha_s_NaR * _linoid(V + _p.V0alpha_s_NaR, _p.Kalpha_s_NaR)
    )


def _bet_s_NaR(V: float) -> float:
    x1 = (V + _p.V0beta_s_NaR) / _p.Kbeta_s_NaR
    if x1 > 200.0:
        x1 = 200.0
    return _Q10_20 * (
        _p.Shiftbeta_s_NaR
        + _p.Abeta_s_NaR * (V + _p.V0beta_s_NaR) / (math.exp(x1) - 1.0)
    )


def _s_NaR_inf(V: float) -> float:
    a = _alp_s_NaR(V)
    b = _bet_s_NaR(V)
    return a / (a + b)


def _s_NaR_tau(V: float) -> float:
    a = _alp_s_NaR(V)
    b = _bet_s_NaR(V)
    return 1.0 / (a + b)


def _alp_f_NaR(V: float) -> float:
    return (
        _Q10_20 * _p.Aalpha_f_NaR * math.exp((V - _p.V0alpha_f_NaR) / _p.Kalpha_f_NaR)
    )


def _bet_f_NaR(V: float) -> float:
    return _Q10_20 * _p.Abeta_f_NaR * math.exp((V - _p.V0beta_f_NaR) / _p.Kbeta_f_NaR)


def _f_NaR_inf(V: float) -> float:
    a = _alp_f_NaR(V)
    b = _bet_f_NaR(V)
    return a / (a + b)


def _f_NaR_tau(V: float) -> float:
    a = _alp_f_NaR(V)
    b = _bet_f_NaR(V)
    return 1.0 / (a + b)


# --- Golgi_NaP ---------------------------------------------------------------


def _alp_m_NaP(V: float) -> float:
    # Q10 ref 30°C → factor = 1
    return _Q10_30 * _p.Aalpha_m_NaP * _linoid(V - _p.V0alpha_m_NaP, _p.Kalpha_m_NaP)


def _bet_m_NaP(V: float) -> float:
    return _Q10_30 * _p.Abeta_m_NaP * _linoid(V - _p.V0beta_m_NaP, _p.Kbeta_m_NaP)


def _m_NaP_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp(-(V - _p.V0_minf_NaP) / _p.B_minf_NaP))


def _m_NaP_tau(V: float) -> float:
    a = _alp_m_NaP(V)
    b = _bet_m_NaP(V)
    return 5.0 / (a + b)


# --- Golgi_KV ----------------------------------------------------------------


def _alp_n_KV(V: float) -> float:
    return _Q10_63 * _p.Aalpha_n_KV * _linoid(V - _p.V0alpha_n_KV, _p.Kalpha_n_KV)


def _bet_n_KV(V: float) -> float:
    return _Q10_63 * _p.Abeta_n_KV * math.exp((V - _p.V0beta_n_KV) / _p.Kbeta_n_KV)


def _n_KV_inf(V: float) -> float:
    a = _alp_n_KV(V)
    b = _bet_n_KV(V)
    return a / (a + b)


def _n_KV_tau(V: float) -> float:
    a = _alp_n_KV(V)
    b = _bet_n_KV(V)
    return 1.0 / (a + b)


# --- Golgi_KA ----------------------------------------------------------------


def _alp_a_KA(V: float) -> float:
    return _Q10_255 * _p.Aalpha_a_KA * _sigm(V - _p.V0alpha_a_KA, _p.Kalpha_a_KA)


def _bet_a_KA(V: float) -> float:
    return _Q10_255 * _p.Abeta_a_KA / math.exp((V - _p.V0beta_a_KA) / _p.Kbeta_a_KA)


def _a_KA_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp((V - _p.V0_ainf_KA) / _p.K_ainf_KA))


def _a_KA_tau(V: float) -> float:
    a = _alp_a_KA(V)
    b = _bet_a_KA(V)
    return 1.0 / (a + b)


def _alp_b_KA(V: float) -> float:
    return _Q10_255 * _p.Aalpha_b_KA * _sigm(V - _p.V0alpha_b_KA, _p.Kalpha_b_KA)


def _bet_b_KA(V: float) -> float:
    return _Q10_255 * _p.Abeta_b_KA * _sigm(V - _p.V0beta_b_KA, _p.Kbeta_b_KA)


def _b_KA_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp((V - _p.V0_binf_KA) / _p.K_binf_KA))


def _b_KA_tau(V: float) -> float:
    a = _alp_b_KA(V)
    b = _bet_b_KA(V)
    return 1.0 / (a + b)


# --- Golgi_KM ----------------------------------------------------------------


def _alp_n_KM(V: float) -> float:
    return _p.Aalpha_n_KM * math.exp((V - _p.V0alpha_n_KM) / _p.Kalpha_n_KM)


def _bet_n_KM(V: float) -> float:
    return _p.Abeta_n_KM * math.exp((V - _p.V0beta_n_KM) / _p.Kbeta_n_KM)


def _n_KM_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp(-(V - _p.V0_ninf_KM) / _p.B_ninf_KM))


def _n_KM_tau(V: float) -> float:
    # Q10 applied to sum: tau = 1 / (Q10_22 * (alpha + beta))
    return 1.0 / (_Q10_22 * (_alp_n_KM(V) + _bet_n_KM(V)))


# --- Golgi_Ca_HVA (same kinetics as D'Angelo) --------------------------------


def _alp_s_HVA(V: float) -> float:
    return (
        _Q10_20 * _p.Aalpha_s_HVA * math.exp((V - _p.V0alpha_s_HVA) / _p.Kalpha_s_HVA)
    )


def _bet_s_HVA(V: float) -> float:
    return _Q10_20 * _p.Abeta_s_HVA * math.exp((V - _p.V0beta_s_HVA) / _p.Kbeta_s_HVA)


def _s_HVA_inf(V: float) -> float:
    a = _alp_s_HVA(V)
    b = _bet_s_HVA(V)
    return a / (a + b)


def _s_HVA_tau(V: float) -> float:
    a = _alp_s_HVA(V)
    b = _bet_s_HVA(V)
    return 1.0 / (a + b)


def _alp_u_HVA(V: float) -> float:
    return (
        _Q10_20 * _p.Aalpha_u_HVA * math.exp((V - _p.V0alpha_u_HVA) / _p.Kalpha_u_HVA)
    )


def _bet_u_HVA(V: float) -> float:
    return _Q10_20 * _p.Abeta_u_HVA * math.exp((V - _p.V0beta_u_HVA) / _p.Kbeta_u_HVA)


def _u_HVA_inf(V: float) -> float:
    a = _alp_u_HVA(V)
    b = _bet_u_HVA(V)
    return a / (a + b)


def _u_HVA_tau(V: float) -> float:
    a = _alp_u_HVA(V)
    b = _bet_u_HVA(V)
    return 1.0 / (a + b)


# --- Golgi_Ca_LVA ------------------------------------------------------------


def _m_LVA_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp((V + _p.shift_LVA - _p.v0_m_inf_LVA) / _p.k_m_inf_LVA))


def _m_LVA_tau(V: float) -> float:
    vs = V + _p.shift_LVA
    return (
        _p.C_tau_m_LVA
        + _p.A_tau_m_LVA
        / (
            math.exp((vs - _p.v0_tau_m1_LVA) / _p.k_tau_m1_LVA)
            + math.exp((vs - _p.v0_tau_m2_LVA) / _p.k_tau_m2_LVA)
        )
    ) / _PHI_M_LVA


def _h_LVA_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp((V + _p.shift_LVA - _p.v0_h_inf_LVA) / _p.k_h_inf_LVA))


def _h_LVA_tau(V: float) -> float:
    vs = V + _p.shift_LVA
    return (
        _p.C_tau_h_LVA
        + _p.A_tau_h_LVA
        / (
            math.exp((vs - _p.v0_tau_h1_LVA) / _p.k_tau_h1_LVA)
            + math.exp((vs - _p.v0_tau_h2_LVA) / _p.k_tau_h2_LVA)
        )
    ) / _PHI_H_LVA


# --- Golgi_hcn1 --------------------------------------------------------------


def _of_hcn1_inf(V: float) -> float:
    o_inf = 1.0 / (1.0 + math.exp((V - _p.Ehalf_hcn1) * _p.c_hcn1))
    r = _p.rA_hcn1 * V + _p.rB_hcn1
    return o_inf * r


def _os_hcn1_inf(V: float) -> float:
    o_inf = 1.0 / (1.0 + math.exp((V - _p.Ehalf_hcn1) * _p.c_hcn1))
    r = _p.rA_hcn1 * V + _p.rB_hcn1
    return o_inf * (1.0 - r)


def _of_hcn1_tau(V: float) -> float:
    return math.exp(_p.tEf_hcn1 * (_p.tCf_hcn1 * V - _p.tDf_hcn1))


def _os_hcn1_tau(V: float) -> float:
    return math.exp(_p.tEs_hcn1 * (_p.tCs_hcn1 * V - _p.tDs_hcn1))


# --- Golgi_hcn2 --------------------------------------------------------------


def _of_hcn2_inf(V: float) -> float:
    o_inf = 1.0 / (1.0 + math.exp((V - _p.Ehalf_hcn2) * _p.c_hcn2))
    if V >= -64.70:
        r = 0.0
    elif V <= -108.70:
        r = 1.0
    else:
        r = _p.rA_hcn2 * V + _p.rB_hcn2
    return o_inf * r


def _os_hcn2_inf(V: float) -> float:
    o_inf = 1.0 / (1.0 + math.exp((V - _p.Ehalf_hcn2) * _p.c_hcn2))
    if V >= -64.70:
        r = 0.0
    elif V <= -108.70:
        r = 1.0
    else:
        r = _p.rA_hcn2 * V + _p.rB_hcn2
    return o_inf * (1.0 - r)


def _of_hcn2_tau(V: float) -> float:
    return math.exp(_p.tEf_hcn2 * (_p.tCf_hcn2 * V - _p.tDf_hcn2))


def _os_hcn2_tau(V: float) -> float:
    return math.exp(_p.tEs_hcn2 * (_p.tCs_hcn2 * V - _p.tDs_hcn2))


# ---------------------------------------------------------------------------
# LUT
# ---------------------------------------------------------------------------


def _lut_all(luts: dict, V: float) -> tuple:
    """Linear interpolation across all 34 LUT arrays."""
    idx_f = (V - _LUT_V_MIN) * _LUT_INV_DV
    idx = int(idx_f)
    if idx < 0:
        idx = 0
    elif idx >= _LUT_N - 1:
        idx = _LUT_N - 2
    frac = idx_f - idx
    i1 = idx + 1

    def _i(arr):
        return arr[idx] + frac * (arr[i1] - arr[idx])

    return (
        _i(luts["m_inf_Na"]),
        _i(luts["m_tau_Na"]),
        _i(luts["h_inf_Na"]),
        _i(luts["h_tau_Na"]),
        _i(luts["s_inf_NaR"]),
        _i(luts["s_tau_NaR"]),
        _i(luts["f_inf_NaR"]),
        _i(luts["f_tau_NaR"]),
        _i(luts["m_inf_NaP"]),
        _i(luts["m_tau_NaP"]),
        _i(luts["n_inf_KV"]),
        _i(luts["n_tau_KV"]),
        _i(luts["a_inf_KA"]),
        _i(luts["a_tau_KA"]),
        _i(luts["b_inf_KA"]),
        _i(luts["b_tau_KA"]),
        _i(luts["n_inf_KM"]),
        _i(luts["n_tau_KM"]),
        _i(luts["s_inf_HVA"]),
        _i(luts["s_tau_HVA"]),
        _i(luts["u_inf_HVA"]),
        _i(luts["u_tau_HVA"]),
        _i(luts["m_inf_LVA"]),
        _i(luts["m_tau_LVA"]),
        _i(luts["h_inf_LVA"]),
        _i(luts["h_tau_LVA"]),
        _i(luts["of_inf_hcn1"]),
        _i(luts["of_tau_hcn1"]),
        _i(luts["os_inf_hcn1"]),
        _i(luts["os_tau_hcn1"]),
        _i(luts["of_inf_hcn2"]),
        _i(luts["of_tau_hcn2"]),
        _i(luts["os_inf_hcn2"]),
        _i(luts["os_tau_hcn2"]),
    )


def _build_luts() -> dict:
    vs = [_LUT_V_MIN + i * _LUT_DV for i in range(_LUT_N)]
    return {
        "m_inf_Na": [_m_Na_inf(v) for v in vs],
        "m_tau_Na": [_m_Na_tau(v) for v in vs],
        "h_inf_Na": [_h_Na_inf(v) for v in vs],
        "h_tau_Na": [_h_Na_tau(v) for v in vs],
        "s_inf_NaR": [_s_NaR_inf(v) for v in vs],
        "s_tau_NaR": [_s_NaR_tau(v) for v in vs],
        "f_inf_NaR": [_f_NaR_inf(v) for v in vs],
        "f_tau_NaR": [_f_NaR_tau(v) for v in vs],
        "m_inf_NaP": [_m_NaP_inf(v) for v in vs],
        "m_tau_NaP": [_m_NaP_tau(v) for v in vs],
        "n_inf_KV": [_n_KV_inf(v) for v in vs],
        "n_tau_KV": [_n_KV_tau(v) for v in vs],
        "a_inf_KA": [_a_KA_inf(v) for v in vs],
        "a_tau_KA": [_a_KA_tau(v) for v in vs],
        "b_inf_KA": [_b_KA_inf(v) for v in vs],
        "b_tau_KA": [_b_KA_tau(v) for v in vs],
        "n_inf_KM": [_n_KM_inf(v) for v in vs],
        "n_tau_KM": [_n_KM_tau(v) for v in vs],
        "s_inf_HVA": [_s_HVA_inf(v) for v in vs],
        "s_tau_HVA": [_s_HVA_tau(v) for v in vs],
        "u_inf_HVA": [_u_HVA_inf(v) for v in vs],
        "u_tau_HVA": [_u_HVA_tau(v) for v in vs],
        "m_inf_LVA": [_m_LVA_inf(v) for v in vs],
        "m_tau_LVA": [_m_LVA_tau(v) for v in vs],
        "h_inf_LVA": [_h_LVA_inf(v) for v in vs],
        "h_tau_LVA": [_h_LVA_tau(v) for v in vs],
        "of_inf_hcn1": [_of_hcn1_inf(v) for v in vs],
        "of_tau_hcn1": [_of_hcn1_tau(v) for v in vs],
        "os_inf_hcn1": [_os_hcn1_inf(v) for v in vs],
        "os_tau_hcn1": [_os_hcn1_tau(v) for v in vs],
        "of_inf_hcn2": [_of_hcn2_inf(v) for v in vs],
        "of_tau_hcn2": [_of_hcn2_tau(v) for v in vs],
        "os_inf_hcn2": [_os_hcn2_inf(v) for v in vs],
        "os_tau_hcn2": [_os_hcn2_tau(v) for v in vs],
    }


_LUTS: dict = _build_luts()


# ---------------------------------------------------------------------------
# Rush-Larsen helper
# ---------------------------------------------------------------------------


def _rl(x: float, x_inf: float, tau: float, dt: float) -> float:
    if tau < 1e-9:
        return x_inf
    return x_inf + (x - x_inf) * math.exp(max(-dt / tau, -700.0))


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class Solinas2007CellModel(CellModel):
    """
    Solinas et al. 2007 cerebellar Golgi cell.

    State vector (26 variables):
      V(0)
      m_Na(1), h_Na(2)           — Golgi_Na
      s_NaR(3), f_NaR(4)        — Golgi_NaR
      m_NaP(5)                   — Golgi_NaP
      n_KV(6)                    — Golgi_KV
      a_KA(7), b_KA(8)           — Golgi_KA
      n_KM(9)                    — Golgi_KM
      c_BK(10)                   — Golgi_BK
      c2_SK2(11), c3_SK2(12),
      c4_SK2(13), o1_SK2(14),
      o2_SK2(15)                 — Golgi_SK2 (c1 = 1 - sum)
      s_HVA(16), u_HVA(17)      — Golgi_Ca_HVA
      m_LVA(18), h_LVA(19)      — Golgi_Ca_LVA
      of_hcn1(20), os_hcn1(21)  — Golgi_hcn1
      of_hcn2(22), os_hcn2(23)  — Golgi_hcn2
      cai(24)                    — Ca²⁺ HVA pool (read by BK, SK2)
      ca2i(25)                   — Ca²⁺ LVA pool (used for eca2)
    """

    def __init__(self, params: Solinas2007Params | None = None) -> None:
        self._p = params or Solinas2007Params()
        p = self._p
        self._Cm: float = p.cm_spec * 1e-3  # mF/cm²
        self._area_cm2: float = math.pi * p.diam * 1e-4 * p.L * 1e-4  # cm²
        self._state: list[float] = self._make_initial_state()

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def step(self, dt: float, I_ext_nA: float = 0.0) -> None:
        I_ext = I_ext_nA * 1e-6 / self._area_cm2  # mA/cm²
        p = self._p

        (
            V,
            m_Na,
            h_Na,
            s_NaR,
            f_NaR,
            m_NaP,
            n_KV,
            a_KA,
            b_KA,
            n_KM,
            c_BK,
            c2_SK2,
            c3_SK2,
            c4_SK2,
            o1_SK2,
            o2_SK2,
            s_HVA,
            u_HVA,
            m_LVA,
            h_LVA,
            of_hcn1,
            os_hcn1,
            of_hcn2,
            os_hcn2,
            cai,
            ca2i,
        ) = self._state

        # --- gate kinetics from LUT -----------------------------------------
        (
            m_inf_Na,
            m_tau_Na,
            h_inf_Na,
            h_tau_Na,
            s_inf_NaR,
            s_tau_NaR,
            f_inf_NaR,
            f_tau_NaR,
            m_inf_NaP,
            m_tau_NaP,
            n_inf_KV,
            n_tau_KV,
            a_inf_KA,
            a_tau_KA,
            b_inf_KA,
            b_tau_KA,
            n_inf_KM,
            n_tau_KM,
            s_inf_HVA,
            s_tau_HVA,
            u_inf_HVA,
            u_tau_HVA,
            m_inf_LVA,
            m_tau_LVA,
            h_inf_LVA,
            h_tau_LVA,
            of_inf_hcn1,
            of_tau_hcn1,
            os_inf_hcn1,
            os_tau_hcn1,
            of_inf_hcn2,
            of_tau_hcn2,
            os_inf_hcn2,
            os_tau_hcn2,
        ) = _lut_all(_LUTS, V)

        # --- BK: Ca- and V-dependent (computed per step) --------------------
        _cai_safe = max(cai, 1e-10)
        alp_c_BK = _Q10_30 * p.Aalpha_c_BK / (
            1.0 + p.Balpha_c_BK * math.exp(V / p.Kalpha_c_BK) / _cai_safe
        )
        bet_c_BK = _Q10_30 * p.Abeta_c_BK / (
            1.0 + _cai_safe / (p.Bbeta_c_BK * math.exp(V / p.Kbeta_c_BK))
        )
        c_inf_BK = alp_c_BK / (alp_c_BK + bet_c_BK)
        tau_c_BK = 1.0 / (alp_c_BK + bet_c_BK)

        # --- SK2: 6-state Markov chain (diagonal-exact per state + renormalise) ---
        tcorr_SK2 = _Q10_23
        invc1_t = p.invc1_SK2 * tcorr_SK2
        invc2_t = p.invc2_SK2 * tcorr_SK2
        invc3_t = p.invc3_SK2 * tcorr_SK2
        invo1_t = p.invo1_SK2 * tcorr_SK2
        invo2_t = p.invo2_SK2 * tcorr_SK2
        diro1_t = p.diro1_SK2 * tcorr_SK2
        diro2_t = p.diro2_SK2 * tcorr_SK2
        dirc2_t = p.dirc2_SK2 * _cai_safe * tcorr_SK2 / p.diff_SK2
        dirc3_t = p.dirc3_SK2 * _cai_safe * tcorr_SK2 / p.diff_SK2
        dirc4_t = p.dirc4_SK2 * _cai_safe * tcorr_SK2 / p.diff_SK2

        c1_SK2 = max(0.0, 1.0 - c2_SK2 - c3_SK2 - c4_SK2 - o1_SK2 - o2_SK2)

        # For each stored state x: dx/dt = src(t) - lam*x, sources frozen at step start.
        # Exact solution: x_new = eq + (x - eq)*exp(-lam*dt), eq = src/lam >= 0.
        # Guarantees x_new >= 0 without clipping; lam > 0 always (constant reverse rates).
        lam2 = invc1_t + dirc3_t
        eq2 = (dirc2_t * c1_SK2 + invc2_t * c3_SK2) / lam2
        c2_new = eq2 + (c2_SK2 - eq2) * math.exp(max(-lam2 * dt, -700.0))

        lam3 = invc2_t + dirc4_t + diro1_t
        eq3 = (dirc3_t * c2_SK2 + invc3_t * c4_SK2 + invo1_t * o1_SK2) / lam3
        c3_new = eq3 + (c3_SK2 - eq3) * math.exp(max(-lam3 * dt, -700.0))

        lam4 = invc3_t + diro2_t
        eq4 = (dirc4_t * c3_SK2 + invo2_t * o2_SK2) / lam4
        c4_new = eq4 + (c4_SK2 - eq4) * math.exp(max(-lam4 * dt, -700.0))

        eq_o1 = diro1_t * c3_SK2 / invo1_t
        o1_new = eq_o1 + (o1_SK2 - eq_o1) * math.exp(max(-invo1_t * dt, -700.0))

        eq_o2 = diro2_t * c4_SK2 / invo2_t
        o2_new = eq_o2 + (o2_SK2 - eq_o2) * math.exp(max(-invo2_t * dt, -700.0))

        # Rescale if approximation error from frozen sources pushes sum above 1 (keeps c1 >= 0)
        _sk2_sum = c2_new + c3_new + c4_new + o1_new + o2_new
        if _sk2_sum > 1.0:
            _inv_sum = 1.0 / _sk2_sum
            c2_new *= _inv_sum
            c3_new *= _inv_sum
            c4_new *= _inv_sum
            o1_new *= _inv_sum
            o2_new *= _inv_sum

        # --- LVA Ca reversal potential (Nernst, uses previous ca2i) ---------
        eca2 = _NERNST_CA * math.log(p.cao / max(ca2i, 1e-10))

        # --- Conductances ---------------------------------------------------
        g_Na = p.gnabar_Na * m_Na * m_Na * m_Na * h_Na
        g_NaR = p.gnabar_NaR * s_NaR * f_NaR
        g_NaP = p.gnabar_NaP * m_NaP
        g_KV = p.gkbar_KV * n_KV * n_KV * n_KV * n_KV
        g_KA = p.gkbar_KA * a_KA * a_KA * a_KA * b_KA
        g_KM = p.gkbar_KM * n_KM
        g_BK = p.gkbar_BK * c_BK
        g_SK2 = p.gkbar_SK2 * (o1_SK2 + o2_SK2)
        g_HVA = p.gcabar_HVA * s_HVA * s_HVA * u_HVA
        g_LVA = p.gca2bar_LVA * m_LVA * m_LVA * h_LVA
        g_hcn1 = p.gbar_hcn1 * (of_hcn1 + os_hcn1)
        g_hcn2 = p.gbar_hcn2 * (of_hcn2 + os_hcn2)
        g_lkg = p.glbar_lkg

        # --- Voltage: Rush-Larsen via conductance sum -----------------------
        g_tot = (
            g_Na
            + g_NaR
            + g_NaP
            + g_KV
            + g_KA
            + g_KM
            + g_BK
            + g_SK2
            + g_HVA
            + g_LVA
            + g_hcn1
            + g_hcn2
            + g_lkg
        )
        I_gE = (
            g_Na * p.ena
            + g_NaR * p.ena
            + g_NaP * p.ena
            + g_KV * p.ek
            + g_KA * p.ek
            + g_KM * p.ek
            + g_BK * p.ek
            + g_SK2 * p.ek
            + g_HVA * p.eca
            + g_LVA * eca2
            + g_hcn1 * p.Erev_hcn1
            + g_hcn2 * p.Erev_hcn2
            + g_lkg * p.el
        )
        V_inf = (I_ext + I_gE) / g_tot
        V_new = V_inf + (V - V_inf) * math.exp(-g_tot * dt / self._Cm)

        # --- Ca²⁺: semi-implicit Backward Euler (HVA pool) ------------------
        ica_HVA = g_HVA * (V_new - p.eca)
        ca_src_HVA = -ica_HVA * _CA_FACTOR / p.ca_d
        cai_new = (cai + dt * (ca_src_HVA + p.ca_beta * p.cai0)) / (
            1.0 + dt * p.ca_beta
        )
        cai_new = max(cai_new, 1e-10)

        # --- Ca²⁺: semi-implicit Backward Euler (LVA pool) ------------------
        ica_LVA = g_LVA * (V_new - eca2)
        ca_src_LVA = -ica_LVA * _CA_FACTOR / p.ca_d
        ca2i_new = (ca2i + dt * (ca_src_LVA + p.ca_beta * p.cai0)) / (
            1.0 + dt * p.ca_beta
        )
        ca2i_new = max(ca2i_new, 1e-10)

        # --- Gate variables: Rush-Larsen ------------------------------------
        m_Na_new = _rl(m_Na, m_inf_Na, m_tau_Na, dt)
        h_Na_new = _rl(h_Na, h_inf_Na, h_tau_Na, dt)
        s_NaR_new = _rl(s_NaR, s_inf_NaR, s_tau_NaR, dt)
        f_NaR_new = _rl(f_NaR, f_inf_NaR, f_tau_NaR, dt)
        m_NaP_new = _rl(m_NaP, m_inf_NaP, m_tau_NaP, dt)
        n_KV_new = _rl(n_KV, n_inf_KV, n_tau_KV, dt)
        a_KA_new = _rl(a_KA, a_inf_KA, a_tau_KA, dt)
        b_KA_new = _rl(b_KA, b_inf_KA, b_tau_KA, dt)
        n_KM_new = _rl(n_KM, n_inf_KM, n_tau_KM, dt)
        c_BK_new = _rl(c_BK, c_inf_BK, tau_c_BK, dt)
        s_HVA_new = _rl(s_HVA, s_inf_HVA, s_tau_HVA, dt)
        u_HVA_new = _rl(u_HVA, u_inf_HVA, u_tau_HVA, dt)
        m_LVA_new = _rl(m_LVA, m_inf_LVA, m_tau_LVA, dt)
        h_LVA_new = _rl(h_LVA, h_inf_LVA, h_tau_LVA, dt)
        of_hcn1_new = _rl(of_hcn1, of_inf_hcn1, of_tau_hcn1, dt)
        os_hcn1_new = _rl(os_hcn1, os_inf_hcn1, os_tau_hcn1, dt)
        of_hcn2_new = _rl(of_hcn2, of_inf_hcn2, of_tau_hcn2, dt)
        os_hcn2_new = _rl(os_hcn2, os_inf_hcn2, os_tau_hcn2, dt)

        self._state = [
            V_new,
            m_Na_new,
            h_Na_new,
            s_NaR_new,
            f_NaR_new,
            m_NaP_new,
            n_KV_new,
            a_KA_new,
            b_KA_new,
            n_KM_new,
            c_BK_new,
            c2_new,
            c3_new,
            c4_new,
            o1_new,
            o2_new,
            s_HVA_new,
            u_HVA_new,
            m_LVA_new,
            h_LVA_new,
            of_hcn1_new,
            os_hcn1_new,
            of_hcn2_new,
            os_hcn2_new,
            cai_new,
            ca2i_new,
        ]

    def get_voltage(self) -> float:
        return float(self._state[0])

    def get_calcium(self) -> float:
        return float(self._state[24])

    def reset(self) -> None:
        self._state = self._make_initial_state()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _make_initial_state(self) -> list[float]:
        V0 = -60.0  # mV — near trough of Golgi pacemaker oscillation
        p = self._p

        # BK at V0
        _cai = p.cai0
        alp_c0 = _Q10_30 * p.Aalpha_c_BK / (
            1.0 + p.Balpha_c_BK * math.exp(V0 / p.Kalpha_c_BK) / _cai
        )
        bet_c0 = _Q10_30 * p.Abeta_c_BK / (
            1.0 + _cai / (p.Bbeta_c_BK * math.exp(V0 / p.Kbeta_c_BK))
        )
        c_BK0 = alp_c0 / (alp_c0 + bet_c0)

        # SK2 — nearly all in c1 at resting cai
        c2_0 = c3_0 = c4_0 = o1_0 = o2_0 = 0.0

        return [
            V0,
            _m_Na_inf(V0),
            _h_Na_inf(V0),
            _s_NaR_inf(V0),
            _f_NaR_inf(V0),
            _m_NaP_inf(V0),
            _n_KV_inf(V0),
            _a_KA_inf(V0),
            _b_KA_inf(V0),
            _n_KM_inf(V0),
            c_BK0,
            c2_0,
            c3_0,
            c4_0,
            o1_0,
            o2_0,
            _s_HVA_inf(V0),
            _u_HVA_inf(V0),
            _m_LVA_inf(V0),
            _h_LVA_inf(V0),
            _of_hcn1_inf(V0),
            _os_hcn1_inf(V0),
            _of_hcn2_inf(V0),
            _os_hcn2_inf(V0),
            p.cai0,
            p.cai0,
        ]
