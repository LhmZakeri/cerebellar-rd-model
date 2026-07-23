"""
D'Angelo & Solinas granule cell ionic model — Phase 1 Python prototype.

Single compartment, 10 state variables.

References:
  D'Angelo et al. (2001) J Physiol — GrC theta-frequency bursting
  Solinas et al. (2010) Front Neurosci — GrC/GoC model

Performance:
  All voltage-dependent gate kinetics (x_inf, tau_x) are precomputed into
  lookup tables over [_LUT_V_MIN, _LUT_V_MAX] at module import.  Each call
  to step() computes one integer index then does 14 linear interpolations —
  no transcendental function calls for gate kinetics in the hot path.

The code has been written in Python using the Modeldb script in Neuron
"""
import math
from dataclasses import dataclass
# --- helpers shared by several rate functions ---------------------------------


def _linoid(x: float, y: float) -> float:
    """
    Linearised exponential: x / (exp(x/y) - 1)
    Gaurds against overflow for larg positive x/y (occursr in GrG_Nar beta_s where
    Kbeta_s = 0.10818 mV is very small)
    """
    ratio = x / y
    if abs(ratio) < 1e-6:
        return y * (1.0 - ratio / 2.0)
    if ratio > 500.0:
        return 0.0
    if ratio < -500.0:
        return -x
    return x / (math.exp(ratio) - 1.0)


def _sigm(x: float, y: float) -> float:
    return 1.0 / (math.exp(x/y) + 1.0)


# --- Lookup table constants ---------------------------------------------------
_LUT_V_MIN: float = -100.0
_LUT_V_MAX: float = 80.0
_LUT_DV: float = 0.1
_LUT_INV_DV: float = 10.0
_LUT_N: int = 1801

# --- Parameters ---------------------------------------------------------------


@dataclass
class DAngelo2001Params:
    """
    All conductance densities and reversal potantials
    Units : g [S/cm^2], E[mV]
    diam = L = 9.76 um, Cm = 1 uF/cm^2
    """
    celsius: float = 30.0
    # --- Reversal potentials --------------------------------------------------
    ena: float = 87.39  # mv
    ek: float = -84.69  # mv
    eca: float = 129.33  # mv
    ecl: float = -65.0  # mv
    # --- GrG_Na m*3 h fast transien Na ----------------------------------------
    gnabar_Na: float = 0.013  # S/cm^2

    Aalpha_m_Na: float = -0.3  # /ms-mv
    Kalpha_m_Na: float = -10.0  # mv
    V0alpha_m_Na: float = -19.0  # mv
    Abeta_m_Na: float = 12.0  # /ms
    Kbeta_m_Na: float = -18.182  # mv
    V0beta_m_Na: float = -44.0  # mv

    Aalpha_h_Na: float = 0.105  # /ms
    Kalpha_h_Na: float = -3.333  # mv
    V0alpha_h_Na: float = -44.0  # mv
    Abeta_h_Na: float = 1.5  # /ms
    Kbeta_h_Na: float = -5.0  # mv
    V0beta_h_Na: float = -11.0  # mv

    # --- GrG_Nar s.f resurgant Na ---------------------------------------------
    gnabar_Nar: float = 0.0005  # S/cm^2

    # s gate (slow inactivation / en-channel block)
    Aalpha_s_Nar: float = -0.00493  # /ms
    V0alpha_s_Nar: float = -4.48754  # mv
    Kalpha_s_Nar: float = -6.81881  # mv
    shiftalpha_s_Nar: float = 0.00008  # /ms
    Abeta_s_Nar: float = 0.01558  # /ms
    V0beta_s_Nar: float = 43.97494  # mv
    kbeta_s_Nar: float = 0.10818  # mv
    Shiftbeta_s_Nar: float = 0.04752  # /ms

    # f gate (fast recovery)
    Aalpha_f_Nar: float = 0.31836  # /ms
    V0alpha_f_Nar: float = -80.0  # mV
    Kalpha_f_Nar: float = -62.52621  # mV
    Abeta_f_Nar: float = 0.01014  # /ms
    V0beta_f_Nar: float = -83.3332  # mv
    Kbeta_f_Nar: float = 16.05379  # mv

    # --- GrG_pNa m persistant Na ----------------------------------------------
    gnabar_pNa: float = 2e-5  # S/cm^2

    Aalpha_m_pNa: float = -0.091  # /mv -ms
    Kalpha_m_pNa: float = -5.0  # mv
    V0alpha_m_pNa: float = -42.0  # mv
    Abeta_m_pNa: float = 0.062  # /ms
    Kbeta_m_pNa: float = 5.0  # mv
    V0beta_m_pNa: float = -42.0  # mv
    V0_minf_pNa: float = -42.0  # mv
    B_minf_pNa: float = 5.0  # mv

    # --- GrG_KV n^4 delayed rectifier k ---------------------------------------
    gkbar_KV: float = 0.003  # S/cm^2

    Aalpha_n_KV: float = -0.01  # /ms-mv
    Kalpha_n_KV: float = -10.0  # mv
    V0alpha_n_KV: float = -25.0  # mv
    Abeta_n_KV: float = 0.125  # /ms
    Kbeta_n_KV: float = -80.0  # mv
    V0beta_n_KV: float = -35.0  # mv

    # --- GrG_KM n M-current / K slow ------------------------------------------
    gkbar_KM: float = 0.00035  # S/cm^2

    Aalpha_n_KM: float = 0.0033  # /ms
    Kalpha_n_KM: float = 40.0  # mv
    V0alpha_n_KM: float = -30.0  # mv
    Abeta_n_KM: float = 0.0033  # /ms
    Kbeta_n_KM: float = -20.0  # mv
    V0beta_n_KM: float = -30.0  # mv
    V0_ninf_KM: float = -30.0  # mv
    B_ninf_Km: float = 6.0  # mv

    # --- Grc_KA a^3b A-type K -------------------------------------------------
    gkbar_KA: float = 0.004  # S/cm^2

    Aalpha_a_KA: float = 4.88826  # /ms
    Kalpha_a_KA: float = -23.32708  # mv
    V0alpha_a_KA: float = -9.17203  # mv
    Abeta_a_KA: float = 0.99285  # /ms
    Kbeta_a_KA: float = 19.47175  # mv
    V0beta_a_KA: float = -18.27914  # mv

    Aalpha_b_KA: float = 0.11042  # /ms
    Kalpha_b_Ka: float = 12.8433  # mv
    V0alpha_b_KA: float = -111.33209  # mv
    Abeta_b_KA: float = 0.10353  # /ms
    Kbeta_b_KA: float = -8.90123  # mv
    V0beta_b_KA: float = -49.9537  # mv

    V0_ainf_KA: float = -46.7  # mv
    K_ainf_KA: float = -19.8  # mv
    V0_binf_KA: float = -78.8  # mv
    K_binf_KA: float = 8.4  # mv

    # --- GrC_Kir d inward rectifier K -----------------------------------------
    gkbar_Kir: float = 0.0009  # S/cm^2

    Aalpha_d_Kir: float = 0.13289  # /ms
    Kalpha_d_Kir: float = -24.3902  # mv
    V0alpha_d_Kir: float = -83.94  # mv
    Abeta_d_Kir: float = 0.16994  # /ms
    Kbeta_d_Kir: float = 35.714  # mv
    V0beta_d_Kir: float = -83.94  # mv

    # --- GrC_KCa c Ca-activated K ---------------------------------------------
    gkbar_KCa: float = 0.004    # S/cm^2

    Aalpha_c_KCa: float = 2.5   # /ms
    Balpha_c_KCa: float = 1.5e-3  # mM
    Kalpha_c_KCa: float = -11.765  # mV
    Abeta_c_KCa: float = 1.5  # /ms
    Bbeta_c_KCa: float = 0.15e-3  # mM
    Kbeta_c_KCa: float = -11.765  # mV

    # --- GrC_CaHVA s^2u HVA Ca ------------------------------------------------
    gcabar_Ca: float = 0.00046  # S/cm^2

    Aalpha_s_Ca: float = 0.04944  # /ms
    Kalpha_s_Ca: float = 15.87301587302  # mv
    V0alpha_s_Ca: float = -29.06  # mv
    Abeta_s_Ca: float = 0.08298  # /ms
    Kbeta_s_Ca: float = -25.641  # mv
    V0beta_s_Ca: float = -18.66  # mv

    Aalpha_u_Ca: float = 0.0013  # /ms
    Kalpha_u_Ca: float = -18.183  # mv
    V0alpha_u_Ca: float = -48.0  # mv
    Abeta_u_Ca: float = 0.0013  # /ms
    Kbeta_u_Ca: float = 83.33  # mv
    V0beta_u_Ca: float = -48.0  # mv

    # --- Calc Ca^2+ dynamics ----------------------------------------------------
    # d[Ca] / dt = -ica/(2F.d)*1e4 - beta *(cai - cai0)
    # d = shell depth [um], beta = removal rate [/ms]
    ca_d: float = 0.2  # um shell depth
    ca_beta: float = 1.5  # /ms removal rate
    cai0: float = 1e-4  # mM resting [Ca^2+]i
    cao: float = 2.0  # mM extracellular

    # ics [mA/cm^2], d [um], cai [mM]
    # factor = 1/(2 * 9685 C/mol) * 1e4 [um-> cm] * 1e-3[mA-> A] * 1e3[mol/L-> mM]
    # = 1e4 / (2 * 96485) = 0.05182
    ca_factor: float = 1.0 / (2.0 * 96485.0)  # mol/(mA.cm.ms)

    # --- Leaks ------------------------------------------------------------------
    gl_Lkg1: float = 5.68e-5  # S/cm^2
    el_Lkg1: float = -58.0  # mv
    ggaba_Lkg2: float = 2.17e-5  # S/cm^2
    egaba_Lkg2: float = -65.0  # mv

    # --- Cell -------------------------------------------------------------------
    diam: float = 9.76  # um
    L: float = 9.76  # um
    cm_spec: float = 1.0  # uF/cm^2

# --- LUT helper ---------------------------------------------------------------


def _lut_all_2001(luts: dict, V: float) -> tuple:
    """Linear interpolation"""

    idx_f = (V - _LUT_V_MIN) * _LUT_INV_DV
    idx = int(idx_f)
    if idx < 0:
        idx = 0
    elif idx >= _LUT_N - 1:
        idx = _LUT_N - 2
    frac = idx_f - idx
    i1 = idx + 1

    def _interp(arr):
        return arr[idx] + frac * (arr[i1] - arr[idx])

    return (
        _interp(luts['m_inf_Na']), _interp(luts['m_tau_Na']),
        _interp(luts['h_inf_Na']), _interp(luts['h_tau_Na']),
        _interp(luts['s_inf_Nar']), _interp(luts['s_tau_Nar']),
        _interp(luts['f_inf_Nar']), _interp(luts['f_tau_Nar']),
        _interp(luts['m_inf_pNa']), _interp(luts['m_tau_pNa']),
        _interp(luts['n_inf_KV']), _interp(luts['n_tau_KV']),
        _interp(luts['n_inf_KM']), _interp(luts['n_tau_KM']),
        _interp(luts['a_inf_KA']), _interp(luts['a_tau_KA']),
        _interp(luts['b_inf_KA']), _interp(luts['b_tau_KA']),
        _interp(luts['d_inf_Kir']), _interp(luts['d_tau_Kir']),
        _interp(luts['s_inf_Ca']), _interp(luts['s_tau_Ca']),
        _interp(luts['u_inf_Ca']), _interp(luts['u_tau_Ca']),
    )

# --- Model --------------------------------------------------------------------


class DAngelo2001CellModel:
    """
    D'Angelo 2001 cerebrall granule cell.

    State vector:
      V(0),
      m_Na(1), h_Na(2),           - Grc_Na      fast transiet Na
      s_Na(3), f_Nar(4),          - GrC_Nar     resurgent Na
      m_pNa(5),                   - GrC_pNa     persistent Na
      n_KV(6),                    - GrG_KV      delayed rectifier K
      n_KM(7),                    - GrG_KM      M-current / K slow
      a_KA(8), b_KA(9),           - GrC_KA      A-type K
      d_Kir(10),                  - GrC_Kir     inward rectifier K
      c_KCa(11),                  - GrC_KCa     Ca-activated K
      s_Ca(12), u_Ca(13),         - GrC_CaHVA   HVA Ca
      cai(14)                     - Calc        intracellular Ca^2+
    """
    # ----------------------------------------------------------------------------

    def __init__(self, params: DAngelo2001Params | None = None) -> None:
        """
        # Unit system: mV, ms, mA/cm^2, S/cm^2, mF/cm^2
        # tau = Cm[mF/cm^2] / g[S/cm^2]
        # 1 uF/cm^2  = 1e-3 mF/cm^2 -> tau =1e-3/0.013 ~ 0.077 ms 
        """
        self._p = params or DAngelo2001Params()
        self._Cm: float = self._p.cm_spec * 1e-3  # mF/cm^2
        self._area_cm2: float = (
            math.pi * self._p.diam * 1e-4 * self._p.L * 1e-4)  # cm^2
        self._state: list[float] = self._make_initial_state()

    # --- Public interface -----------------------------------------------------
    def step(self, dt: float, I_ext_nA: float = 0.0) -> None:
        """
        Advanced by dt [ms] with I_ext_nA [nA] injected current. Internally 
        converted to mA/cm^2 via: I_nA * 1e-6 / area_cm^2.
        """
        I_ext_mA_cm2 = I_ext_nA * 1e-6 / self._area_cm2
        p = self._p
        (V, m_Na, h_Na, s_Nar, f_Nar, m_pNa, n_KV, n_KM, a_KA,
         b_KA, d_Kir, c_KCa, s_Ca, u_Ca, cai) = self._state

        # --- gate kinetics from LUT -------------------------------------------
        (m_inf_Na, m_tau_Na,
         h_inf_Na, h_tau_Na,
         s_inf_Nar, s_tau_Nar,
         f_inf_Nar, f_tau_Nar,
         m_inf_pNa, m_tau_pNa,
         n_inf_KV, n_tau_KV,
         n_inf_KM, n_tau_KM,
         a_inf_KA, a_tau_KA,
         b_inf_KA, b_tau_KA,
         d_inf_Kir, d_tau_Kir,
         s_inf_Ca, s_tau_Ca,
         u_inf_Ca, u_tau_Ca) = _lut_all_2001(_LUTS_2001, V)

        # --- KCa: c_inf / tau_c depend on cai ---------------------------------
        # alpha_c = alpha_c = Aalpha_c / (1 + Balpha_c*exp(V/Kalpha_c)/cai)
        # beta_c = Abeta_c  / (1 + cai/(Bbeta_c*exp(V/Kbeta_c)))
        # Q10 for KCa is referenced to 30 degree -> Q10 = 1 at celcius=30

        _cai_safe = max(cai, 1e-10)
        alpha_c = p.Aalpha_c_KCa / \
            (1.0 + p.Balpha_c_KCa * math.exp(V / p.Kalpha_c_KCa)/_cai_safe)
        beta_c = p.Abeta_c_KCa / \
            (1.0 + _cai_safe / (p.Bbeta_c_KCa * math.exp(V / p.Kbeta_c_KCa)))
        c_inf = alpha_c / (alpha_c + beta_c)
        tau_c = 1.0 / (alpha_c + beta_c)

        # --- Conductances -----------------------------------------------------
        g_Na = p.gnabar_Na * m_Na * m_Na * m_Na * h_Na
        g_Nar = p.gnabar_Nar * s_Nar * f_Nar
        g_pNa = p.gnabar_pNa * m_pNa
        g_KV = p.gkbar_KV * n_KV * n_KV * n_KV * n_KV
        g_KM = p.gkbar_KM * n_KM
        g_KA = p.gkbar_KA * a_KA * a_KA * a_KA * b_KA
        g_Kir = p.gkbar_Kir * d_Kir
        g_KCa = p.gkbar_KCa * c_KCa
        g_Ca = p.gcabar_Ca * s_Ca * s_Ca * u_Ca
        g_Lkg1 = p.gl_Lkg1
        g_Lkg2 = p.ggaba_Lkg2

        # --- Voltage: Rush-Larsen via conoductance sum ------------------------
        g_tot = (g_Na + g_Nar + g_pNa + g_KV + g_KM + g_KA +
                 g_Kir + g_KCa + g_Ca + g_Lkg1 + g_Lkg2)
        I_gE = (g_Na * p.ena + g_Nar * p.ena + g_pNa * p.ena
                + g_KV * p.ek + g_KM * p.ek + g_KA * p.ek + g_Kir * p.ek
                + g_KCa * p.ek + g_Ca * p.eca + g_Lkg1 * p.el_Lkg1 +
                g_Lkg2 * p.egaba_Lkg2)
        V_inf = (I_ext_mA_cm2 + I_gE) / g_tot
        V_new = V_inf + (V - V_inf) * math.exp(-g_tot * dt / self._Cm)

        # --- Ca^2+: semi-implicit Backward Euler ------------------------------
        # cai' = -ica/(2F.d)*1e4 - beta*(cai - cai0)
        # source: ica evaluated at V_new for better accuracy
        ica_at_Vnew = g_Ca * (V_new - p.eca)
        # -ica/(2F) * 1e4 [um -> cm conversion]:
        ca_source = -ica_at_Vnew * 1e4 * p.ca_factor / p.ca_d
        cai_new = (cai + dt*(ca_source + p.ca_beta * p.cai0)) / \
            (1.0 + dt * p.ca_beta)
        cai_new = max(cai_new, 1e-10)

        # --- Gate variables: Rush-Larsen --------------------------------------
        m_Na_new = _rl(m_Na, m_inf_Na, m_tau_Na, dt)
        h_Na_new = _rl(h_Na, h_inf_Na, h_tau_Na, dt)
        s_Nar_new = _rl(s_Nar, s_inf_Nar, s_tau_Nar, dt)
        f_Nar_new = _rl(f_Nar, f_inf_Nar, f_tau_Nar, dt)
        m_pNa_new = _rl(m_pNa, m_inf_pNa, m_tau_pNa, dt)
        n_KV_new = _rl(n_KV, n_inf_KV, n_tau_KV, dt)
        n_KM_new = _rl(n_KM, n_inf_KM, n_tau_KM, dt)
        a_KA_new = _rl(a_KA, a_inf_KA, a_tau_KA, dt)
        b_KA_new = _rl(b_KA, b_inf_KA, b_tau_KA, dt)
        d_Kir_new = _rl(d_Kir, d_inf_Kir, d_tau_Kir, dt)
        c_KCa_new = _rl(c_KCa, c_inf, tau_c, dt)
        s_Ca_new = _rl(s_Ca, s_inf_Ca, s_tau_Ca, dt)
        u_Ca_new = _rl(u_Ca, u_inf_Ca, u_tau_Ca, dt)

        self._state = [
            V_new,
            m_Na_new, h_Na_new,
            s_Nar_new, f_Nar_new,
            m_pNa_new,
            n_KV_new, n_KM_new,
            a_KA_new, b_KA_new,
            d_Kir_new,
            c_KCa_new,
            s_Ca_new, u_Ca_new,
            cai_new,
        ]
    # --------------------------------------------------------------------------

    def get_voltage(self) -> float:
        return float(self._state[0])

    def get_calcium(self) -> float:
        return float(self._state[14])

    def reset(self) -> None:
        self._state = self._make_initial_state()

    # --- Initialisation -------------------------------------------------------
    def _make_initial_state(self) -> list[float]:
        V0 = -80
        cai = self._p.cai0
        p = self._p
        _cai_safe = max(cai, 1e-10)
        a_c0 = p.Aalpha_c_KCa / \
            (1.0 + p.Balpha_c_KCa * math.exp(V0 / p.Kalpha_c_KCa)/_cai_safe)
        b_c0 = p.Abeta_c_KCa / \
            (1.0 + _cai_safe / (p.Bbeta_c_KCa * math.exp(V0 / p.Kbeta_c_KCa)))
        c0 = a_c0 / (a_c0 + b_c0)
        return [
            V0,
            _m_Na_inf(V0), _h_Na_inf(V0),
            _s_Nar_inf(V0), _f_Nar_inf(V0),
            _m_pNa_inf(V0),
            _n_KV_inf(V0), _n_KM_inf(V0),
            _a_KA_inf(V0), _b_KA_inf(V0),
            _d_Kir_inf(V0),
            c0,
            _s_Ca_inf(V0), _u_Ca_inf(V0),
            cai,
        ]

# ------------------------------------------------------------------------------


def _rl(x: float, x_inf: float, tau: float, dt: float) -> float:
    """Exact solution of dx/dt = (x_inf - x )/tau."""
    return x_inf + (x - x_inf) * math.exp(-dt / tau)

# --- Rate functions -----------------------------------------------------------
# Q10 factors are evaluated at celcius=30 ad folded in:
# GrG_Na, GrC_CaHVA, GrC_KA, GrC_Kir, GrC_Nar: Q10 ref 20 degree -> Q10^(10/10)=3
# GrG_KV: Q10 ref 6.3 degree -> Q10^(23.7/10) = 3^2.37 ~ 13.97
# GrG_KM: Q10 re 22 degree -> Q10^(8/10) = 3^0.8 ~ 2.408
# GrC_KCa: Q10 ref 30 degree -> Q10^0  = 1
# GrC_pNa: Q10 ref 30 degree -> Q10^0  = 1


_CELSIUS = 30.0
_Q10_20 = 3.0 ** ((_CELSIUS - 20.0) / 10.0)  # = 3.0
_Q10_63 = 3.0 ** ((_CELSIUS - 6.3) / 10.0)  # ~ 13.97
_Q10_22 = 3.0 ** ((_CELSIUS - 22.0) / 10.0)  # ~ 2.408

# --- GRG_Na -------------------------------------------------------------------
p_Na = DAngelo2001Params()


def _alp_m_Na(V: float) -> float:
    return _Q10_20 * p_Na.Aalpha_m_Na * _linoid(V - p_Na.V0alpha_m_Na,
                                                p_Na.Kalpha_m_Na)


def _bet_m_Na(V: float) -> float:
    return _Q10_20 * p_Na.Abeta_m_Na * math.exp((V - p_Na.V0beta_m_Na
                                                 ) / p_Na.Kbeta_m_Na)


def _m_Na_inf(V: float) -> float:
    a = _alp_m_Na(V)
    b = _bet_m_Na(V)
    return a / (a + b)


def _m_Na_tau(V: float) -> float:
    a = _alp_m_Na(V)
    b = _bet_m_Na(V)
    return 1.0 / (a + b)


def _alp_h_Na(V: float) -> float:
    return _Q10_20 * p_Na.Aalpha_h_Na * math.exp((V - p_Na.V0alpha_h_Na
                                                  ) / p_Na.Kalpha_h_Na)


def _bet_h_Na(V: float) -> float:
    return _Q10_20 * p_Na.Abeta_h_Na / (1.0 + math.exp((V - p_Na.V0beta_h_Na
                                                        ) / p_Na.Kbeta_h_Na))


def _h_Na_inf(V: float) -> float:
    a = _alp_h_Na(V)
    b = _bet_h_Na(V)
    return a / (a + b)


def _h_Na_tau(V: float) -> float:
    a = _alp_h_Na(V)
    b = _bet_h_Na(V)
    return 1.0 / (a + b)


# --- GrG_Nar ------------------------------------------------------------------
p_Nar = DAngelo2001Params()


def _alp_s_Nar(V: float) -> float:
    return _Q10_20*(p_Nar.shiftalpha_s_Nar + p_Nar.Aalpha_s_Nar *
                    _linoid(V + p_Nar.V0alpha_s_Nar, p_Nar.Kalpha_s_Nar))


def _bet_s_Nar(V: float) -> float:
    return _Q10_20*(p_Nar.Shiftbeta_s_Nar + p_Nar.Abeta_s_Nar *
                    _linoid(V + p_Nar.V0beta_s_Nar, p_Nar.kbeta_s_Nar))


def _s_Nar_inf(V: float) -> float:
    a = _alp_s_Nar(V)
    b = _bet_s_Nar(V)
    return a / (a + b)


def _s_Nar_tau(V: float) -> float:
    a = _alp_s_Nar(V)
    b = _bet_s_Nar(V)
    return 1.0 / (a + b)


def _alp_f_Nar(V: float) -> float:
    return _Q10_20 * p_Nar.Aalpha_f_Nar * math.exp(
        (V - p_Nar.V0alpha_f_Nar)/p_Nar.Kalpha_f_Nar)


def _bet_f_Nar(V: float) -> float:
    return _Q10_20 * p_Nar.Abeta_f_Nar * math.exp(
        (V - p_Nar.V0beta_f_Nar) / p_Nar.Kbeta_f_Nar)


def _f_Nar_inf(V: float) -> float:
    a = _alp_f_Nar(V)
    b = _bet_f_Nar(V)
    return a / (a + b)


def _f_Nar_tau(V: float) -> float:
    a = _alp_f_Nar(V)
    b = _bet_f_Nar(V)
    return 1.0 / (a + b)


# --- GrC_pNa-------------------------------------------------------------------
# Q10 ref 30 degree -> factor = 1
p_pNa = DAngelo2001Params()


def _alp_m_pNa(V: float) -> float:
    # Q10 at celcius=30
    return p_pNa.Aalpha_m_pNa * _linoid(V - p_pNa.V0alpha_m_pNa,
                                        p_pNa.Kalpha_m_pNa)


def _bet_m_pNa(V: float) -> float:
    return p_pNa.Abeta_m_pNa * _linoid(V - p_pNa.V0beta_m_pNa,
                                       p_pNa.Kbeta_m_pNa)


def _m_pNa_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp(-(V - p_pNa.V0_minf_pNa)/p_pNa.B_minf_pNa))


def _m_pNa_tau(V: float) -> float:
    a = _alp_m_pNa(V)
    b = _bet_m_pNa(V)
    return 5.0 / (a + b)


# --- GrG_KV -------------------------------------------------------------------
p_KV = DAngelo2001Params()


def _alp_n_KV(V: float) -> float:
    return _Q10_63 * p_KV.Aalpha_n_KV * _linoid(
        V - p_KV.V0alpha_n_KV, p_KV.Kalpha_n_KV)


def _bet_n_KV(V: float) -> float:
    return _Q10_63 * p_KV.Abeta_n_KV * math.exp((
        V - p_KV.V0beta_n_KV) / p_KV.Kbeta_n_KV)


def _n_KV_inf(V: float) -> float:
    a = _alp_n_KV(V)
    b = _bet_n_KV(V)
    return a / (a + b)


def _n_KV_tau(V: float) -> float:
    a = _alp_n_KV(V)
    b = _bet_n_KV(V)
    return 1.0 / (a + b)


# --- GrG_KM -------------------------------------------------------------------
p_KM = DAngelo2001Params()


def _alp_n_KM(V: float) -> float:
    return _Q10_22 * p_KM.Aalpha_n_KM * math.exp((
        V - p_KM.V0alpha_n_KM) / p_KM.Kalpha_n_KM)


def _bet_n_KM(V: float) -> float:
    return _Q10_22 * p_KM.Abeta_n_KM * math.exp((
        V - p_KM.V0beta_n_KM) / p_KM.Kbeta_n_KM)


def _n_KM_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp(-(V - p_KM.V0_ninf_KM)/p_KM.B_ninf_Km))


def _n_KM_tau(V: float) -> float:
    a = _alp_n_KM(V)
    b = _bet_n_KM(V)
    return 1.0 / (a + b)


# --- GrC_KA -------------------------------------------------------------------
p_KA = DAngelo2001Params()


def _alp_a_KA(V: float) -> float:
    return _Q10_20 * p_KA.Aalpha_a_KA * _sigm(V - p_KA.V0alpha_a_KA,
                                              p_KA.Kalpha_a_KA)


def _bet_a_KA(V: float) -> float:
    return _Q10_20 * p_KA.Abeta_a_KA / math.exp((
        V - p_KA.V0beta_a_KA) / p_KA.Kbeta_a_KA)


def _a_KA_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp((V - p_KA.V0_ainf_KA)/p_KA.K_ainf_KA))


def _a_KA_tau(V: float) -> float:
    a = _alp_a_KA(V)
    b = _bet_a_KA(V)
    return 1.0 / (a+b)


def _alp_b_KA(V: float) -> float:
    return _Q10_20 * p_KA.Aalpha_b_KA * _sigm(V - p_KA.V0alpha_b_KA,
                                              p_KA.Kalpha_b_Ka)


def _bet_b_KA(V: float) -> float:
    return _Q10_20 * p_KA.Abeta_b_KA * _sigm(V - p_KA.V0beta_b_KA,
                                             p_KA.Kbeta_b_KA)


def _b_KA_inf(V: float) -> float:
    return 1.0 / (1.0 + math.exp((V - p_KA.V0_binf_KA)/p_KA.K_binf_KA))


def _b_KA_tau(V: float) -> float:
    a = _alp_b_KA(V)
    b = _bet_b_KA(V)
    return 1.0 / (a + b)


# --- GrC_Kir ------------------------------------------------------------------
p_Kir = DAngelo2001Params()


def _alp_d_Kir(V: float) -> float:
    return _Q10_20 * p_Kir.Aalpha_d_Kir * math.exp((V - p_Kir.V0alpha_d_Kir
                                                    )/p_Kir.Kalpha_d_Kir)


def _bet_d_Kir(V: float) -> float:
    return _Q10_20 * p_Kir.Abeta_d_Kir * math.exp((V - p_Kir.V0beta_d_Kir
                                                   ) / p_Kir.Kbeta_d_Kir)


def _d_Kir_inf(V: float) -> float:
    a = _alp_d_Kir(V)
    b = _bet_d_Kir(V)
    return a / (a + b)


def _d_Kir_tau(V: float) -> float:
    a = _alp_d_Kir(V)
    b = _bet_d_Kir(V)
    return 1.0 / (a + b)


# --- GrC_CaHVA ----------------------------------------------------------------
p_Ca = DAngelo2001Params()


def _alp_s_Ca(V: float) -> float:
    return _Q10_20 * p_Ca.Aalpha_s_Ca * math.exp((V - p_Ca.V0alpha_s_Ca
                                                  ) / p_Ca.Kalpha_s_Ca)


def _bet_s_Ca(V: float) -> float:
    return _Q10_20 * p_Ca.Abeta_s_Ca * math.exp((V - p_Ca.V0beta_s_Ca
                                                 ) / p_Ca.Kbeta_s_Ca)


def _s_Ca_inf(V: float) -> float:
    a = _alp_s_Ca(V)
    b = _bet_s_Ca(V)
    return a / (a+b)


def _s_Ca_tau(V: float) -> float:
    a = _alp_s_Ca(V)
    b = _bet_s_Ca(V)
    return 1.0 / (a+b)


def _alp_u_Ca(V: float) -> float:
    return _Q10_20 * p_Ca.Aalpha_u_Ca * math.exp((V - p_Ca.V0alpha_u_Ca
                                                  ) / p_Ca.Kalpha_u_Ca)


def _bet_u_Ca(V: float) -> float:
    return _Q10_20 * p_Ca.Abeta_u_Ca * math.exp((V - p_Ca.V0beta_u_Ca
                                                 )/p_Ca.Kbeta_u_Ca)


def _u_Ca_inf(V: float) -> float:
    a = _alp_u_Ca(V)
    b = _bet_u_Ca(V)
    return a / (a+b)


def _u_Ca_tau(V: float) -> float:
    a = _alp_u_Ca(V)
    b = _bet_u_Ca(V)
    return 1.0 / (a+b)

# --- Lookup tables (built once at module import) ------------------------------


def _build_luts_2001() -> dict[str, list]:
    """Precompute all 24 voltage-dependent kinetic array over [V_MIN, V_MAX]."""
    vs = [_LUT_V_MIN + i * _LUT_DV for i in range(_LUT_N)]
    return {
        "m_inf_Na": [_m_Na_inf(v) for v in vs],
        "m_tau_Na": [_m_Na_tau(v) for v in vs],
        "h_inf_Na": [_h_Na_inf(v) for v in vs],
        "h_tau_Na": [_h_Na_tau(v) for v in vs],
        "s_inf_Nar": [_s_Nar_inf(v) for v in vs],
        "s_tau_Nar": [_s_Nar_tau(v) for v in vs],
        "f_inf_Nar": [_f_Nar_inf(v) for v in vs],
        "f_tau_Nar": [_f_Nar_tau(v) for v in vs],
        "m_inf_pNa": [_m_pNa_inf(v) for v in vs],
        "m_tau_pNa": [_m_pNa_tau(v) for v in vs],
        "n_inf_KV": [_n_KV_inf(v) for v in vs],
        "n_tau_KV": [_n_KV_tau(v) for v in vs],
        "n_inf_KM": [_n_KM_inf(v) for v in vs],
        "n_tau_KM": [_n_KM_tau(v) for v in vs],
        "a_inf_KA": [_a_KA_inf(v) for v in vs],
        "a_tau_KA": [_a_KA_tau(v) for v in vs],
        "b_inf_KA": [_b_KA_inf(v) for v in vs],
        "b_tau_KA": [_b_KA_tau(v) for v in vs],
        "d_inf_Kir": [_d_Kir_inf(v) for v in vs],
        "d_tau_Kir": [_d_Kir_tau(v) for v in vs],
        "s_inf_Ca": [_s_Ca_inf(v) for v in vs],
        "s_tau_Ca": [_s_Ca_tau(v) for v in vs],
        "u_inf_Ca": [_u_Ca_inf(v) for v in vs],
        "u_tau_Ca": [_u_Ca_tau(v) for v in vs]

    }


_LUTS_2001: dict[str, list] = _build_luts_2001()
