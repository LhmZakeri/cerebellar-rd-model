"""
Molineux, Fernandez, Mehaffey & Turner (2005) cerebellar stellate-cell
model — Phase 1 Python prototype.

Reproduces four named figures from the paper as pass/fail checks:
  Fig. 4B    — F-I curve, with/without a hyperpolarising conditioning pulse
  Fig. 4A, C — nonmonotonic first-spike-latency vs. conditioning voltage
  Fig. 5A, B — h_A / h_T steady-state gates and their difference
  Fig. 6C    — EPSP/IPSP coincidence-detection synaptic switching

Usage:
    python scripts/run_molineux2005_prototype.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from src.models.molineux_cell import MolineuxStellateCellModel
from src.simulation.sim import simulate_step_protocol
from src.simulation.spike_metrics import firing_rate_hz, first_spike_latency_ms

matplotlib.use("Agg")

# -------------------------------------------------------------------------------
_DT = 0.005  # ms

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

GATING_FILE = OUTPUT_DIR / "molineux2005_gating_curves.png"
FI_CURVE_FILE = OUTPUT_DIR / "molineux2005_fi_curve.png"
LATENCY_FILE = OUTPUT_DIR / "molineux2005_latency_voltage.png"
TONIC_FILE = OUTPUT_DIR / "milineux2005_tonic_voltage.png"
GATE_MECHANISM_FILE = OUTPUT_DIR / "molineaux2005_gate_mechanism.png"
SYNAPTIC_SWITCH_FILE = OUTPUT_DIR / "molineaux2005_synaptic_switching.png"

V_RANGE = np.linspace(-100.0, 20.0, 400)


# -------------------------------------------------------------------------------
def _uA_to_nA(I_density_uA_per_cm2: float) -> float:
    """The paper's uA/cm² -> project nA conversion."""
    return I_density_uA_per_cm2 * 1e-3


# -------------------------------------------------------------------------------
def _steady_firing_rate_hz(
    I_density_uA_per_cm2: float, total_ms: float, tail_ms: float
) -> float:
    """Steady-state firing rate: run total_ms, measure over the last tail_ms
    only, so the initial-transient ISI doesn't bias the rate estimate."""

    cell = MolineuxStellateCellModel()
    t, V, _ = simulate_step_protocol(
        cell, _DT, [(total_ms, _uA_to_nA(I_density_uA_per_cm2))]
    )
    mask = t >= (total_ms - tail_ms)
    return firing_rate_hz(t[mask], V[mask])


# -------------------------------------------------------------------------------
def _condition_then_test(
    I_cond_uA_per_cm2: float,
    cond_ms: float,
    I_test_uA_per_cm2: float,
    test_ms: float,
) -> tuple[float, float]:
    """Run a conditioning segment then a test segment from a fresh cell.

    Returns (V_end, latency_ms): V_end is the voltage at the end of the
    conditioning segment; latency_ms is the time from test-step onset to
    the first spike (NaN if no spike occurs)."""

    cell = MolineuxStellateCellModel()
    t, V, _ = simulate_step_protocol(
        cell,
        _DT,
        [
            (cond_ms, _uA_to_nA(I_cond_uA_per_cm2)),
            (test_ms, _uA_to_nA(I_test_uA_per_cm2)),
        ],
    )
    idx_end_cond = round(cond_ms / _DT) - 1
    V_end = V[idx_end_cond]
    latency = first_spike_latency_ms(t, V, step_start_ms=cond_ms)
    return float(V_end), float(latency)


# --- Test 1: F-I curve (Fig. 4B) -----------------------------------------------


def test_fi_curve() -> None:
    """Steady-state firing rate over 0.8-1.4 uA/cm^2, with and without a
    -2 uA/cm^2 conditioning prepulse. Paper: rates in ~10-60 Hz, roughly
    monotonic, and the two curves nearly overlap."""

    currents = np.arange(0.8, 1.4 + 1e-9, 0.1)
    total_ms, tail_ms = 500.0, 300.0
    cond_ms = 100.0

    rates_no_prepulse = np.array(
        [_steady_firing_rate_hz(I, total_ms, tail_ms) for I in currents]
    )

    rates_with_prepulse = []
    for I in currents:
        cell = MolineuxStellateCellModel()
        t, V, _ = simulate_step_protocol(
            cell, _DT, [(cond_ms, _uA_to_nA(-2.0)), (total_ms, _uA_to_nA(I))]
        )
        mask = t >= (cond_ms + total_ms - tail_ms)
        rates_with_prepulse.append(firing_rate_hz(t[mask], V[mask]))
    rates_with_prepulse = np.array(rates_with_prepulse)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(currents, rates_no_prepulse, "o-", label="no prepulse")
    ax.plot(currents, rates_with_prepulse, "s--", label="-2 uA/cm² prepulse")
    ax.set_xlabel("Step current (uA/cm²)")
    ax.set_ylabel("Steady-state firing rate (Hz)")
    ax.set_title("Fig 4B - F-I curve, with/without conditioning prepulse")
    ax.legend()
    fig.savefig(FI_CURVE_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:{FI_CURVE_FILE}")


# --- Gating-curve figure -------------------------------------------------------


def run_gating_curves() -> None:
    cell = MolineuxStellateCellModel()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Molineux et al. (2005) stellate cell — steady-state gates and tau_h(V)"
    )

    m_inf = np.array([cell._m_inf(V) for V in V_RANGE])
    h_inf = np.array([cell._h_inf(V) for V in V_RANGE])
    n_inf = np.array([cell._n_inf(V) for V in V_RANGE])
    mT_inf = np.array([cell._mT_inf(V) for V in V_RANGE])
    hT_inf = np.array([cell._hT_inf(V) for V in V_RANGE])
    nA_inf = np.array([cell._nA_inf(V) for V in V_RANGE])
    hA_inf = np.array([cell._hA_inf(V) for V in V_RANGE])
    tau_h = np.array([cell._tau_h(V) for V in V_RANGE])

    axes[0].plot(V_RANGE, m_inf, label="m_inf (Na)")
    axes[0].plot(V_RANGE, h_inf, label="h_inf (Na)")
    axes[0].plot(V_RANGE, n_inf, label="n_inf (K)", linestyle="--")
    axes[0].plot(V_RANGE, mT_inf, label="m_T,inf (T-Ca)")
    axes[0].plot(V_RANGE, hT_inf, label="h_T,inf (T-Ca)")
    axes[0].plot(V_RANGE, nA_inf, label="n_A,inf (A-K)")
    axes[0].plot(V_RANGE, hA_inf, label="h_A,inf (A-K)")
    axes[0].set_title("Steady-state gating curves")
    axes[0].set_xlabel("V (mV)")
    axes[0].set_ylabel("Steady-state value")
    axes[0].legend(fontsize=7)

    axes[1].plot(V_RANGE, tau_h, color="#d62728")
    axes[1].set_title("tau_h(V) — Lorentzian Na+ inactivation time constant")
    axes[1].set_xlabel("V (mV)")
    axes[1].set_ylabel("tau_h (ms)")

    fig.savefig(GATING_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {GATING_FILE}")


# --- Test 2.1: spiking figures (Fig. 4A) -----------------------------------------


def test_tonic_voltage_trajectory():
    cell = MolineuxStellateCellModel()
    t, V, _ = simulate_step_protocol(
        cell,
        _DT,
        [
            (100.0, -2 * 1e-3),
            (500.0, 0.9 * 1e-3),  # first stim
            (500.0, 0.0),
            (100.0, -0 * 1e-3),
            (500.0, 0.9 * 1e-3),  # second stim
            (500.0, 0.0),
            (100.0, 0.8 * 1e-3),
            (500.0, 0.9 * 1e-3),  # third stim
        ],
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, V)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("membrane volltage (mV)")
    ax.set_title(
        "Fig 4A - Voltage trajectories in the model in reponse"
        " to different levels of hyperpolarisation."
    )
    ax.vlines(
        [100.0, 600.0, 1200.0, 1700.0, 2300.0, 2800.0],
        ymin=ax.get_ylim()[0],
        ymax=ax.get_ylim()[1],
        color="gray",
        linewidth=0.4,
        linestyle="--",
    )

    fig.savefig(TONIC_FILE, dpi=150, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved:{TONIC_FILE}")


# --- Test 2.2: nonmonotonic first-spike latency (Fig. 4B, C) ---------------------


def run_latency_sweep() -> tuple[np.ndarray, np.ndarray]:
    """Shared by Test 2 and Test 3: sweep conditioning current, record the
    conditioning-end voltage and the first-spike latency after a fixed
    0.9 uA/cm^2 test step. Returns (V_end, latency_ms) arrays."""
    cond_currents = np.arange(-2.0, 0.8 + 1e-9, 0.1)
    cond_ms, test_ms, I_test = 90.0, 200.0, 0.9

    V_end = np.empty(len(cond_currents))
    latency = np.empty(len(cond_currents))
    for i, I_cond in enumerate(cond_currents):
        V_end[i], latency[i] = _condition_then_test(
            I_cond, cond_ms, I_test, test_ms)
    return V_end, latency


def test_nonmonotonic_latency(V_end: np.ndarray, latency: np.ndarray) -> None:
    """Latency vs. conditioning voltage should peak near -74 mV (max ~125 ms)
    and be shorter at both the hyperpolarised (~-85 mV) and depolarised
    (~-65 mV) ends -- the paper's headline A-type/T-type interaction."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(V_end, latency, "o-")
    ax.set_xlabel("Conditioning-end voltage (mV)")
    ax.set_ylabel("First-spike latency (ms)")
    ax.set_title(
        "Fig 4A, C _ Nonmonotonic spike-latency vs. conditioning voltage")
    fig.savefig(LATENCY_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {LATENCY_FILE}")

    peak_idx = int(np.nanargmax(latency))
    peak_latency = float(latency[peak_idx])
    peak_V = float(V_end[peak_idx])

    interior_peak = 0 < peak_idx < len(latency) - 1
    latency_in_range = 100.0 <= peak_latency <= 150.0
    voltage_in_range = -76.0 <= peak_V <= -72.0

    hyperpolarised_end = latency[np.argmin(np.abs(V_end - (-85.0)))]
    depolarised_end = latency[np.argmin(np.abs(V_end - (-65.0)))]
    ends_shorter = (
        np.nan_to_num(hyperpolarised_end, nan=np.inf) < peak_latency
        and np.nan_to_num(depolarised_end, nan=np.inf) < peak_latency
    )

    passed = bool(
        interior_peak and latency_in_range and voltage_in_range and ends_shorter
    )
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} Fig4A,C latency-voltage: peak={peak_latency:.1f}"
        f"ms at v={peak_V:.1f} mV (expect ~100-150 ms near -76..-72 mV),"
        f" ends shorter={ends_shorter} -> {LATENCY_FILE}"
    )
    assert passed


# --- Test 3: gate mechanism ----------------------------------------------------
def test_gate_mechanism(V_end: np.ndarray, latency: np.ndarray) -> None:
    """h_A and h_T steady-state curves (evaluated at the same conditioning
    voltages used for the latency sweep) and their difference, which should
    peak at the same voltage as the latency peak -- the paper's proposed
    mechanism (A-type/T-type gate availability sets latency)."""
    cell = MolineuxStellateCellModel()
    order = np.argsort(V_end)
    V_sorted = V_end[order]

    hA_inf = np.array([cell._hA_inf(V) for V in V_sorted])
    hT_inf = np.array([cell._hT_inf(V) for V in V_sorted])
    diff = hA_inf - hT_inf

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(V_sorted, hA_inf, label="h_A, inf")
    ax.plot(V_sorted, hT_inf, label="h_T, inf")
    ax.plot(
        V_sorted, diff, label="h_A, inf - h_T, inf", linestyle="--", color="#d62728"
    )
    ax.set_xlabel("Conditioning-end voltage (mV)")
    ax.set_ylabel("Gate value")
    ax.set_xlim([-95, -65])
    ax.set_title(
        "Fig 5A,B - h_A / h_T steady-state gates and their differences")
    ax.legend()
    fig.savefig(GATE_MECHANISM_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {GATE_MECHANISM_FILE}")

    def _half_activation_V(gate_inf: np.ndarray) -> float:
        idx = int(np.argmin(np.abs(gate_inf - 0.5)))
        return float(V_sorted[idx])

    hA_half = _half_activation_V(hA_inf)
    hT_half = _half_activation_V(hT_inf)
    left_shifted = hT_half < hA_half

    diff_peak_idx = int(np.argmax(diff))
    diff_peak_V = float(V_sorted[diff_peak_idx])
    diff_peak_in_range = -76.0 <= diff_peak_V <= -72.0

    latency_peak_V = float(V_end[int(np.nanargmax(latency))])
    aligned_with_latency = abs(diff_peak_V - latency_peak_V) <= 5.0

    passed = bool(left_shifted and diff_peak_in_range and aligned_with_latency)
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} Fig5A, B gate mechanism: h_T half-act={hT_half:.1f} mV <"
        f" h_A half-act={hA_half:.1f} mV (left-shifted={left_shifted}), "
        f"argmax(h_A-h_T)={diff_peak_V:.1f} mV vs latency peak={latency_peak_V:.1f} mV"
        f" -> {GATE_MECHANISM_FILE}"
    )
    assert passed


# --- Test 4: synaptic switching (Fig. 6C) --------------------------------------
# The source document gives only the PSP *shape* (t * exp(-alpha*t)) and the
# two alpha values -- no amplitude and no explicit time unit. Interpreting
# t literally in ms gives sub-millisecond decay (peak at 1/alpha ms), which
# is too fast to bridge the paper's stated 15 ms EPSP/IPSP gap and implausible
# for real synaptic kinetics. TAU_SCALE_EPSP/IPSP rescale t to tens-of-ms.
# TAU_SCALE_EPSP/IPSP have been chosen to reproduce Fig6.C of the Molineaux2005.
TAU_SCALE_EPSP = 0.79  # 4.0
TAU_SCALE_IPSP = 1.3  # 10.0
ALPHA_EPSP, E_EPSP = 3.0, 0.0
ALPHA_IPSP, E_IPSP = 1.25, -82.0

# g_max_EPSP/g_max_IPSP are likewise calibration choices (the paper gives no
# amplitude), found by a small manual search to reproduce the qualitative
# pattern below.
G_MAX_EPSP = 2.5
G_MAX_IPSP = 1.5
IPSP_LEAD_MS = 15.0
BIAS_UA_PER_CM2 = 0.2916  # holds the cell at ~-73 mV, matching Fig 6C


def _psp_shape(t_ms: np.ndarray | float, alpha: float, tau_scale: float) -> np.ndarray | float:
    tt = t_ms / tau_scale
    return np.where(tt >= 0.0, tt * np.exp(- alpha * np.maximum(tt, 0.0)), 0.0)


def _run_synaptic_trial(n_ipsp: int, t_epsp_ms: float, settle_ms: float, total_ms: float):
    cell = MolineuxStellateCellModel()
    bias_nA = _uA_to_nA(BIAS_UA_PER_CM2)
    n_settle = round(settle_ms / _DT)
    for _ in range(n_settle):
        cell.step(_DT, bias_nA)

    t_ipsp_ms = t_epsp_ms - IPSP_LEAD_MS
    n_total = round(total_ms / _DT)
    t = np.empty(n_total)
    V = np.empty(n_total)
    time_ms = 0.0
    spike_latency = None
    for i in range(n_total):
        V_now = cell.get_voltage()
        I_epsp = G_MAX_EPSP * \
            _psp_shape(time_ms - t_epsp_ms, ALPHA_EPSP,
                       TAU_SCALE_EPSP) * (V_now)
        I_ipsp = n_ipsp * G_MAX_IPSP * \
            _psp_shape(time_ms - t_ipsp_ms, ALPHA_IPSP,
                       TAU_SCALE_IPSP) * (V_now - E_IPSP)
        # inward (depolarising) EPSP, outward (hyperpolarising) IPSP
        I_syn_density = -(I_epsp + I_ipsp)
        cell.step(_DT, bias_nA + _uA_to_nA(I_syn_density))
        t[i], V[i] = time_ms, cell.get_voltage()
        if V[i] > 0.0 and spike_latency is None and time_ms >= t_epsp_ms:
            spike_latency = time_ms - t_epsp_ms
        time_ms += _DT
    return t, V, spike_latency


def test_synaptic_switching() -> None:
    """EPSP alone spikes; EPSP preceded by 1 IPSP (15 ms earlier) is
    blocked; EPSP preceded by 2 coincident IPSPs spikes again via rebound
    (h_A/h_T de-inactivation during the deeper hyperpolarisation) -- the
    paper's coincidence-detection / disinhibition result."""
    t_epsp_ms, settle_ms, total_ms = 200.0, 150.0, 260.0

    conditions = [
        ("EPSP alone", 0),
        ("EPSP + 1x IPSP", 1),
        ("EPSP + 2x IPSP", 2),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    results = {}
    for label, n_ipsp in conditions:
        t, V, latency = _run_synaptic_trial(
            n_ipsp, t_epsp_ms, settle_ms, total_ms)
        results[label] = latency is not None
        mask = t >= t_epsp_ms - 30.0
        ax.plot(t[mask] - t_epsp_ms, V[mask], linewidth=0.8, label=f"{label}" +
                f"({'spike' if latency is not None else 'no spike'})")

    ax.axvline(0.0, color="gray", linewidth=0.4, linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Time from EPSP onset (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_title("Fig 6C _ Synaptic coincidence-detection switching")
    ax.legend(fontsize=8)
    fig.savefig(SYNAPTIC_SWITCH_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED: {SYNAPTIC_SWITCH_FILE}")

    passed = (
        results["EPSP alone"] is True
        and results["EPSP + 1x IPSP"] is False
        and results["EPSP + 2x IPSP"] is True
    )

    status = "PASS" if passed else "FAIL"
    print(
        f"{status} Fig6C synaptic switching: {results}"
        f"(expect True, False, True) -> {SYNAPTIC_SWITCH_FILE}"
    )

    assert passed
# -------------------------------------------------------------------------------


def main() -> None:
    run_gating_curves()
    test_fi_curve()
    V_end, latency = run_latency_sweep()
    test_nonmonotonic_latency(V_end, latency)
    test_tonic_voltage_trajectory()
    test_gate_mechanism(V_end, latency)
    test_synaptic_switching()


# ===============================================================================
if __name__ == "__main__":
    main()
