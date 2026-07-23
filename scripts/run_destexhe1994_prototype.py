"""
Destexhe, Mainen & Sejnowski (1994/1998) two-state kinetic synapse —
Phase 1 Python prototype validation script.

Reproduces:
  Fig. 6 (top-left panel) — single AMPA/kainate PSC: fast rise, mono-exponential
                             decay. Best paper-faithful single-event test.
  Table 1                 — GABA-A and NMDA single-pulse kinetics (fitted rate
                             constants; the paper gives no dedicated figure panel
                             for either, unlike AMPA).

Fig. 7 (four-spike, 20 ms/0.1 nA current-pulse summation test) is NOT
reproduced here: it requires a presynaptic Na/K membrane model, a stochastic
calcium-dependent vesicular release model, and a 6-state postsynaptic AMPA
kinetic scheme (Fig 7's panel C), none of which exist in this repo and none
of which match TwoStateDestexhe (the minimal 2-state scheme the project
roadmap in DESIGN.md commits to project-wide).

Source: ModelDB accession #18198, https://modeldb.science/18198
  (ampa.mod / gabaa.mod). NMDA rate constants are Table 1 values from the
  paper itself; TwoStateDestexhe has no NMDA preset (only AMPA/GABA-A are on
  the DESIGN.md roadmap), so they are applied directly via TwoStateDestexheParams
  here, for validation only.

Usage:
    python scripts/run_destexhe1994_prototype.py
"""

import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

from src.models.destexhe_synapse import TwoStateDestexhe, TwoStateDestexheParams

matplotlib.use("Agg")

# -----------------------------------------------------------------------------
_DT = 0.025  # ms
REST_V = -70.0  # mV -- presynaptic voltage with no spike
SPIKE_V = 30.0  # mV -- presynaptic voltage during an spike

GMAX = 1.0  # nS
EXC_V_HOLD = -70.0  # mV -- Voltage-clamp holding potential for AMPA/NMDA PSCs
INH_V_HOLD = -40.0  # mV -- Holding potential that reveals outward GABA-A IPSCs
# (Erev=-80 mV)

NMDA_ALPHA = 0.072  # /ms/mM
NMDA_BETA = 0.0066  # /ms
NMDA_EREV = 0.0  # mV cation channel - same reversal voltage as AMPA


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
AMPA_FILE = OUTPUT_DIR / "destexhe1994_fig6_ampa_pulse.png"
GABAA_FILE = OUTPUT_DIR / "destexhe1994_gabaa_pulse.png"
NMDA_FILE = OUTPUT_DIR / "destexhe1994_nmda_pulse.png"
OVERLAY_FILE = OUTPUT_DIR / "destexhe1994_overlay_ampa_nmda_gabaa.png"
DEADTIME_FILE = OUTPUT_DIR / "destexhe1994_deadtime_block.png"


# --- Drivers -----------------------------------------------------------------
def _drive_single_pulse(syn, dt=_DT, pulse_ms=1.0, total_ms=60.0):
    """
    Hold V_pre above threshold for pulse_ms, then at rest for the remainder.
    Returns (t, R) sampled after every step.
    """
    n_pulse = round(pulse_ms / dt)
    n_total = round(total_ms / dt)
    t = np.empty(n_total)
    R = np.empty(n_total)
    time_ms = 0.0
    for i in range(n_total):
        V_pre = SPIKE_V if i < n_pulse else REST_V
        syn.step(dt, V_pre)
        time_ms += dt
        t[i] = time_ms
        R[i] = syn.R
    return t, R


def _drive_spike_train(syn, dt, onsets_ms, pulse_ms=1.0, total_ms=None):
    """Drive V_pre above threshold during [onset, onset+pulse_ms) for each onset
    in onsets_ms, at rest otherwise. Returns (t, R)."""
    if total_ms is None:
        total_ms = onsets_ms[-1] + pulse_ms + 40.0
    n_total = round(total_ms / dt)
    t = np.empty(n_total)
    R = np.empty(n_total)
    time_ms = 0.0
    for i in range(n_total):
        in_pulse = any(onset <= time_ms < onset + pulse_ms for onset in onsets_ms)
        V_pre = SPIKE_V if in_pulse else REST_V
        syn.step(dt, V_pre)
        time_ms += dt
        t[i] = time_ms
        R[i] = syn.R
    return t, R


def _decay_rate(t, R, start_ms, end_ms):
    """Fit log(R) = -beta*t + c over [start_ms, end_ms]; return fitted beta."""
    mask = (t >= start_ms) & (t <= end_ms) & (R > 0)
    slope, _ = np.polyfit(t[mask], np.log(R[mask]), 1)
    return -slope


# --- Test 1: AMPA single pulse (Fig. 6, top-left panel) ----------------------
def test_ampa_single_pulse_fig6():
    """Single presynaptic spike, 1 ms transmitter pulse, AMPA two-state kinetics.
    Fig. 6: fast rise, approximately mono-exponential decay."""
    syn = TwoStateDestexhe.excitatory(gmax=GMAX)
    t, R = _drive_single_pulse(syn)
    I = GMAX * R * (EXC_V_HOLD - syn._p.Erev)

    peak_t = float(t[int(np.argmax(R))])
    beta_fit = _decay_rate(t, R, start_ms=5.0, end_ms=40.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, -I, color="#1f77b4")  # inward-positive display convention,
    # matching Fig 6's upward EPSC
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("-PSC (pA), inward-positive")
    ax.set_title(f"Fig 6 (AMPA) - single-pulse PSC, peak@{peak_t:.2f} ms")
    fig.savefig(AMPA_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {AMPA_FILE}")

    fast_rise = peak_t <= 1.0 + 2 * _DT
    mono_exp_decay = abs(beta_fit - syn._p.Beta) / syn._p.Beta < 0.02

    passed = bool(fast_rise and mono_exp_decay)
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} Fig6 AMPA single pulse: peak@{peak_t:.3f}ms (<=~1ms),"
        f"decay beta={beta_fit:.4f}/ms vs target {syn._p.Beta:.4f}/ms -> {AMPA_FILE}"
    )

    assert passed


# --- Test 2: GABA-A single pulse (Table 1) -----------------------------------


def test_gabaa_single_pulse_table1():
    """
    Same protocol as Test 1 with GABA-A kinetics (Table 1 / gabaa.mod defaults).
    """
    syn = TwoStateDestexhe.inhibitory(gmax=GMAX)
    t, R = _drive_single_pulse(syn)
    I = GMAX * R * (INH_V_HOLD - syn._p.Erev)

    peak_t = float(t[int(np.argmax(R))])
    beta_fit = _decay_rate(t, R, start_ms=5.0, end_ms=40.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, I, color="#d62728")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("IPSC (pA)")
    ax.set_title(f"Table 1 (GABA-A) -- single-pulse IPSC, peak@{peak_t:.2f} ms")
    fig.savefig(GABAA_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {GABAA_FILE}")

    fast_rise = peak_t <= 1.0 + 2 * _DT
    mono_exp_decay = abs(beta_fit - syn._p.Beta) / syn._p.Beta < 0.02

    passed = bool(fast_rise and mono_exp_decay)
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} Table1 GABA-A single pulse@{peak_t:.3f}ms (<=:1ms), "
        f"decay beta ={beta_fit:.4f}/ms vs target {syn._p.Beta:.4f}/ms -> {GABAA_FILE}"
    )
    assert passed


# --- Test 3: NMDA single pulse (Table 1) -------------------------------------
def test_nmda_single_pulse_table():
    """
    Same protocol with NMDA rate constants (Table 1): much slower decay
    than AMPA or GABA-A. For the 2-state fit, r1 = 72 s⁻¹mM⁻¹ and r2 = 6.6 s⁻¹,
    so the decay time constant after transmitter is removed is approximatley
    1 / Beta ~ 1/6.6 s ~ 150 ms. By comparison, AMPA and GABA-A have Beta ~ 190 s⁻1
    and 180 s⁻¹, giving decay constants of about 5.3 ms and 5.6 ms, respectively.
    """
    syn = TwoStateDestexhe(
        TwoStateDestexheParams(
            gmax=GMAX, Alpha=NMDA_ALPHA, Beta=NMDA_BETA, Erev=NMDA_EREV
        )
    )
    t, R = _drive_single_pulse(syn, total_ms=400.0)
    I = GMAX * R * (EXC_V_HOLD - syn._p.Erev)

    peak_t = float(t[np.argmax(R)])
    beta_fit = _decay_rate(t, R, start_ms=20.0, end_ms=300.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, -I, color="#9467bd")  # change the inward positive rule here
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("PSC (pA)")
    ax.set_title(f"Table 1 (NMDA) - single-pulse PSC, peak@{peak_t:.2f} ms")
    fig.savefig(NMDA_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Save:{NMDA_FILE}")

    fast_rise = peak_t <= 1.0 + 2 * _DT
    mono_exp_decay = abs(beta_fit - NMDA_BETA) / NMDA_BETA < 0.02

    passed = bool(fast_rise and mono_exp_decay)
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} Table1 NMDA single pulse: peak@{peak_t:.2f}ms (<=~1ms),"
        f"decay beta={beta_fit:.5f}/ms vs target {NMDA_BETA:.5f}/ms -> {NMDA_FILE}"
    )

    assert passed


# --- Test 4: overlay of AMPA, NMDA, GABA-A -----------------------------------


def test_overlay_ampa_nmda_gabaa():
    """
    Derived validation (consistent with Table 1 and the general kinetic framework,
    not a single reproduced figure): NMDA decay must be much slower than both AMPA
    and GABA-A.
    """
    syn_ampa = TwoStateDestexhe.excitatory(gmax=GMAX)
    syn_gabaa = TwoStateDestexhe.inhibitory(gmax=GMAX)
    syn_nmda = TwoStateDestexhe(
        TwoStateDestexheParams(
            gmax=GMAX, Alpha=NMDA_ALPHA, Beta=NMDA_BETA, Erev=NMDA_EREV
        )
    )

    t_a, R_a = _drive_single_pulse(syn_ampa, total_ms=400.0)
    t_g, R_g = _drive_single_pulse(syn_gabaa, total_ms=400.0)
    t_n, R_n = _drive_single_pulse(syn_nmda, total_ms=400.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_a, R_a / R_a.max(), label="AMPA")
    ax.plot(t_g, R_g / R_g.max(), label="GABA-A")
    ax.plot(t_n, R_n / R_n.max(), label="NMDA")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("R / R_peak (normalised)")
    ax.set_xlim(0, 400)
    ax.set_title(
        "Overlay - normalised open-fraction kinetics " "(Table 1 rate constants)"
    )
    ax.legend()
    fig.savefig(OVERLAY_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OVERLAY_FILE}")

    tau_a = 1.0 / _decay_rate(t_a, R_a, 5.0, 40.0)
    tau_g = 1.0 / _decay_rate(t_g, R_g, 5.0, 40.0)
    tau_n = 1.0 / _decay_rate(t_n, R_n, 20.0, 400.0)

    passed = bool(tau_n > tau_a and tau_n > tau_g)
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} overlay kinetics: tau_AMPA={tau_a:.2f}ms, tau_GABAB={tau_g:.2f}"
        f"ms, tau_NMDA={tau_n:.2f}ms (expect NMDA >> both) -> {OVERLAY_FILE}"
    )
    assert passed


# --- Test 5 : deadtime / retransmission block --------------------------------
def test_deadtime_retransmission_block():
    """
    Implementation detail (ModelDB #18198): a trigger attempt within Deadtime
    of the last release must be ignored; one outside that window must trigger a
    fresh release.
    """
    syn = TwoStateDestexhe.excitatory(gmax=GMAX)
    Cdur, Deadtime = syn._p.Cdur, syn._p.Deadtime
    blocked_until = Cdur + Deadtime  # 2ms: earliest a retrigger is honoured

    onsets = [
        0.0,
        1.5,
        4.0,
    ]  # 1.5 ms falls inside the blocked window, 4.0 ms outside it
    blip_ms = 2 * _DT
    assert onsets[1] + blip_ms < blocked_until < onsets[2]
    t, R = _drive_spike_train(syn, _DT, onsets, pulse_ms=blip_ms, total_ms=40.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, R, color="#9467bd")
    for onset in onsets:
        ax.axvline(onset, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("R (open fraction)")
    ax.set_title("Deadtime block: spike @1.5 ms ignored, spike @4ms retriggers")
    fig.savefig(DEADTIME_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {DEADTIME_FILE}")

    decay_window = (t > Cdur + 2 * _DT) & (t < onsets[2])
    monotonic_decay = bool(np.all(np.diff(R[decay_window]) <= 1e-12))

    idx_at_retrigger = int(np.searchsorted(t, onsets[2]))
    idx_after = int(np.searchsorted(t, onsets[2] + 0.5))
    retriggered = bool(R[idx_after] > R[idx_at_retrigger])

    passed = bool(monotonic_decay and retriggered)
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} deadtime block: monotonic decay through blocked window="
        f"{monotonic_decay}, retrigger @{onsets[2]}ms works={retriggered} -> {DEADTIME_FILE}"
    )
    assert passed

# -----------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # test_ampa_single_pulse_fig6()
    # test_gabaa_single_pulse_table1()
    # test_nmda_single_pulse_table()
    # test_overlay_ampa_nmda_gabaa()
    # test_deadtime_retransmission_block()
    test_deadtime_retransmission_block()


# =============================================================================
if __name__ == "__main__":

    main()
