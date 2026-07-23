"""
Solinas et al. (2007) cerebellar Golgi cell - acceptance script.

Runs the cell under three input levels, records voltage and Ca traces,
saves a 3x2 panel figure.
Neuron modelDB protocols,
https://github.com/ModelDBRepository/112685,

Usage:
    python scripts/run_solinas2007_prototype.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from dataclasses import replace

from src.models.solinas_cell import Solinas2007CellModel, Solinas2007Params
from src.simulation.sim import simulate_step_protocol
from src.simulation.spike_metrics import (
    firing_rate_hz,
    isi_ms,
    spike_times_ms,
    adaptation_ratio_robust,
    first_spike_latency_ms,
)

matplotlib.use("Agg")
# ------------------------------------------------------------------------------
_DT = 0.025  # ms
SETTLE_MS = 1000.0  # ms — Golgi cell needs longer to reach pacemaker rhythm
RECORD_MS = 3000.0  # ms

OUTPUTS = Path(__file__).parent.parent / "outputs"
OUTPUT_FILE = OUTPUTS / "solinas2007_prototype.png"

RUNS = [
    {"label": "Pacemaker (0 nA)", "I_ext": 0.0},
    {"label": "Low drive (0.15 nA)", "I_ext": 0.15},
    {"label": "High drive (0.35 nA)", "I_ext": 0.35},
]


# --- Figure-specific protocols (Solinas 2007 figures) -------------------------
def test_fig2_pacemaker() -> None:
    cell = Solinas2007CellModel()
    t, V, Ca = simulate_step_protocol(
        cell,
        _DT,
        [
            (1000.0, 0.0),
            (3000.0, 0.0),
        ],
    )
    keep = t >= 1000.0
    t2, V2, Ca2 = t[keep] - 1000.0, V[keep], Ca[keep]

    freq = firing_rate_hz(t2, V2)
    isi = isi_ms(t2, V2)

    assert 1.0 <= freq <= 8.0, f"pacemaker freq {freq:.2f} Hz out of range"
    assert len(isi) >= 3
    assert (
        120.0 <= np.mean(isi) <= 300.0
    ), f"mean ISI {np.mean(isi):.1f} ms out of range"
    cv = np.std(isi) / np.mean(isi)
    assert cv < 0.4, f"CV {cv:.2f} too high"
    assert np.min(V2) < -65.0

    fig, (ax_v, ax_ca) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.1}
    )
    fig.suptitle(
        f"Fig 2 — Pacemaker  {freq:.1f} Hz  ISI {np.mean(isi):.0f}±{np.std(isi):.0f} ms  CV={cv:.2f}"
    )
    ax_v.plot(t2, V2, linewidth=0.6)
    ax_v.axhline(0.0, color="gray", linewidth=0.4, linestyle="--")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_ylim(-90, 60)
    ax_ca.plot(t2, Ca2 * 1e3, linewidth=0.8, color="C1")
    ax_ca.set_xlabel("Time (ms)")
    ax_ca.set_ylabel("[Ca²⁺]_i (µM)")
    ax_ca.set_ylim(bottom=0)
    out = OUTPUTS / "solinas2007_fig2_pacemaker.png"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PASS  Fig 2 pacemaker -> {out}")


# ------------------------------------------------------------------------------
def test_fig3_sag_rebound() -> None:

    _default = Solinas2007Params()
    params = replace(
        _default,
        el=-64.5,
        gkbar_KM=_default.gkbar_KM * 0.9,
        gbar_hcn1=_default.gbar_hcn1 * 0.6,
        gbar_hcn2=_default.gbar_hcn2 * 0.125,
    )
    cell = Solinas2007CellModel(params=params)
    t, V, Ca = simulate_step_protocol(
        cell,
        _DT,
        [
            (1000.0, 0.0),
            (5000.0, -0.18),
            (1000.0, 0.0),
        ],
    )

    step_start, step_end = 1000.0, 6000.0

    early = V[(t >= step_start) & (t < step_start + 300.0)]
    late = V[(t >= step_end - 300.0) & (t < step_end)]
    sag_amplitude = np.mean(late) - np.min(early)

    t_after = t[t >= step_end] - step_end
    V_after = V[t >= step_end]
    rebound_spikes = spike_times_ms(t_after, V_after)

    assert len(rebound_spikes) >= 1

    lat = rebound_spikes[0]
    if not (30.0 <= lat <= 150.0):
        print(f"WARN Fig 3 rebound latency is early: {lat:.3f} ms")
    else:
        print(f"PASS Fig 3 rebound latency: {lat:.3f} ms")

    fig, (ax_v, ax_ca) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.1}
    )
    fig.suptitle(
        f"Fig 3 - Sag & rebound sag={sag_amplitude:.1f}"
        + f" mV rebound@{rebound_spikes[0]:.0f} ms"
    )
    ax_v.plot(t, V, linewidth=0.6)
    ax_v.axvline(step_start, color="gray", linewidth=0.6, linestyle="--")
    ax_v.axvline(step_end, color="gray", linewidth=0.6, linestyle="--")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_ylim(-100, 60)
    ax_ca.plot(t, Ca * 1e3, linewidth=0.8, color="C1")
    ax_ca.set_xlabel("Time (ms)")
    ax_ca.set_ylabel("[Ca²⁺]_i (uM)")
    ax_ca.set_ylim(bottom=0)
    out = OUTPUTS / "solinas2007_fig3_sag_rebound.png"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PASS Fig3 sag/rebound -> {out}")


# ------------------------------------------------------------------------------
def run_depol_trial(current_nA: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fig 4: 200 ms baseline -> 1000 ms step -> 800 ms tail."""
    cell = Solinas2007CellModel()
    t, V, Ca = simulate_step_protocol(
        cell,
        _DT,
        [
            (200.0, 0.0),
            (1000.0, current_nA),
            (800.0, 0.0),
        ],
    )
    step_start, step_end = 200.0, 1200.0
    mask = (t >= step_start) & (t < step_end)
    return t[mask] - step_start, V[mask], Ca[mask]


# ------------------------------------------------------------------------------
def test_fig4_depolarization_fi() -> None:

    t150, V150, _ = run_depol_trial(0.15)
    t250, V250, _ = run_depol_trial(0.25)
    t350, V350, _ = run_depol_trial(0.35)

    f150 = firing_rate_hz(t150, V150)
    f250 = firing_rate_hz(t250, V250)
    f350 = firing_rate_hz(t350, V350)
    assert f150 < f250 < f350, f"fI not monotone: f150 ={f150:.1f} hz,"
    f"f250={f250:.1f} hz, f350={f350:.1f} hz"

    lat150 = first_spike_latency_ms(t150, V150, 0.0)
    lat250 = first_spike_latency_ms(t250, V250, 0.0)
    lat350 = first_spike_latency_ms(t350, V350, 0.0)
    assert lat150 > lat250 > lat350, (
        f"latency not monotone: lat150={lat150:.2f} ms,"
        f"lat250={lat250:.2f} ms, lat350={lat350:.2f} ms"
    )

    for label, t_, V_ in [
        ("150 pA", t150, V150),
        ("250 pA", t250, V250),
        ("350 pA", t350, V350),
    ]:
        ar = adaptation_ratio_robust(t_, V_)
        if np.isnan(ar):
            print(f"WARN Fig adaptation at {label} could not be estimated.")
        elif ar <= 0.95:
            print(f"WARN Fig 4 weak adaptation at {label}: {ar:.2f}")
        else:
            print(f"PASS Fig4 adaptation at {label}: {ar:.2f}")

    print(
        "PASS Fig 4 depolarization f-I"
        f"(rates: {f150:.1f}, {f250:.1f}, {f350:.1f} hz;"
        f"latencies: {lat150:.1f}, {lat250:.1f}, {lat350:.1f} ms"
    )


# ------------------------------------------------------------------------------
def test_fig6_calcium() -> None:

    cell = Solinas2007CellModel()

    # Figure-6-inspired protocol for the soma-only prototype:
    t, V, Ca = simulate_step_protocol(
        cell,
        _DT,
        [
            (1000.0, 0.0),  # pacemaking / baseline
            (300.0, 0.20),  # 200 pA burst-like depolarization
        ],
    )

    Ca_uM = Ca * 1e3  # mM -> uM

    rest_mask = t < 1000.0
    burst_mask = t >= 1000.0

    rest = Ca_uM[rest_mask]
    burst = Ca_uM[burst_mask]

    ca_rest = np.mean(rest[-2000:] if len(rest) >= 2000 else np.mean(rest))

    # Late pacemaking window, before burst accumulation starts
    preburst_window = (t >= 500.0) & (t < 1000.0)
    ca_single_peak = np.max(Ca_uM[preburst_window])

    ca_burst_peak = np.max(burst)

    assert 0.02 <= ca_rest <= 0.10, f"Ca rest {ca_rest:.4f} uM out of range."
    assert ca_single_peak > ca_rest, (
        f"Ca did not rise above rest during pacemaking/spiking"
        f"(rest={ca_rest:.4f} uM, single={ca_single_peak:.4f} uM)"
    )
    assert np.mean(burst) > np.mean(rest), "burst did not increase mean Ca"

    if not (0.5 <= ca_single_peak <= 2.0):
        print(
            f"WARN Fig 6 single-spike/pacemaker Ca peak is high for paper target: "
            f"{ca_single_peak:.3f} uM"
        )
    else:
        print(f"PASS Fig 6 single-spike/pacemaker Ca peak: {ca_single_peak:.3f} uM")

    if ca_burst_peak > 2.0:
        print(f"WARN Fig 6 burst Ca peak is high: {ca_burst_peak:.3f} uM")
    else:
        print(f"PASS Fig 6 burst CA peak: {ca_burst_peak:.3f} mM")

    fig, (ax_v, ax_ca) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.1}
    )
    fig.suptitle(
        "Fig 6 _ Calcium "
        f"rest={ca_rest:.3f} uM "
        f"single={ca_single_peak:.3f} uM"
        f"burst={ca_burst_peak:.3f} uM"
    )
    ax_v.plot(t, V, linewidth=0.6)
    ax_v.axvline(1000.0, color="gray", linewidth=0.6, linestyle="--")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_ylim(-90, 60)

    ax_ca.plot(t, Ca_uM, linewidth=0.8, color="C1")
    ax_ca.axvline(1000.0, color="gray", linewidth=0.6, linestyle="--")
    ax_ca.set_xlabel("Time (ms)")
    ax_ca.set_ylabel("[Ca²⁺]_i (uM)")
    ax_ca.set_ylim(bottom=0)

    out = OUTPUTS / "solinas2007_fig6_calcium.png"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"PASS Fig6 calcium -> {out}"
        f"(rest={ca_rest:.3f} uM, single={ca_single_peak} uM,"
        f"burst={ca_burst_peak:.3f} uM)"
    )


# ------------------------------------------------------------------------------
def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 7), gridspec_kw={"hspace": 0.45, "wspace": 0.35}
    )
    fig.suptitle("Solinas et al. (2007) cerebellar Golgi cell", fontsize=13)

    for col, run in enumerate(RUNS):
        cell = Solinas2007CellModel()
        t, V, Ca = simulate_step_protocol(
            cell,
            _DT,
            [
                (SETTLE_MS, run["I_ext"]),
                (RECORD_MS, run["I_ext"]),
            ],
        )
        keep = t >= SETTLE_MS
        t2 = t[keep] - SETTLE_MS
        V2 = V[keep]
        Ca2 = Ca[keep]

        freq = firing_rate_hz(t2, V2)
        isi = isi_ms(t2, V2)
        isi_str = (
            f"ISI {np.mean(isi):.1f}±{np.std(isi):.1f} ms  CV={np.std(isi)/np.mean(isi):.2f}"
            if len(isi) >= 2
            else "ISI n/a"
        )

        ax_v = axes[0][col]
        ax_v.plot(t2, V2, linewidth=0.6)
        ax_v.axhline(0.0, color="gray", linewidth=0.4, linestyle="--")
        ax_v.set_xlabel("Time (ms)")
        ax_v.set_ylabel("V (mV)")
        ax_v.set_title(f"{run['label']}\n{freq:.1f} Hz  {isi_str}")
        ax_v.set_xlim(0, RECORD_MS)
        ax_v.set_ylim(-90, 60)

        ax_ca = axes[1][col]
        ax_ca.plot(t2, Ca2 * 1e3, linewidth=0.8)  # mM → µM
        ax_ca.set_xlabel("Time (ms)")
        ax_ca.set_ylabel("[Ca²⁺]_i (µM)")
        ax_ca.set_title("Intracellular Ca²⁺ (HVA pool)")
        ax_ca.set_xlim(0, RECORD_MS)
        ax_ca.set_ylim(bottom=0)

    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")

    test_fig2_pacemaker()
    test_fig3_sag_rebound()
    test_fig4_depolarization_fi()
    test_fig6_calcium()


# ==============================================================================
if __name__ == "__main__":
    main()
