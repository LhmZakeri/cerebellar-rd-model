"""
Mitry, Alexander, Farjami, Bowie & Khadra (2020) cerebellar stellate-cell
model — Phase 1 Python prototype.

Usage:
    python scripts/run_mitry2020_prototype.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.models.mitry_cell import MitryStellateCellModel, MitryStellateParams
from src.simulation.sim import simulate_step_protocol
from src.simulation.spike_metrics import spike_times_ms, firing_rate_hz

# ------------------------------------------------------------------------------
DT = 0.01  # ms

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

GATING_FILE = OUTPUT_DIR / "mitry2020_gating_curves.png"
PROTOTYPE_FILE = OUTPUT_DIR / "mitry2020_prototype.png"
SINGLE_SPIKE_FILE = OUTPUT_DIR / "mitry2020_single_spike.png"

VARIANTS = [
    ("pre-runup", MitryStellateParams.pre_runup(), "#1f77b4"),
    ("post-runup", MitryStellateParams.post_runup(), "#d62728"),
]

V_RANGE = np.linspace(-100.0, 20.0, 400)


# ------------------------------------------------------------------------------
def _estimate_rheobase(
    params: MitryStellateParams,
    lo: float = 0.05,
    hi: float = 0.4,
    pulse_ms: float = 300.0,
    iters: int = 25,
) -> float:
    """Bisect the smallest sustained current (nA) that crosses 0 mV within
    pulse_ms, given lo stays subthreshold and hi fires."""

    def fires(I_ext: float) -> bool:
        cell = MitryStellateCellModel(params)
        _, V, _ = simulate_step_protocol(cell, DT, [(pulse_ms, I_ext)])
        return bool(V.max() > 0.0)

    assert not fires(lo) and fires(hi), "rheobase bisection bounds must bracket threshold"
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fires(mid):
            hi = mid
        else:
            lo = mid
    return hi


# --- Gating-curve comparison --------------------------------------------------


def run_gating_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Mitry et al. (2020) stellate cell — pre- vs post-runup gating curves")

    for label, params, color in VARIANTS:
        cell = MitryStellateCellModel(params)
        m_inf = np.array([cell._m_inf(V) for V in V_RANGE])
        h_inf = np.array([cell._h_inf(V) for V in V_RANGE])
        nA_inf = np.array([cell._nA_inf(V) for V in V_RANGE])
        hA_inf = np.array([cell._hA_inf(V) for V in V_RANGE])

        axes[0].plot(V_RANGE, m_inf, color=color, linewidth=1.2, label=f"{label} m_inf")
        axes[0].plot(V_RANGE, h_inf, color=color, linewidth=1.2, linestyle="--", label=f"{label} h_inf")
        axes[1].plot(V_RANGE, nA_inf, color=color, linewidth=1.2, label=f"{label} n_A,inf")
        axes[1].plot(V_RANGE, hA_inf, color=color, linewidth=1.2, linestyle="--", label=f"{label} h_A,inf")

    axes[0].set_title("I_Na steady-state curves")
    axes[1].set_title("I_A steady-state curves")
    for ax in axes:
        ax.set_xlabel("V (mV)")
        ax.set_ylabel("Steady-state value")
        ax.legend(fontsize=7)

    fig.savefig(GATING_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {GATING_FILE}")


# --- Voltage trace at 1.5x rheobase (acceptance figure) -----------------------


def run_prototype() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Mitry et al. (2020) stellate cell — voltage trace at 1.5x rheobase")

    for ax, (label, params, color) in zip(axes, VARIANTS):
        rheobase = _estimate_rheobase(params)
        I_ext = 1.5 * rheobase

        cell = MitryStellateCellModel(params)
        t, V, _ = simulate_step_protocol(cell, DT, [(50.0, 0.0), (300.0, I_ext)])
        n_spikes = len(spike_times_ms(t, V))

        ax.plot(t, V, color=color, linewidth=0.6)
        ax.axhline(0.0, color="gray", linewidth=0.4, linestyle="--")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("V (mV)")
        ax.set_title(
            f"{label}: rheobase={rheobase*1000:.1f} pA\n"
            f"I={I_ext*1000:.1f} pA, {n_spikes} spike(s)"
        )
        ax.set_xlim(0, 350)
        ax.set_ylim(-90, 60)

    fig.savefig(PROTOTYPE_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PROTOTYPE_FILE}")


# --- Brief near-rheobase pulse: clean single spike -----------------------------


def run_single_spike_pulse() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("Mitry et al. (2020) stellate cell — 10 ms pulse at 1.2x rheobase")

    for label, params, color in VARIANTS:
        rheobase = _estimate_rheobase(params)
        I_ext = 1.2 * rheobase

        cell = MitryStellateCellModel(params)
        t, V, _ = simulate_step_protocol(
            cell, DT, [(50.0, 0.0), (10.0, I_ext), (200.0, 0.0)]
        )
        n_spikes = len(spike_times_ms(t, V))
        ax.plot(t, V, color=color, linewidth=0.7, label=f"{label} ({n_spikes} spike)")

    ax.axvline(50.0, color="gray", linewidth=0.4, linestyle="--")
    ax.axvline(60.0, color="gray", linewidth=0.4, linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_xlim(0, 260)
    ax.set_ylim(-90, 60)
    ax.legend(fontsize=8)

    fig.savefig(SINGLE_SPIKE_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {SINGLE_SPIKE_FILE}")


# --- Rheobase comparison (printed, per source-doc curve shifts) --------------


def print_rheobase_comparison() -> None:
    rheobases = {label: _estimate_rheobase(params) for label, params, _ in VARIANTS}
    pre_r, post_r = rheobases["pre-runup"], rheobases["post-runup"]
    status = "PASS" if post_r < pre_r else "FAIL"
    print(
        f"{status} rheobase: pre-runup={pre_r * 1000:.1f} pA, "
        f"post-runup={post_r * 1000:.1f} pA "
        f"(post-runup must be lower, per the shifted I_Na/I_A curves)"
    )


# ------------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_gating_curves()
    run_prototype()
    run_single_spike_pulse()
    print_rheobase_comparison()


# ==============================================================================
if __name__ == "__main__":
    main()
