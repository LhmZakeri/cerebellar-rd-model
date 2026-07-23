"""
D'Angelo et al. (2001) granule cell - acceptance script.

Runs the cell under two input levels matching Neuron modelDB protocols,
https://github.com/ModelDBRepository/46839 , records voltage and Ca traces,
saves a 4-panel figure (Issue #1 acceptance criterion), plus the Fig. 6A
current-step f/I sweep and the Fig. 6B conductance-perturbation panels.

Usage:
    python scripts/run_dangelo2001_prototype.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace
from pathlib import Path

from src.models.dangelo_cell import DAngelo2001CellModel, DAngelo2001Params
from src.simulation.sim import simulate_single, simulate_sequential, count_spikes
from src.simulation.spike_metrics import firing_rate_hz

# ------------------------------------------------------------------------------
DT = 0.025          # ms
SETTLE_MS = 100.0   # ms

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

PROTOTYPE_FILE = OUTPUT_DIR / "dangelo2001_prototype.png"
FIG6A_FILE = OUTPUT_DIR / "dangelo2001_Fig6A.pdf"
FIG6B_FILE = OUTPUT_DIR / "dangelo2001_Fig6B.pdf"

PROTOTYPE_RECORD_MS = 500.0   # ms
RUNS = [
    {"label": "High input 0.05 nA", "I_ext": 0.05, "color": "#1f77b4"},
    {"label": "Low input 0.02 nA",  "I_ext": 0.02, "color": "#d62728"},
]

FIG6A_RECORD_MS = 500.0   # ms per current step
CURRENTS = np.arange(0.002, 0.022, 0.002)  # nA

FIG6B_RECORD_MS = 800.0   # ms per epoch
_defaults = DAngelo2001Params()
PANELS = [
    {
        "label": "Control",
        "currents": [0, 0.0107, 0, 0.0107, 0, 0.0107, 0],
        "params": _defaults,
    },
    {
        "label": "4× g_Na",
        "currents": [0, 0.0103, 0, 0.0103, 0, 0.0103, 0],
        "params": replace(_defaults, gnabar_Na=_defaults.gnabar_Na * 4),
    },
    {
        "label": "1.4× g_KM",
        "currents": [0, 0.0118, 0, 0.0118, 0, 0.0118, 0],
        "params": replace(_defaults, gkbar_KM=_defaults.gkbar_KM * 1.4),
    },
]


# ------------------------------------------------------------------------------
def run_prototype() -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 7), gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    fig.suptitle("D'Angelo et al. (2001) cerebellar granule cell", fontsize=13)

    for col, run in enumerate(RUNS):
        cell = DAngelo2001CellModel()
        t, V, Ca = simulate_single(cell, run["I_ext"], PROTOTYPE_RECORD_MS, DT, SETTLE_MS)
        n_spikes = count_spikes(V)
        freq = firing_rate_hz(t, V)
        color = run["color"]

        ax_v = axes[0][col]
        ax_v.plot(t, V, color=color, linewidth=0.6)
        ax_v.axhline(0.0, color="gray", linewidth=0.4, linestyle="--")
        ax_v.set_xlabel("Time (ms)")
        ax_v.set_ylabel("V (mv)")
        ax_v.set_title(f"{run['label']}\n{n_spikes} spikes -> {freq:.1f} Hz")
        ax_v.set_xlim(0, PROTOTYPE_RECORD_MS)
        ax_v.set_ylim(-90, 60)

        ax_ca = axes[1][col]
        ax_ca.plot(t, Ca * 1e3, color=color, linewidth=0.8)  # mM -> uM
        ax_ca.set_xlabel("Time (ms)")
        ax_ca.set_ylabel("[Ca²⁺]_i (uM)")
        ax_ca.set_title("Intracellular CA²⁺")
        ax_ca.set_xlim(0, PROTOTYPE_RECORD_MS)
        ax_ca.set_ylim(bottom=0)

    fig.savefig(PROTOTYPE_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {PROTOTYPE_FILE}")


def run_fig6a() -> None:
    cell = DAngelo2001CellModel()
    t, V = simulate_sequential(cell, CURRENTS, FIG6A_RECORD_MS, DT, SETTLE_MS)

    total_ms = FIG6A_RECORD_MS * len(CURRENTS)
    boundaries = np.arange(len(CURRENTS)) * FIG6A_RECORD_MS

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("D'Angelo et al. (2001) cerebellar granule cell — Fig. 6A", fontsize=13)

    ax.plot(t, V, linewidth=0.6)
    ax.axhline(0.0, color="gray", linewidth=0.4, linestyle="--")
    ax.vlines(boundaries, ymin=-90, ymax=60, color="gray", linewidth=0.4, linestyle=":")

    for x, current in zip(boundaries + FIG6A_RECORD_MS / 2, CURRENTS):
        ax.text(x, -85, f"{int(current * 1000)} pA",
                ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mv)")
    ax.set_xlim(0, total_ms)
    ax.set_ylim(-90, 60)

    fig.savefig(FIG6A_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {FIG6A_FILE}")


def run_fig6b() -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 9), sharex=False,
        gridspec_kw={"hspace": 0.45},
    )
    fig.suptitle("D'Angelo et al. (2001) cerebellar granule cell — Fig. 6B",
                 fontsize=13)

    for ax, panel in zip(axes, PANELS):
        cell = DAngelo2001CellModel(panel["params"])
        t, V = simulate_sequential(
            cell, panel["currents"], FIG6B_RECORD_MS, DT, SETTLE_MS
        )
        total_ms = FIG6B_RECORD_MS * len(panel["currents"])

        ax.plot(t, V, linewidth=0.6)
        ax.axhline(0.0, color="gray", linewidth=0.4, linestyle="--")
        ax.set_ylabel("V (mV)")
        ax.set_title(panel["label"])
        ax.set_xlim(0, total_ms)
        ax.set_ylim(-90, 60)

    axes[-1].set_xlabel("Time (ms)")

    fig.savefig(FIG6B_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {FIG6B_FILE}")


# ------------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_prototype()
    run_fig6a()
    run_fig6b()


# ==============================================================================
if __name__ == "__main__":
    main()
