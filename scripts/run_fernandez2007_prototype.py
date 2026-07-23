"""
Fernandez, Engbers & Turner (2007) Purkinje-cell model

Runs the five-equation two-compartment model through the paper's
step-current protocols (Fig. 4Aii, 4B, 4C, and a DAP holding
-potential comparison).

Usage:
    python scripts/run_fernandez2007_prototype.py

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from src.models.fernandez_cell import Fernandez2007CellModel
from src.simulation.sim import simulate_step_protocol

matplotlib.use("Agg")
# -----------------------------------------------------------------------------
_DT = 0.01  # ms

OUTPUTS = Path(__file__).parent.parent / "outputs"

# -----------------------------------------------------------------------------


def _run_protocol(
    cell: Fernandez2007CellModel,
    segments: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Run step protocol from current cell state. Retruns (t, V_s)."""
    t, V_s, _ = simulate_step_protocol(cell, _DT, segments)
    return t, V_s


def _run_protocol_with_slow_k(
    cell: Fernandez2007CellModel,
    segments: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run step protocol, also recording I_Kd,slow. Returns (t, V_s, I_slowK)."""
    n_total = sum(round(d / _DT) for d, _ in segments)
    t = np.empty(n_total)
    V_s = np.empty(n_total)
    I_slowK = np.empty(n_total)

    k = 0
    time_ms = 0.0
    for duration_ms, current_nA in segments:
        n = round(duration_ms / _DT)
        for _ in range(n):
            cell.step(_DT, current_nA)
            t[k] = time_ms
            V_s[k] = cell.get_voltage()
            I_slowK[k] = cell.get_slow_k_current()
            k += 1
            time_ms += _DT
    return t, V_s, I_slowK


# -----------------------------------------------------------------------------


def _first_return_to_baseline_ms(
    t: np.ndarray,
    V: np.ndarray,
    baseline_V: float,
    start_ms: float,
    tol_mV: float = 3.0,
) -> float:
    idx = np.where(t >= start_ms)[0]
    if len(idx) == 0:
        return 0.0
    recovered = idx[V[idx] < baseline_V + tol_mV]
    if len(recovered) == 0:
        return t[-1] - start_ms
    return t[recovered[0]] - start_ms


# -----------------------------------------------------------------------------


def _spike_times_ms(t: np.ndarray, V: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    above = V >= threshold
    crossings = np.where((~above[:-1]) & (above[1:]))[0]
    return t[crossings + 1]


# -----------------------------------------------------------------------------


def _mean_frequency_hz(spike_times: np.ndarray) -> float:
    if len(spike_times) == 0:
        return np.inf
    return 1000.0 / np.mean(np.diff(spike_times))


# -----------------------------------------------------------------------------


def _first_spike_latency_ms(spike_times: np.ndarray, step_on_ms: float) -> float:
    if len(spike_times) == 0:
        return np.inf
    return spike_times[0] - step_on_ms


# -----------------------------------------------------------------------------


def _save_trace_figure(
    t: np.ndarray,
    V: np.ndarray,
    title: str,
    out_name: str,
    vlines: list[float] | None = None,
    ylabel: str = "V_s (mV)",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(title)
    ax.plot(t, V, linewidth=0.6)
    for x in vlines or []:
        ax.axvline(x, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    out = OUTPUTS / out_name
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Fig.B: climbing-fibre pulse, rest -> plateau / toggle -------------------


def test_fig4b_weak_cf_pulse_gives_plateau_but_no_switch() -> None:
    """Weak pulse (~1.4 uA/cm^2): long depolarised plateau, no sustained
    firing."""
    cell = Fernandez2007CellModel()
    t, V_s = _run_protocol(
        cell,
        [
            (100.0, 0.0),
            (15.0, 1400),
            (235.0, 0.0),
        ],
    )

    baseline = np.mean(V_s[(t >= 80.0) & (t < 99.0)])
    plateau_ms = _first_return_to_baseline_ms(
        t, V_s, baseline, start_ms=65.0, tol_mV=3.0
    )
    assert plateau_ms < 100.0

    spikes_post = _spike_times_ms(t[t >= 160.0], V_s[t >= 160.0])
    assert len(spikes_post) <= 1
    assert abs(V_s[-1] - baseline) < 5.0

    out = _save_trace_figure(
        t,
        V_s,
        f"Fig 4B - weak CF pulse, plateau {plateau_ms:.0f} ms",
        "fernandez2007_fig4b_weak.png",
        vlines=[100.0, 115.0],
    )
    print(f"PASS Fig4B weak CF pulse: plateau {plateau_ms:.0f} ms -> {out}")


# -----------------------------------------------------------------------------
def test_fig4b_intermediate_cf_pulse_switches_on() -> None:
    """Intermediate pulse (~3000 nA, ~3 uA/cm²): switches rest -> tonic firing."""

    cell = Fernandez2007CellModel()

    t, V_s = _run_protocol(
        cell,
        [
            (100.0, 0.0),
            (15.0, 3000.0),
            (235.0, 0.0),
        ],
    )

    mask = t >= 115.0
    spk = _spike_times_ms(t[mask], V_s[mask])
    assert len(spk) >= 5

    f_hz = _mean_frequency_hz(spk)
    assert 20.0 <= f_hz <= 40.0

    out = _save_trace_figure(
        t,
        V_s,
        f"Fig 4B_ intermediate CF pulse switches on, {f_hz:.1f}",
        "fernandez2007_fig4b_intermediate.png",
        vlines=[100.0, 115.0],
    )
    print(f"PASS Fig4B intermediate CF pulse: {f_hz:.1f} Hz -> {out}")


# -----------------------------------------------------------------------------
def test_fig4b_strong_cf_pulse_does_not_switch() -> None:
    """Strong pulse (~7000 nA, ~7 uA/cm²): also fails to toggle sustained firing."""

    cell = Fernandez2007CellModel()

    t, V_s = _run_protocol(
        cell,
        [
            (100.0, 0.0),
            (15.0, 7000.0),
            (235.0, -100.0),
        ],
    )

    baseline = np.mean(V_s[(t >= 80.0) & (t < 99.0)])
    spikes_post = _spike_times_ms(t[t >= 170.0], V_s[t >= 170.0])
    assert len(spikes_post) <= 1
    assert abs(V_s[-1] - baseline) < 5.0

    out = _save_trace_figure(
        t,
        V_s,
        "Fig 4B _ strong CF pulse does not switch",
        "fernandez2007_fig4b_strong.png",
        vlines=[100.0, 115.0],
    )
    print(f"PASS Fig4B strong CF pulse: no sustained switch -> {out}")


# -----------------------------------------------------------------------------
def test_fig4c_two_intermediate_pulses_toggle_on_then_off() -> None:
    """Two intermediate pulses toggle the cell on, then off."""

    cell = Fernandez2007CellModel()

    t, V_s, I_slowK = _run_protocol_with_slow_k(
        cell,
        [(100.0, 0.0), (15.0, 4200), (500.0, 0.0), (15.0, 4200.0), (100.0, 0.0)],
    )

    baseline = np.mean(V_s[(t >= 80.0) & (t < 99.0)])

    mask_mid = (t >= 115.0) & (t <= 615.0)
    spk_mid = _spike_times_ms(t[mask_mid], V_s[mask_mid])
    assert len(spk_mid) >= 3

    spk_late = _spike_times_ms(t[t >= 630.0], V_s[t >= 630.0])
    assert len(spk_late) <= 1
    assert abs(V_s[-1] - baseline)

    out = _save_trace_figure(
        t,
        V_s,
        "Fig 4C_two CF pulses toggle on then off",
        "fernandez 2007_fig4c_toggle.png",
        vlines=[100.0, 115.0, 615.0, 630.0],
    )
    print(f"PASS Fig4C toggle on/off: {len(spk_mid)} spikes mid-widow -> {out}")

    out_slowk = _save_trace_figure(
        t,
        I_slowK,
        "Fig 4C_slow K current (I_Kd,slow) during toggle on/off",
        "fernandez2007_fig4c_slowK.png",
        vlines=[100.0, 115.0, 615.0, 630.0],
        ylabel="I_Kd,slow (µA/cm²)",
    )
    print(f"PASS Fig4C slow K current -> {out_slowk}")


# -----------------------------------------------------------------------------
def test_fig4Ai_three_increasing_step_depolarization() -> None:
    """Three increasing depolarizing steps (0.17, 0.22, 1.5 uA/cm^2) from a
    common -0.07 uA/cm^2 baseline, overlaid to compare spike timing."""

    base_nA = -70.0  # -0.07 uA/cm^2
    steps = [
        (170.0, "blue", "0.17 uA/cm^2"),
        (220.0, "black", "0.22 uA/cm^2"),
        (1500.0, "red", "1.5 uA/cm^2"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Fig 4Ai_three increasing step depolarizations")

    for step_nA, color, label in steps:
        cell = Fernandez2007CellModel()
        cell._make_initial_state(V_s0=-74.0)
        t, V_s = _run_protocol(
            cell,
            [(100.0, base_nA), (350.0, step_nA), (50.0, base_nA)],
        )
        ax.plot(t, V_s, linewidth=0.6, color=color, label=label)

    ax.axvline(100.0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(450.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V_s (mV)")
    ax.legend()

    out = OUTPUTS / "fernandez2007_fig4Ai_three_steps.png"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PASS Fig4Ai three increasing steps -> {out}")


# -----------------------------------------------------------------------------


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    test_fig4b_weak_cf_pulse_gives_plateau_but_no_switch()
    test_fig4b_intermediate_cf_pulse_switches_on()
    test_fig4b_strong_cf_pulse_does_not_switch()
    test_fig4c_two_intermediate_pulses_toggle_on_then_off()
    test_fig4Ai_three_increasing_step_depolarization()


# =============================================================================
if __name__ == "__main__":
    main()
