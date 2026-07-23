"""Trajectory-divergence utilities for twin-simulation sensitive-dependence
(chaos) tests: given two voltage recordings of the same cell population that
started from a tiny initial-condition perturbation, measure how far apart
they drift over time and whether that separation grows exponentially
(log-linear) before saturating -- the standard operational signature of
sensitive dependence on initial conditions (DESIGN.md).
"""
from __future__ import annotations

import numpy as np


def trajectory_distance(V_A: np.ndarray, V_B: np.ndarray) -> np.ndarray:
    """L2 norm across cells at each recorded timestep: D(t) = ||V_A(t) -
    V_B(t)||. V_A, V_B: (n_frames, n_cells), same shape. Returns (n_frames,)."""
    return np.linalg.norm(V_A - V_B, axis=1)


def log_divergence_growth_rate(
    D_t: np.ndarray,
    t_ms: np.ndarray,
    fit_start_ms: float,
    fit_end_ms: float,
) -> tuple[float, np.ndarray]:
    """Linear fit of log(D(t)) vs t over [fit_start_ms, fit_end_ms) -- the
    slope is a Lyapunov-exponent-like growth rate (per ms): positive and
    roughly constant over a real pre-saturation window is the operational
    signature of sensitive dependence on initial conditions. This function
    does NOT auto-detect the pre-saturation window -- pass it explicitly
    (chosen by inspecting the full log(D(t)) curve, e.g. in the saved
    figure) rather than an automatic saturation-detector, since "before
    saturation" is a judgment call this codebase's convention (per the
    protocol this backs) treats as a human-in-the-loop decision, not a fully
    automated one.

    D_t <= 0 entries (can only happen if V_A and V_B are bit-identical at
    that frame -- e.g. before the perturbation has propagated at all) are
    dropped before the log/fit, since log(0) is undefined.

    Returns (slope_per_ms, log_D_fit_window) -- slope_per_ms is nan if fewer
    than 2 valid points fall in the window.
    """
    mask = (t_ms >= fit_start_ms) & (t_ms < fit_end_ms) & (D_t > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan"), np.array([])
    log_D = np.log(D_t[mask])
    slope, _ = np.polyfit(t_ms[mask], log_D, 1)
    return float(slope), log_D
