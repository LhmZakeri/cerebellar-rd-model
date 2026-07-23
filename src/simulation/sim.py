"""Generic simulation engine for single-compartment ionic cell models.

All functions accept a pre-constructed CellModel instance, making them
independent of any specific ionic model.
"""
from __future__ import annotations

import numpy as np

from src.models.cell_model import CellModel
from src.simulation.spike_metrics import spike_count as _spike_count


def simulate_single(
    cell: CellModel,
    I_ext: float,
    record_ms: float,
    dt: float,
    settle_ms: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run one constant-current episode.

    Returns (t, V, Ca) where Ca is None if the model has no get_calcium().
    """
    n_settle = round(settle_ms / dt)
    n_record = round(record_ms / dt)

    for _ in range(n_settle):
        cell.step(dt, 0.0)

    t = np.arange(n_record) * dt
    V = np.empty(n_record)
    _has_ca = hasattr(cell, "get_calcium")
    Ca = np.empty(n_record) if _has_ca else None
    for i in range(n_record):
        cell.step(dt, I_ext)
        V[i] = cell.get_voltage()
        if _has_ca:
            Ca[i] = cell.get_calcium()

    return t, V, Ca


def simulate_step_protocol(
    cell: CellModel,
    dt: float,
    segments: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run a sequence of (duration_ms, current_nA) segments back-to-back.

    Returns (t, V, Ca) where Ca is None if the model has no get_calcium().
    """
    n_total = sum(round(d / dt) for d, _ in segments)
    t = np.empty(n_total)
    V = np.empty(n_total)
    _has_ca = hasattr(cell, "get_calcium")
    Ca = np.empty(n_total) if _has_ca else None

    k = 0
    time_ms = 0.0
    for duration_ms, current_nA in segments:
        n = round(duration_ms / dt)
        for _ in range(n):
            cell.step(dt, current_nA)
            t[k] = time_ms
            V[k] = cell.get_voltage()
            if _has_ca:
                Ca[k] = cell.get_calcium()
            k += 1
            time_ms += dt
    return t, V, Ca


def simulate_sequential(
    cell: CellModel,
    currents: list[float] | np.ndarray,
    record_ms: float,
    dt: float,
    settle_ms: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run current epochs back-to-back on one cell. Returns (t, V)."""
    n_settle = round(settle_ms / dt)
    n_record = round(record_ms / dt)
    n_total = n_record * len(currents)

    for _ in range(n_settle):
        cell.step(dt, 0.0)

    t_step = np.arange(n_record) * dt
    t = np.empty(n_total)
    V = np.empty(n_total)
    for i, current in enumerate(currents):
        offset = i * n_record
        t[offset : offset + n_record] = t_step + offset * dt
        for j in range(n_record):
            cell.step(dt, current)
            V[offset + j] = cell.get_voltage()

    return t, V


def count_spikes(V: np.ndarray, threshold: float = 0.0) -> int:
    """Count threshold crossings (rising edge) in a recorded voltage trace."""
    return _spike_count(V, threshold)
