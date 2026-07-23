"""Spike-train analysis utilities: single-cell (t, V) functions, plus
population-level variants (raster/PSTH/power-spectrum/pairwise-correlation)
for a (n_frames, n_cells) voltage array, added for DESIGN.md.

Every spike-detection function in this module (and formerly 4 independent
copies elsewhere in this repo: sim.py::count_spikes,
pilot_probe_coupling.py::_pooled_isi_cv, discovery_base_run.py::_spike_count)
is built on the same rising-edge-crossing primitive, _rising_edge_mask --
consolidated here rather than left duplicated.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def _rising_edge_mask(above: np.ndarray) -> np.ndarray:
    """True at each index i>0 where above[i] is True and above[i-1] is False
    -- an upward threshold crossing. Generic over the leading axis: works
    identically for a 1D single-cell trace or a 2D (n_frames, n_cells) array
    (crossings are then per-cell, independent per column)."""
    return above[1:] & ~above[:-1]


def spike_times_ms(
    t: np.ndarray, V: np.ndarray, threshold: float = 0.0
) -> np.ndarray:
    """Return times (ms) of each upward threshold crossing."""
    above = V > threshold
    idx = np.where(_rising_edge_mask(above))[0] + 1
    return t[idx]


def isi_ms(t: np.ndarray, V: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Interspike intervals in ms."""
    return np.diff(spike_times_ms(t, V, threshold))


def firing_rate_hz(t: np.ndarray, V: np.ndarray, threshold: float = 0.0) -> float:
    """Mean firing rate over the full recording window."""
    st = spike_times_ms(t, V, threshold)
    duration_s = (t[-1] - t[0]) * 1e-3
    return len(st) / duration_s if duration_s > 0 else 0.0


def adaptation_ratio(t: np.ndarray, V: np.ndarray, threshold: float = 0.0) -> float:
    """Last ISI / first ISI; > 1 means spike-frequency adaptation."""
    isi = isi_ms(t, V, threshold)
    if len(isi) < 2:
        return np.nan
    return isi[-1] / isi[0]


def adaptation_ratio_robust(
    t: np.ndarray, V: np.ndarray, threshold: float = 0.0
) -> float:
    """mean(last-2 ISIs) / mean(ISIs[1:3]); skips the first ISI."""
    isi = isi_ms(t, V, threshold)
    if len(isi) < 5:
        return np.nan
    early = np.mean(isi[1:3])
    late = np.mean(isi[-2:])
    return late / early


def first_spike_latency_ms(
    t: np.ndarray, V: np.ndarray, step_start_ms: float, threshold: float = 0.0
) -> float:
    """Time from step_start_ms to first spike crossing threshold."""
    st = spike_times_ms(t, V, threshold)
    st = st[st >= step_start_ms]
    return np.nan if len(st) == 0 else st[0] - step_start_ms


# ------------------------------------------------------------------------------
# Population-level (multi-cell) variants, added for DESIGN.md. V is
# (n_frames, n_cells) throughout this section -- one column per cell.
# ------------------------------------------------------------------------------


def spike_count(V: np.ndarray, threshold: float = 0.0) -> int:
    """Count threshold crossings (rising edge) in a single 1D voltage trace."""
    above = V > threshold
    return int(np.sum(_rising_edge_mask(above)))


def population_spike_count(V: np.ndarray, threshold: float = 0.0) -> int:
    """Total rising-edge threshold crossings pooled across every cell in a
    (n_frames, n_cells) voltage array -- a coarse population activity count,
    not a rate."""
    above = V > threshold
    return int(np.sum(_rising_edge_mask(above)))


def population_spike_times_ms(
    t_ms: np.ndarray, V: np.ndarray, threshold: float = 0.0
) -> list[np.ndarray]:
    """Per-cell spike times (ms) -- the raw raster data. V: (n_frames,
    n_cells). Returns a list of length n_cells, each element the sorted
    spike-time array for that cell (may be empty)."""
    above = V > threshold
    rising = _rising_edge_mask(above)
    return [t_ms[np.nonzero(rising[:, c])[0] + 1] for c in range(V.shape[1])]


def pooled_isi_cv(V: np.ndarray, dt_ms: float, threshold: float = 0.0) -> tuple[float, int]:
    """V: (n_frames, n_cells) voltage trace. Pools inter-spike intervals
    (rising-edge threshold crossings) across every cell into one
    distribution and returns (CV, n_isis). CV is nan if fewer than 2 pooled
    ISIs were observed (not enough spikes to say anything)."""
    above = V > threshold
    rising = _rising_edge_mask(above)
    all_isis = []
    for c in range(V.shape[1]):
        spike_steps = np.nonzero(rising[:, c])[0]
        if len(spike_steps) >= 2:
            all_isis.append(np.diff(spike_steps) * dt_ms)
    if not all_isis:
        return float("nan"), 0
    isis = np.concatenate(all_isis)
    if len(isis) < 2 or isis.mean() == 0:
        return float("nan"), len(isis)
    return float(isis.std() / isis.mean()), len(isis)


def binned_spike_counts(
    spike_times_list: list[np.ndarray], t_start_ms: float, t_end_ms: float, bin_ms: float
) -> np.ndarray:
    """Per-cell spike counts in fixed-width time bins over [t_start_ms,
    t_end_ms). Returns (n_cells, n_bins) int array -- one row per cell,
    matching spike_times_list's order."""
    edges = np.arange(t_start_ms, t_end_ms + bin_ms, bin_ms)
    n_bins = len(edges) - 1
    counts = np.empty((len(spike_times_list), n_bins), dtype=np.int64)
    for c, st in enumerate(spike_times_list):
        counts[c], _ = np.histogram(st, bins=edges)
    return counts


def psth(
    spike_times_list: list[np.ndarray], t_start_ms: float, t_end_ms: float, bin_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """Peri-stimulus time histogram: pooled per-bin firing rate (Hz),
    averaged across cells. Returns (bin_centers_ms, pop_rate_hz)."""
    counts = binned_spike_counts(spike_times_list, t_start_ms, t_end_ms, bin_ms)
    edges = np.arange(t_start_ms, t_end_ms + bin_ms, bin_ms)
    bin_centers_ms = (edges[:-1] + edges[1:]) / 2.0
    n_cells = max(1, len(spike_times_list))
    pop_rate_hz = counts.sum(axis=0) / n_cells / (bin_ms * 1e-3)
    return bin_centers_ms, pop_rate_hz


def population_power_spectrum(
    spike_times_list: list[np.ndarray],
    t_start_ms: float,
    t_end_ms: float,
    bin_ms: float,
    **welch_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of the pooled population rate (psth's output), via
    scipy.signal.welch. Returns (freqs_hz, power). fs is derived from
    bin_ms (fs = 1000/bin_ms samples/sec) -- pass nperseg via welch_kwargs
    if the default (scipy's, based on signal length) isn't appropriate for
    a short window."""
    _, pop_rate_hz = psth(spike_times_list, t_start_ms, t_end_ms, bin_ms)
    fs_hz = 1000.0 / bin_ms
    freqs_hz, power = welch(pop_rate_hz, fs=fs_hz, **welch_kwargs)
    return freqs_hz, power


def pairwise_correlation(binned_counts: np.ndarray) -> np.ndarray:
    """Pairwise Pearson correlation between cells' binned spike counts.
    binned_counts: (n_cells, n_bins), as returned by binned_spike_counts.
    Returns the (n_cells, n_cells) correlation matrix (np.corrcoef) --
    NaN rows/columns for any cell with zero variance (never fired, or fired
    at a constant rate every bin) are left as NaN, not silently zero-filled,
    since a zero-variance cell's "correlation" with anything is genuinely
    undefined, not zero. A single-cell input is a real, valid case (e.g. a
    sparse Golgi population at discovery scale) -- np.corrcoef collapses a
    1-row input to a 0-d scalar rather than a (1,1) matrix, so the result is
    forced back to 2D via np.atleast_2d for a consistent (n_cells, n_cells)
    shape regardless of n_cells."""
    return np.atleast_2d(np.corrcoef(binned_counts))


def mean_offdiag_correlation(corr_matrix: np.ndarray) -> float:
    """Mean of the off-diagonal entries of a pairwise_correlation() matrix --
    a single scalar synchrony index (the diagonal is always 1 by
    construction and would bias the mean upward if included). NaN entries
    (zero-variance cells) are dropped before averaging, not treated as 0.
    Returns nan if fewer than 2 cells or if every off-diagonal entry is NaN."""
    n = corr_matrix.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    vals = corr_matrix[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")
