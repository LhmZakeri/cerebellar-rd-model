"""Poisson-process mossy-fiber spike-train generation and a minimal
spike-to-current low-pass filter, for experiments that drive
GridNodeBatch.inject_mossy_fiber_input() with a time-varying signal instead
of a static tonic value (DESIGN.md). Generic over any (n_fibers,
n_steps, rate) combination -- not specific to any one experiment protocol.
"""
from __future__ import annotations

import numpy as np


def generate_poisson_spike_train(
    n_fibers: int,
    n_steps: int,
    dt_ms: float,
    rate_hz: float,
    seed: int,
    on_step: int = 0,
    off_step: int | None = None,
) -> np.ndarray:
    """Independent per-fiber Poisson spike train via a per-step Bernoulli
    approximation: p_spike = rate_hz * dt_ms / 1000. Valid only while
    p_spike << 1 (raises ValueError if p_spike >= 1, e.g. dt_ms too coarse
    or rate_hz too high for this approximation to mean anything) -- at
    dt_ms=0.01 and rate_hz<=70, p_spike ~7e-4, safely small.

    Returns a bool array of shape (n_steps, n_fibers), True outside
    [on_step, off_step) everywhere zero (no spikes) -- off_step=None means
    active through n_steps.

    seed: required, explicit -- no default, matching this codebase's
    reproducibility convention for anything connectivity/stimulus-random
    (place_golgi_cells, sample_uniform_positions).
    """
    p_spike = rate_hz * dt_ms / 1000.0
    if p_spike >= 1.0:
        raise ValueError(
            f"p_spike = rate_hz*dt_ms/1000 = {p_spike} >= 1 -- rate_hz={rate_hz} is too high "
            f"(or dt_ms={dt_ms} too coarse) for the per-step Bernoulli approximation to be valid."
        )
    if off_step is None:
        off_step = n_steps

    rng = np.random.default_rng(seed)
    spikes = np.zeros((n_steps, n_fibers), dtype=bool)
    n_active_steps = off_step - on_step
    if n_active_steps > 0:
        spikes[on_step:off_step] = rng.random((n_active_steps, n_fibers)) < p_spike
    return spikes


class ExponentialCurrentFilter:
    """Converts a per-channel boolean spike train into a smoothed injectable
    current via a single-exponential decay recursion:

        I[t+dt] = I[t]*exp(-dt/tau_ms) + amplitude_nA*spike[t]

    Mossy-fiber current has no chemical-synapse gating kinetics in this
    codebase (DESIGN.md -- it's raw experimenter-injected current, no
    presynaptic cell model) -- this is the minimal causal low-pass so a
    single dt-wide spike event has a non-negligible integrated effect on the
    postsynaptic cell, not a claim about real vesicle-release kinetics.

    amplitude_nA/tau_ms are UNCALIBRATED placeholders (matches
    coupling_params.py's g_gap_nS convention) -- amplitude_nA=0.05 matches
    this repo's existing MOSSY_STRENGTH_NA tonic-drive scale; tau_ms=3.0 is
    a generic fast-excitatory-like decay constant. Validate against actual
    firing-rate output before trusting downstream conclusions, not against
    literature.
    """

    def __init__(self, n_channels: int, amplitude_nA: float = 0.05, tau_ms: float = 3.0) -> None:
        self.n_channels = n_channels
        self.amplitude_nA = amplitude_nA
        self.tau_ms = tau_ms
        self._I = np.zeros(n_channels, dtype=np.float64)

    def step(self, dt_ms: float, spike: np.ndarray) -> np.ndarray:
        decay = np.exp(-dt_ms / self.tau_ms)
        self._I = self._I * decay + self.amplitude_nA * spike
        return self._I

    def reset(self) -> None:
        self._I = np.zeros(self.n_channels, dtype=np.float64)
