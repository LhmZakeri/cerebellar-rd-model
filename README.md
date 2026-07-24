# Trilayer Cerebellar Cortex Coupled Circuit Model

A spatially-extended, node-centric model of the cerebellar cortex granular,
Purkinje, and molecular layers, built to study how coupling between the
three layers relates to whether they fire in a shared rhythm or desynchronize
into independent, possibly chaotic dynamics.

## Scientific questions

- **A (primary).** Is there a coupling tipping point — in strength, density,
  or climbing-fiber synchrony — beyond which the three layers desynchronize
  into independent rhythms, and where does it sit?
- **B.** What are the dynamics of the desynchronized regime, and does it
  become chaotic?
- **C.** Does loss of climbing-fiber synchrony reproduce tremor/ataxia-like
  dynamics?

This is Phase 1 of that project: single-cell ionic model prototypes, Numba-
accelerated batch kernels, and a working 2D granular-layer network with real
Golgi-cell coupling, in pure Python.

## What's implemented

Four ionic cell models and one synapse model, each with a validated
single-cell Python prototype and a Numba batch kernel for simulating arrays
of N cells at once:

| Model | Layer | Validation target |
|---|---|---|
| D'Angelo et al. 2001 | Granular — granule cells | 50-100 Hz burst firing |
| Solinas et al. 2007 | Granular — Golgi cells (sparse, gap-junction coupled) | 1-8 Hz pacemaker firing |
| Fernandez et al. 2007 | Purkinje | 40-100 Hz simple spikes; bistable pause after a climbing-fiber pulse |
| Reduced HH (Molineux/Mitry) | Molecular — basket/stellate cells | 20-50 Hz |
| TwoStateDestexhe | Synapses (excitatory/inhibitory) | EPSP 0.5-5 mV / hyperpolarizing IPSP |

On top of the cell models:

- `NodeBatch` — one granule, one Purkinje, and one stellate cell per node,
  connected by three vertical synapses (granule→Purkinje excitatory,
  stellate→Purkinje inhibitory, Purkinje→stellate inhibitory feedback).
  Mossy-fiber and climbing-fiber drive are separate, distinctly-routed
  inputs, targeting granule and Purkinje respectively.
- `GridNodeBatch` — places `NodeBatch` nodes on a 2D grid, adds a sparse
  Golgi cell population (~1:430 Golgi:granule, Poisson-disk placed),
  convergent (not 1:1) Golgi↔granule chemical synapses in both directions,
  and real Golgi↔Golgi gap-junction diffusion. Every inter-layer pathway —
  granule→Purkinje, stellate→Purkinje, Purkinje→stellate, mossy fiber→
  granule/Golgi — is built as a locality-biased convergent graph rather than
  a one-to-one wire; see `DESIGN.md` for why that mattered in practice, not
  just in principle.
- Disk-backed activity recording plus two viewers: an index-based per-layer
  trace view and a position-based spatial scatter view with a
  Golgi-diffusion wavefront overlay.
- A first coupling sweep (`scripts/exp3_grid`-family, see `figures/`) across
  mossy-fiber drive rate with gap junctions on vs. off, tracking Golgi ISI
  variability, firing rate, dominant frequency, and pairwise correlation —
  an early instance of the kind of sweep question A is asking, at a small
  scale.

## Quickstart

```bash
# run the full test suite (also JIT-compiles and validates the Numba kernels)
python -m pytest tests/ -v

# generate a single-cell voltage-trace figure
python scripts/run_dangelo2001_prototype.py    # -> figures/dangelo2001_prototype.png

# run a 2D granular-layer network and record it to disk
python scripts/record_2d_granular_activity.py --view

# view an existing recording without re-running the simulation
python scripts/view_activity.py <recording_dir> --spatial
```

No install step: `conftest.py` puts the repo root on `sys.path`, so every
import is `from src.models.X import Y`.

## Figures

`figures/architecture_schematic.png` — the three coupled layers, the sign of
each connection, and where climbing-fiber input and Golgi↔Golgi diffusion
enter the circuit.

`figures/exp3_coupling_sweep_summary.png` — a 40-run grid (5 mossy-fiber
rates × 4 seeds, gap junctions on vs. off): Golgi ISI coefficient of
variation and pairwise correlation both shift with drive rate and with
whether Golgi↔Golgi diffusion is active — an early look at question A.

`figures/tissue_3000x300um_gj_{off,on}.png` — a single run at tissue scale
(3000 × 300 µm, 9000 cells, 250 Golgi cells, 923/1000 independent
mossy-fiber populations to granule/Golgi) with a climbing-fiber pulse onto
900/9000 Purkinje cells at 1500 ms: Golgi/granule/Purkinje rasters, pooled
PSTH, Golgi power spectrum, and Golgi pairwise-correlation matrix, with and
without Golgi↔Golgi gap-junction coupling. Golgi ISI CV goes from 0.167
(off) to 0.286 (on) and the dominant Golgi frequency shifts from 5.3 Hz to
11.6 Hz — coupling changing both the regularity and the frequency content
of the same population, at a scale closer to the one the model is meant to
run at.

The rest of `figures/` are single-cell validation traces (spike shape,
firing rate, pacemaker behavior, synaptic PSP shape) reproduced against
their source papers.

## Architecture

- **Engine:** pure Python plus Numba (`@njit(parallel=True)`), not a
  second-language core — see `DESIGN.md` for the benchmark that decided
  this (Numba matches hand-tuned Cython/C++ on this workload, because the
  kernel is memory-bandwidth-bound rather than compute-bound).
- **Units contract:** `V [mV]`, `t [ms]`, `g [nS]`, `I [pA]` (`I_ext` given
  in nA, ×1000 internally), `Ca [mM]`, `x, y [µm]`.
- **Operator splitting:** diffusion at `dt = 0.1 ms` (Golgi↔Golgi gap
  junctions only); ODE and synapse subcycles at `dt = 0.01 ms`.

## Repository layout

```
src/models/        single-cell models and their Numba batch kernels
src/simulation/    NodeBatch, GridNodeBatch, geometry, recording, viewers
scripts/           prototype figures, the recording CLI, viewers, sweeps
tests/             single-cell validation, Numba parity, network/grid tests
figures/           validation traces and the exp3 coupling-sweep result
DESIGN.md          the reasoning behind the non-obvious decisions above
```

## About this repo

This is a curated snapshot of an active research project, published to
share the current state of the work. Commit history here is organized by
development milestone rather than mirroring every working commit.

Built with AI pair-programming assistance (Claude Code); the architecture,
calibration choices, and validation targets are mine.
