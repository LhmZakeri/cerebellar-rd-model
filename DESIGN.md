# Design notes

This document collects the reasoning behind the non-obvious decisions in this
model — why the engine is Numba and not C++, why cells are organized the way
they are, and why connectivity is built the way it is. The short version:
almost every decision here started as the "obviously more sophisticated"
option and was replaced by something simpler once I actually measured it
against the alternative. I've kept that trail visible because the negative
results (the things that turned out not to matter) are as informative as the
final design.

## Units contract

`V [mV]`, `t [ms]`, `g [nS]`, `I [pA]` (external drive is specified in nA and
converted internally), `Ca [mM]`, `x, y [µm]`. Held consistently across every
model and every layer so that cross-layer coupling terms don't need hidden
unit conversions.

## Why Numba, not a C++ core

I originally planned a two-language architecture: Python prototypes for
correctness, then a hand-written C++/pybind11 engine for the at-scale runs,
mirroring how I'd built the performance-critical part of my PhD cardiac
model. I got far enough to have a working C++ skeleton before stopping to
actually benchmark the assumption that a second language would be needed.

I ran the real bottleneck — the D'Angelo granule-cell update, 24 lookup
tables and roughly 16 `exp()` calls per node per substep — three ways on
identical hardware: plain Python, Numba (`@njit(parallel=True)`), and Cython
compiled with `-O3 -march=native -fopenmp`. Results:

- Numba's JIT alone, no threading, is already ~65x over plain Python.
  Adding 20 threads gets to ~548x. Nearly all of the win is the JIT, not the
  threading.
- Cython, built with the same optimization flags and thread count, landed at
  the *same* per-node-substep cost as Numba (13.76 ns vs 12.52 ns) — not
  faster. Since Cython and hand-written C++ both compile through the same
  gcc `-O3` backend on the same memory-access pattern, there was no reason
  to expect C++ to do better.
- Thread scaling was close to linear up to 10 threads (the machine's
  physical core count) and essentially flat from 10 to 20 (hyperthreading).
  That's the signature of a memory-bandwidth-bound kernel, not a
  compute-bound one — which also explains why a faster compiler didn't
  help: a faster compiler doesn't move a bandwidth ceiling.

Given that Cython matched Numba on real numbers, and I wanted to stay in one
language, Numba won outright. No second language, no build toolchain, no
Python/C++ drift to maintain. The abandoned C++ skeleton is gone; this
writeup is what survives of it.

I also tested — and ruled out — the hypothesis that the bottleneck was
*layout*: scattered gathers across 24 separate per-gate lookup tables.
Merging them into one interleaved array bought only ~3.7% (13.38 ns → 12.88
ns/node-substep) once I controlled for a flawed first attempt that used
identical voltages on every node (which hides layout effects entirely,
since every node then hits the same cache line). Combined with an earlier
~6% from pre-baking `exp(-dt/tau)` terms, this rules out table layout as the
real cost. The actual cost is simpler and less fixable: the D'Angelo model
carries 15 state arrays (voltage + 13 gates + calcium), each genuinely read
*and* written per node per substep — about 240 MB of real traffic per
substep at 2M nodes. That's intrinsic to the model's state size, not an
artifact of how the arrays are arranged.

This is still an open problem at the timestep the model needs for accuracy
(1 µs): a 2M-node, 10-second run is estimated at 135–200 hours even with 20
threads. The candidates I haven't tried yet are adaptive/variable timestep
(fewer substeps is the one lever that reduces traffic directly, rather than
rearranging it), and GPU, if it becomes available, since higher memory
bandwidth attacks the actual bottleneck rather than working around it.

## Node design

A `Node` co-locates one granule cell (D'Angelo et al. 2001), one Purkinje
cell (Fernandez et al. 2007), and one stellate cell (a reduced
Hodgkin-Huxley basket/stellate model) — one of each, always present. Golgi
cells (Solinas et al. 2007) are the exception: real Golgi:granule density is
about 1:430, so idealizing them to one-per-node would be wrong by two orders
of magnitude. Instead, Golgi cells live in a separate, sparse population
placed at a subset of node positions, chosen by Poisson-disk (hard-core)
sampling — uniform density, no lattice artifacts, no patchy clustering, and
a required explicit random seed, since placement feeds directly into the
coupling/synchrony sweeps this model exists to run, and whether a result
changed because of the physics or because Golgi cells landed differently
needs to stay an answerable question.

Three fixed vertical synapses connect the co-located cells: granule→Purkinje
(excitatory), stellate→Purkinje (inhibitory), and Purkinje→stellate
(inhibitory — GABAergic axon-collateral feedback, not the excitatory version
I originally miscoded). That last connection closes a genuine
reciprocal-inhibition loop with the stellate→Purkinje synapse: each cell
inhibits the other, not an excitation/rebound-inhibition pair.

Golgi↔granule connectivity is different in kind from the three vertical
synapses: it's one-to-many (~2000-cell divergence per Golgi cell, ~3-4 Golgi
cells converging per granule cell) and bidirectional, so it's built as two
neighbour lists sharing one sampled edge list rather than a fixed slot.

Mossy-fiber and climbing-fiber drive are modeled as two separate, distinctly
named inputs rather than one generic target-parameterized injection call.
That's deliberate: one of this model's scientific questions is whether loss
of climbing-fiber synchrony reproduces tremor/ataxia-like dynamics, which
only makes sense to test if the climbing-fiber pathway is represented
distinctly from ordinary mossy-fiber drive from the start, not bolted on
generically later.

## From single wires to real convergence

The first version of inter-layer coupling was a 1:1 wire: one granule cell
synapsing directly onto one Purkinje cell, at literature-reported
single-contact conductances (2.8 nS granule→Purkinje, 1.5 nS
stellate→Purkinje, 4.0 nS Purkinje→stellate). Once I actually measured the
effect, a single fully-spiking granule cell moved Purkinje voltage by about
0.00017 mV — roughly 30,000x below anything visible even in double
precision. The conductances were correct; the topology wasn't. Real synaptic
drive in this circuit comes from convergence — dozens to hundreds of
contacts summing — not from any individual contact being strong.

I rebuilt every inter-layer pathway as a convergent, locality-biased graph
instead of a 1:1 map: granule/parallel-fiber→Purkinje (~100 contacts),
stellate→Purkinje (5, revised down from an initial 50 once a tighter
measured-mean estimate replaced the wider tested-range figure it started
from), Purkinje→stellate (~2), and a parallel
fiber→stellate pathway (~3) — each sampled with a Gaussian distance falloff
so that nearby cells are preferentially connected, since the coupling/
synchrony sweeps this model is built for are sensitive to local vs. global
connectivity structure, not just density. Mossy-fiber drive got the same
treatment: real anatomy averages ~4 mossy-fiber inputs per granule cell and
~100 per Golgi cell, sampled from two independently-sized fiber
populations (not one shared pool, since the ~430:1 granule:Golgi density
ratio means one pool size can't hit both target ratios). Each target cell's
drive is the *mean* of its converging fibers' current, which preserves the
calibrated drive magnitude regardless of how many fibers converge on a given
cell.

Golgi↔Golgi gap-junction coupling is the one genuinely diffusive (monodomain
-style) connection in the model — everything else is chemical synapse or
direct current injection. It's built directly from placed Golgi positions
(radius search, auto-calibrated to 1.5x the median Golgi-Golgi
nearest-neighbour distance) rather than by filtering the dense structural
grid, because the grid's 10 µm spacing is far too fine to ever connect two
Golgi-hosting nodes at 1:430 density.

## What's deliberately not modeled yet

Point-neuron synapses here inject current directly at the soma, with no
cable-filtering equivalent of a dendrite-to-soma correction. I considered
two ways to add this and parked both: a single per-cell random
delay/attenuation would collapse per-pathway anatomical asymmetry onto one
number, which risks corrupting exactly the pathway (climbing fiber) whose
minimal-delay, high-fidelity proximal synapse matters most for the
tremor/ataxia question. The alternative — a small multi-compartment reference
model per pathway, used only to extract a per-synapse-type correction filter
— is the direction I'd take if this becomes necessary, rather than adding
multi-compartment dendrites to the at-scale model directly: since the model
is memory-bandwidth-bound rather than compute-bound, even 10-20
compartments per cell would multiply state-array traffic by roughly that
factor, which would erase most of the headroom gained by decoupling cell
count from the position grid in the first place.

Similarly parked: running at the real folium's physical scale
(~65mm × 110mm, ~71.5M nodes/layer at 10 µm resolution) and running at finer
spatial resolution (1 µm) at the current Phase 1 footprint both hit the same
wall — the current single-node-throughput ceiling described above — before
any new physics gets added on top. Both are one to two orders of magnitude
past a problem that's already unsolved at today's scale, so neither is worth
attempting until the throughput question has an answer.

## Validation approach

Every batch/array kernel is a direct Numba port of its own single-node
Python prototype's parameters and lookup tables — never a re-derivation —
and is checked against that prototype with a parity test (max pointwise
voltage deviation under 0.1 mV, the same bar I used for the original C++
plan). The synapse model's closed-form kernel matches to under 1e-9 mV,
since it has an exact analytical solution rather than an ODE integration to
approximate.
