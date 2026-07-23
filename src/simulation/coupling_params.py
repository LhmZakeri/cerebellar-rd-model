"""
GridCouplingParams -- the coupling-strength/connectivity-shape parameters for
the connectivity pathways GridNodeBatch adds on top of NodeBatch's fixed
vertical synapses: the bidirectional Golgi<->granule chemical synapses
(DESIGN.md -- inhibitory Golgi->granule, excitatory granule->Golgi, over
the same shared edge list), Golgi<->Golgi gap-junction diffusion
(DESIGN.md), the four convergent granule/Purkinje/stellate
pathways that replace NodeBatch's own 1:1 vertical synapses under
GridNodeBatch (DESIGN.md): granule/PF->Purkinje, stellate->Purkinje,
Purkinje->stellate, and parallel-fiber->stellate; and the two convergent
mossy-fiber pathways (DESIGN.md): mossy fiber->granule and mossy
fiber->Golgi.

Plain mutable @dataclass, not frozen -- matches this repo's actual
convention (DAngelo2001Params, Fernandez2007Params, MolineuxStellateParams
are all plain dataclasses), typed float fields with inline unit comments,
no magic numbers in GridNodeBatch.step()/__init__ itself.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GridCouplingParams:
    # --- Golgi<->Golgi gap-junction diffusion --------------------
    g_gap_nS: float = 1.0  # per-edge gap-junction conductance, nS (uncalibrated
    # placeholder -- matches NodeBatch's gmax_*=1.0 default convention)
    golgi_diffusion_radius_um: float | None = None  # um; None -> auto-calibrate
    # to 1.5x the median nearest-neighbour distance among placed Golgi cells
    # (see build_golgi_diffusion_neighbours), so it adapts to whatever
    # golgi_ratio/extent was actually used instead of a fixed magic constant.

    # --- Golgi<->granule chemical synapses (DESIGN.md) -------------------
    gmax_golgi_to_granule: float = 1.0  # nS, inhibitory (Golgi -> granule)
    gmax_granule_to_golgi: float = 1.0  # nS, excitatory (granule -> Golgi);
    # shares the same edge list as gmax_golgi_to_granule (one bidirectional
    # anatomical contact, not two independently-sampled graphs -- DESIGN.md). This is on top of Golgi cells' own direct mossy-fiber excitation
    # (see GridNodeBatch.inject_mossy_fiber_input) -- real Golgi cells receive
    # both mossy-fiber input on basal dendrites and parallel-fiber/granule
    # input on apical dendrites, so the two excitatory pathways coexist.
    golgi_granule_divergence: int = 2000  # cells per Golgi cell, ~1:2000 per ADR 0001
    golgi_granule_locality_sigma_um: float = 250.0  # Gaussian locality falloff
    # scale, um -- matches DESIGN.md's synapticNeighbors (stellate->Purkinje)
    # reach of ~200-300um, the only comparable local-synaptic-reach number
    # already committed to in this codebase.

    # --- granule/PF -> Purkinje, convergent (DESIGN.md) -------------------
    gmax_granule_to_purkinje: float = 2.8  # nS, excitatory, single-contact
    # unitary PF EPSC (Mas17) -- NOT a 1:1 wire; see granule_to_purkinje_contacts.
    granule_to_purkinje_contacts: int = 100  # active PF/GC contacts per
    # Purkinje cell (Mas17, Riz21) -- reduced models commonly use ~100;
    # detailed models use hundreds-to-thousands depending on simulated volume.
    granule_to_purkinje_locality_sigma_um: float = 250.0  # UNCALIBRATED for PF's
    # real anisotropic beam reach (parallel fibers run mm-scale, roughly
    # perpendicular to Purkinje dendritic trees) -- an isotropic Gaussian is a
    # known simplification here, not a validated model. See DESIGN.md.

    # --- stellate -> Purkinje, convergent (DESIGN.md, revised DESIGN.md) -
    gmax_stellate_to_purkinje: float = 1.5  # nS, inhibitory, single-contact
    # (Mas17, Riz21)
    stellate_to_purkinje_contacts: int = 5  # cells -- Sch21's reconstructed
    # connectome measured 5.4+/-2.7 stellate cells converging onto one
    # Purkinje cell; rounded to the nearest integer (DESIGN.md). This
    # replaces the original 50, which was Riz21's *tested range* (25-300,
    # ~100 as their "main condition" for a filtering result), not a measured
    # anatomical convergence count -- Riz21's own number was itself
    # model-dependent, not the actual reconstructed-connectome ratio. Len14
    # reports a different reduced-model value (~20 MLI inputs/Purkinje cell)
    # -- a real cross-source discrepancy, not resolved here; 5 is anchored to
    # the directly-measured reconstructed-connectome number, still flagged
    # for revisit if firing-rate validation favors Len14's higher value.
    stellate_to_purkinje_locality_sigma_um: float = 250.0  # matches DESIGN.md's
    # synapticNeighbors (stellate->Purkinje) reach of ~200-300um.

    # --- Purkinje -> stellate/MLI, convergent (DESIGN.md) -----------------
    gmax_purkinje_to_stellate: float = 4.0  # nS, inhibitory (Len14) -- base
    # PC->MLI conductance; corrected from an original miscoded excitatory
    # version, see node_batch.py and DESIGN.md.
    purkinje_to_stellate_Erev_mV: float = -82.0  # Len14
    purkinje_to_stellate_contacts: int = 2  # ~1-2 Purkinje inputs per MLI
    # (Len14, Hal22)
    purkinje_to_stellate_locality_sigma_um: float = 250.0

    # --- parallel fiber (granule) -> stellate, convergent, NEW (DESIGN.md) -
    gmax_granule_to_stellate: float = 2.3  # nS, excitatory, single-contact
    # (Riz21) -- this pathway did not exist at all before DESIGN.md.
    granule_to_stellate_contacts: int = 3  # ~3 PF contacts per stellate cell
    # in the reduced model (Riz21)
    granule_to_stellate_locality_sigma_um: float = 250.0  # same PF-anisotropy
    # caveat as granule_to_purkinje_locality_sigma_um above.

    # --- mossy fiber -> granule, convergent (DESIGN.md) -------------------
    # Real cerebellar anatomy: mossy fibers are NOT a 1:1 wire to either
    # target population -- each granule cell averages ~4 mossy-fiber inputs,
    # each mossy fiber diverges onto ~39 granule cells. Both numbers set the
    # size of a sampled mossy-fiber-for-granule population (GridNodeBatch:
    # n_mf_granule = n_cells * contacts / divergence), independent of the
    # mossy-fiber-for-Golgi population below -- see DESIGN.md for why one
    # shared pool can't hit both target populations' ratios simultaneously
    # given the ~430:1 granule:Golgi cell-count disparity.
    mossy_to_granule_contacts: int = 4  # avg mossy-fiber inputs per granule cell
    mossy_to_granule_divergence: int = 39  # avg granule cells reached per fiber
    mossy_to_granule_locality_sigma_um: float = 250.0

    # --- mossy fiber -> Golgi, convergent (DESIGN.md) ----------------------
    # Golgi cells receive direct mossy-fiber excitation on their basal
    # dendrites (on top of, not instead of, the granule->Golgi feedback synapse
    # above) -- real anatomy: ~100 mossy-fiber inputs per Golgi cell, ~25 Golgi
    # cells reached per fiber.
    mossy_to_golgi_contacts: int = 100  # avg mossy-fiber inputs per Golgi cell
    mossy_to_golgi_divergence: int = 25  # avg Golgi cells reached per fiber
    mossy_to_golgi_locality_sigma_um: float = 250.0

    # --- Sou11-style distance-scaled conductance (DESIGN.md) ---------------
    # None (default) -> no distance scaling, exact prior behavior (every
    # existing test/script is unaffected). A float -> exponential decay
    # weight exp(-distance_decay_per_um * distance_um) applied to
    # mossy->granule, mossy->Golgi, and the inhibitory Golgi->granule
    # direction -- Sou11's own value (0.01/um) reproduced from the actual
    # paper text. NOT applied to the excitatory granule->Golgi direction
    # (the PF->GoC-equivalent contact), which Sou11 keeps constant along
    # the fiber -- see DESIGN.md for the exact exception.
    distance_decay_per_um: float | None = None
