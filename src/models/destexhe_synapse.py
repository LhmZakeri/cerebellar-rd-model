"""
Destexhe, Mainen & Sejnowski (1994) minimal two-state kinetic synapse model —
Phase 1 Python prototype.

Model: (closed) + T <-> (open)

    dr/dt = Alpha * [T] * (1 - r) - Beta * r
    I     = gmax * r * (V - Erev)

where [T] is the transmitter concentration and r is the fraction of
receptors in the open state. Transmitter is approximated as a brief pulse
of fixed duration Cdur, triggered whenever the presynaptic voltage crosses
Prethresh (and at least Deadtime ms have elapsed since the last release).
With a fixed-duration pulse, this first-order kinetic equation has a
closed-form solution, so step() evaluates that analytic solution directly
rather than integrating an ODE — exactly as in the original mechanism (see
"numerical stability" note in the .mod source: no differential equation is
solved at run time).

Units: t [ms], V [mV], r dimensionless, gmax [nS], I [pA] — consistent with
the project-wide units contract (DESIGN.md), since nS * mV = pA, the same
relation the original mechanism uses for umho * mV = nA.

Source (unmodified formulas and published parameter values):
  ModelDB accession #18198, https://modeldb.science/18198
  (Destexhe, Mainen & Sejnowski 1994/1998) — ampa.mod (excitatory preset)
  and gabaa.mod (inhibitory preset).

References:
  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. (1994) "An efficient method
  for computing synaptic conductances based on a kinetic model of receptor
  binding." Neural Computation 6: 10-14.
  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. (1998) "Kinetic models of
  synaptic transmission." In: Methods in Neuronal Modeling (2nd ed.;
  Koch, C. and Segev, I., eds), MIT Press, pp. 1-25.
  AMPA fit to Xiang et al. (1994) J Neurophysiol 71: 2552-2556.
  GABA-A fit to Otis and Mody (1992) Neuroscience 49: 13-32.
"""

import math
from dataclasses import dataclass

from src.models.synapse_model import SynapseModel


@dataclass
class TwoStateDestexheParams:
    """Two-state kinetic synapse parameters (ModelDB #18198 ampa.mod / gabaa.mod).

    Units: Cmax [mM], Cdur/Deadtime [ms], Alpha [/ms/mM], Beta [/ms], Erev [mV],
    gmax [nS].

    Defaults reproduce ampa.mod exactly (excitatory preset); gmax has no
    published default in the source (it is set per-synapse by the caller),
    so it is a required argument here.
    """
    gmax: float                # nS — maximum conductance (set by caller; no source default)
    Cmax: float = 1.0          # mM — max transmitter concentration
    Cdur: float = 1.0          # ms — transmitter pulse duration
    Alpha: float = 1.1         # /ms/mM — forward (binding) rate
    Beta: float = 0.19         # /ms — backward (unbinding) rate
    Erev: float = 0.0          # mV — reversal potential
    Prethresh: float = 0.0     # mV — presynaptic voltage level necessary for release
    Deadtime: float = 1.0      # ms — minimum time between release events


class TwoStateDestexhe(SynapseModel):
    """Two-state kinetic synapse: single class, parametrised for either
    excitatory (AMPA-like, granular -> Purkinje) or inhibitory
    (GABA-A-like, molecular -> Purkinje) coupling via TwoStateDestexheParams.

    State: R, C, R0, R1, Rinf, Rtau, lastrelease, TimeCount — mirroring the
    ASSIGNED block of the original .mod mechanism one-to-one.
    """

    def __init__(self, params: TwoStateDestexheParams) -> None:
        self._p = params
        self._t = 0.0
        self._reset_state()

    # --- SynapseModel interface -------------------------------------------------

    def step(self, dt: float, V_pre: float) -> None:
        p = self._p
        self._t += dt
        self.TimeCount -= dt

        if self.TimeCount < -p.Deadtime:
            if V_pre > p.Prethresh:
                self.C = p.Cmax
                self.R0 = self.R
                self.lastrelease = self._t
                self.TimeCount = p.Cdur
        elif self.TimeCount > 0:
            pass  # still releasing
        elif self.C == p.Cmax:
            self.R1 = self.R
            self.C = 0.0

        if self.C > 0:
            self.R = self.Rinf + (self.R0 - self.Rinf) * math.exp(
                -(self._t - self.lastrelease) / self.Rtau
            )
        else:
            self.R = self.R1 * math.exp(
                -p.Beta * (self._t - (self.lastrelease + p.Cdur))
            )

    def get_current(self, V_post: float) -> float:
        """I = gmax * R * (V_post - Erev)  [pA]."""
        return self._p.gmax * self.R * (V_post - self._p.Erev)

    def reset(self) -> None:
        self._t = 0.0
        self._reset_state()

    # --- Extra accessor -----------------------------------------------------

    def get_conductance(self) -> float:
        """g = gmax * R  [nS]."""
        return self._p.gmax * self.R

    # --- Presets (published parameter values, unchanged) ---------------------

    @classmethod
    def excitatory(cls, gmax: float) -> "TwoStateDestexhe":
        """AMPA-like preset (ampa.mod defaults): granular -> Purkinje, Erev = 0 mV."""
        return cls(TwoStateDestexheParams(gmax=gmax))

    @classmethod
    def inhibitory(cls, gmax: float) -> "TwoStateDestexhe":
        """GABA-A-like preset (gabaa.mod defaults): molecular -> Purkinje, Erev = -80 mV."""
        return cls(TwoStateDestexheParams(gmax=gmax, Alpha=5.0, Beta=0.18, Erev=-80.0))

    # --- Internal -------------------------------------------------------------

    def _reset_state(self) -> None:
        p = self._p
        self.R = 0.0
        self.C = 0.0
        self.R0 = 0.0
        self.R1 = 0.0
        self.Rinf = p.Cmax * p.Alpha / (p.Cmax * p.Alpha + p.Beta)
        self.Rtau = 1.0 / (p.Alpha * p.Cmax + p.Beta)
        self.lastrelease = -1000.0
        self.TimeCount = -1.0
