"""Abstract base class for two-terminal synaptic models."""
from abc import ABC, abstractmethod


class SynapseModel(ABC):
    """
    Minimal public interface shared by all synaptic models.

    Units contract (must be honoured by every subclass):
        dt     — milliseconds (ms)
        V_pre  — millivolts (mV), presynaptic voltage (spike/release detection)
        V_post — millivolts (mV), postsynaptic voltage
        g      — nanosiemens (nS)
        I      — picoamperes (pA)
    """

    @abstractmethod
    def step(self, dt: float, V_pre: float) -> None:
        """Advance internal kinetic state by dt ms given presynaptic voltage V_pre [mV]."""

    @abstractmethod
    def get_current(self, V_post: float) -> float:
        """Return synaptic current [pA] delivered to the postsynaptic compartment at V_post [mV]."""

    @abstractmethod
    def reset(self) -> None:
        """Restore state to initial conditions."""
