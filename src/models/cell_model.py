"""Abstract base class for single-compartment ionic cell models."""
from abc import ABC, abstractmethod

class CellModel(ABC):
    """
    Minimal public interface shared by all ionic cell models.
    
    Units contract (must be honoured by every subclass):)
        dt     — milliseconds (ms)
        I_ext  — nanoampres (nA)
        V      — millivolts (mv)
    """
    
    @abstractmethod
    def step(self, dt: float, I_ext: float) -> None:
        """Advance internal state by dt ms under external current __ext (nA). """
        
    @abstractmethod
    def get_voltage(self) -> float:
        """Return membrane coltage [mV]."""
        
    @abstractmethod 
    def reset(self) -> None:
        """Restore state to initial conditions."""
        
                 