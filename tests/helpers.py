"""Shared test helpers for single-compartment cell model tests."""
from src.models.cell_model import CellModel

_DT = 0.025  # ms — standard timestep for all model tests


def run_cell(cell: CellModel, duration_ms: float, I_nA: float) -> None:
    """Advance cell for duration_ms at constant current, discarding output."""
    for _ in range(round(duration_ms / _DT)):
        cell.step(_DT, I_nA)


def count_spikes_cell(
    cell: CellModel,
    duration_ms: float,
    I_nA: float,
    threshold: float = 0.0,
) -> int:
    """Step cell for duration_ms and count upward threshold crossings."""
    spikes, above = 0, False
    for _ in range(round(duration_ms / _DT)):
        cell.step(_DT, I_nA)
        V = cell.get_voltage()
        if V > threshold and not above:
            spikes += 1
            above = True
        elif V <= threshold:
            above = False
    return spikes


def max_voltage_cell(cell: CellModel, duration_ms: float, I_nA: float) -> float:
    """Step cell for duration_ms and return the peak voltage reached."""
    V_max = cell.get_voltage()
    for _ in range(round(duration_ms / _DT)):
        cell.step(_DT, I_nA)
        V = cell.get_voltage()
        if V > V_max:
            V_max = V
    return V_max


def max_calcium_cell(cell: CellModel, duration_ms: float, I_nA: float) -> float:
    """Step cell for duration_ms and return the peak intracellular Ca2+ reached."""
    Ca_max = cell.get_calcium()
    for _ in range(round(duration_ms / _DT)):
        cell.step(_DT, I_nA)
        Ca = cell.get_calcium()
        if Ca > Ca_max:
            Ca_max = Ca
    return Ca_max
