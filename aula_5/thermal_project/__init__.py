from .models import BoundaryConfig, SimulationResult, ThermalConfig
from .solver import solve_problem
from .ui import MainWindow

__all__ = [
    "BoundaryConfig",
    "SimulationResult",
    "ThermalConfig",
    "solve_problem",
    "MainWindow",
]
