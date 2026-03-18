from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayMap = dict[str, np.ndarray]


@dataclass(frozen=True)
class BoundaryCondition:
    condition_type: str
    value: float = 0.0
    robin_alpha: float = 1.0
    robin_beta: float = 1.0
    robin_gamma: float = 0.0


@dataclass(frozen=True)
class PdeSpec:
    pde_id: str
    name: str
    equation: str
    variables: tuple[str, ...]
    domain: dict[str, tuple[float, float]]
    default_steps: dict[str, float]
    default_counts: dict[str, int]
    parameters: dict[str, float]
    boundary_sides: tuple[str, ...]
    default_boundary_conditions: dict[str, BoundaryCondition]
    exact_boundary_conditions: dict[str, BoundaryCondition] | None = None
    exact_solution: Callable[[ArrayMap, dict[str, float]], np.ndarray] | None = None
    exact_expression: str | None = None


PDES: dict[str, PdeSpec] = {
    "heat_1d": PdeSpec(
        pde_id="heat_1d",
        name="Calor 1D",
        equation="u_t = alpha * u_xx",
        variables=("x", "t"),
        domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        default_steps={"x": 0.25, "t": 0.001},
        default_counts={"x": 5, "t": 1001},
        parameters={"alpha": 1},
        boundary_sides=("x_min", "x_max"),
        default_boundary_conditions={
            "x_min": BoundaryCondition("dirichlet", value=373.0),
            "x_max": BoundaryCondition("dirichlet", value=273.0),
        },
        exact_boundary_conditions={
            "x_min": BoundaryCondition("dirichlet", value=0.0),
            "x_max": BoundaryCondition("dirichlet", value=0.0),
        },
        exact_solution=lambda grid, params: np.exp(-(np.pi**2) * params["alpha"] * grid["t"]) * np.sin(np.pi * grid["x"]),
        exact_expression="u(x,t) = exp(-pi^2*alpha*t) * sin(pi*x)",
    ),
    "heat_2d": PdeSpec(
        pde_id="heat_2d",
        name="Calor 2D",
        equation="u_t = alpha * (u_xx + u_yy)",
        variables=("x", "y", "t"),
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 0.1)},
        default_steps={"x": 0.1, "y": 0.1, "t": 0.002},
        default_counts={"x": 10, "y": 10, "t": 50},
        parameters={"alpha": 0.05},
        boundary_sides=("x_min", "x_max", "y_min", "y_max"),
        default_boundary_conditions={
            "x_min": BoundaryCondition("dirichlet", value=0.0),
            "x_max": BoundaryCondition("dirichlet", value=0.0),
            "y_min": BoundaryCondition("dirichlet", value=0.0),
            "y_max": BoundaryCondition("dirichlet", value=0.0),
        },
        exact_boundary_conditions={
            "x_min": BoundaryCondition("dirichlet", value=0.0),
            "x_max": BoundaryCondition("dirichlet", value=0.0),
            "y_min": BoundaryCondition("dirichlet", value=0.0),
            "y_max": BoundaryCondition("dirichlet", value=0.0),
        },
        exact_solution=lambda grid, params: np.exp(-2.0 * (np.pi**2) * params["alpha"] * grid["t"])
        * np.sin(np.pi * grid["x"])
        * np.sin(np.pi * grid["y"]),
        exact_expression="u(x,y,t) = exp(-2*pi^2*alpha*t) * sin(pi*x) * sin(pi*y)",
    ),
}
