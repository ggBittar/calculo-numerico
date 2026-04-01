from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayMap = dict[str, np.ndarray]


@dataclass(frozen=True)
class FunctionSpec:
    function_id: str
    name: str
    expression: str
    variables: tuple[str, ...]
    derivative_variable: str
    domain: dict[str, tuple[float, float]]
    default_points: dict[str, float]
    evaluator: Callable[[ArrayMap], np.ndarray]
    analytical_derivative: Callable[[ArrayMap], np.ndarray] | None = None
    derivative_expression: str | None = None


FUNCTIONS: dict[str, FunctionSpec] = {
    "cos_x": FunctionSpec(
        function_id="cos_x",
        name="cosseno 1D",
        expression="f(x) = cos(x)",
        variables=("x",),
        derivative_variable="x",
        domain={"x": (-2.0 * np.pi, 2.0 * np.pi)},
        default_points={},
        evaluator=lambda grid: np.cos(grid["x"]),
        analytical_derivative=lambda grid: -np.sin(grid["x"]),
        derivative_expression="df/dx = -sin(x)",
    ),
    "exp_xy": FunctionSpec(
        function_id="exp_xy",
        name="exponencial 2D",
        expression="f(x, y) = exp(x*y)",
        variables=("x", "y"),
        derivative_variable="x",
        domain={"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        default_points={"y": 0.0},
        evaluator=lambda grid: np.exp(grid["x"] * grid["y"]),
        analytical_derivative=lambda grid: grid["y"] * np.exp(grid["x"] * grid["y"]),
        derivative_expression="df/dx = y*exp(x*y)",
    ),
    "sin_xt": FunctionSpec(
        function_id="sin_xt",
        name="seno espaço-tempo",
        expression="f(x, t) = sin(x + t)",
        variables=("x", "t"),
        derivative_variable="x",
        domain={"x": (0.0, 2.0 * np.pi), "t": (0.0, 1.0)},
        default_points={"t": 0.5},
        evaluator=lambda grid: np.sin(grid["x"] + grid["t"]),
        analytical_derivative=lambda grid: np.cos(grid["x"] + grid["t"]),
        derivative_expression="df/dx = cos(x + t)",
    ),
    "xyz_polynomial": FunctionSpec(
        function_id="xyz_polynomial",
        name="polinômio 3D",
        expression="f(x, y, z) = x^2 + y*z",
        variables=("x", "y", "z"),
        derivative_variable="x",
        domain={"x": (-2.0, 2.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        default_points={"y": 0.0, "z": 0.0},
        evaluator=lambda grid: grid["x"] ** 2 + grid["y"] * grid["z"],
        analytical_derivative=lambda grid: 2.0 * grid["x"],
        derivative_expression="df/dx = 2*x",
    ),
}

