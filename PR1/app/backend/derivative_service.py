from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .derivative_methods import METHODS, MethodSpec
from .functions import FUNCTIONS, FunctionSpec


@dataclass(frozen=True)
class DerivativeResult:
    function_spec: FunctionSpec
    method_spec: MethodSpec
    axis_name: str
    axis_values: np.ndarray
    sampled_values: np.ndarray
    approx_axis_values: np.ndarray
    approx_values: np.ndarray
    exact_values: np.ndarray | None
    slice_description: str
    metadata: dict[str, str]


def _build_axes(function_spec: FunctionSpec, ni_by_variable: dict[str, int]) -> dict[str, np.ndarray]:
    axes: dict[str, np.ndarray] = {}

    for variable in function_spec.variables:
        lower, upper = function_spec.domain[variable]
        points = max(3, int(ni_by_variable.get(variable, 21)))
        axes[variable] = np.linspace(lower, upper, points)

    return axes


def _build_slice(function_spec: FunctionSpec, axes: dict[str, np.ndarray]) -> tuple[tuple[object, ...], str]:
    slicer: list[object] = []
    labels: list[str] = []

    for variable in function_spec.variables:
        axis = axes[variable]
        if variable == function_spec.derivative_variable:
            slicer.append(slice(None))
            continue

        mid_index = len(axis) // 2
        slicer.append(mid_index)
        labels.append(f"{variable}={axis[mid_index]:.4f}")

    if not labels:
        return tuple(slicer), "sem corte adicional"

    return tuple(slicer), ", ".join(labels)


def estimate_derivative(
    function_id: str,
    method_id: str,
    ni_by_variable: dict[str, int],
) -> DerivativeResult:
    function_spec = FUNCTIONS[function_id]
    method_spec = METHODS[method_id]

    axes = _build_axes(function_spec, ni_by_variable)
    mesh = np.meshgrid(*(axes[name] for name in function_spec.variables), indexing="ij")
    grid = {name: array for name, array in zip(function_spec.variables, mesh, strict=True)}

    values = function_spec.evaluator(grid)
    slicer, slice_description = _build_slice(function_spec, axes)
    sampled_values = np.asarray(values[slicer], dtype=float)

    derivative_axis = function_spec.derivative_variable
    axis_values = axes[derivative_axis]
    delta = axis_values[1] - axis_values[0]

    approx_indices, approx_values = method_spec.estimator(sampled_values, delta)
    approx_axis_values = axis_values[approx_indices.astype(int)]

    exact_values = None
    if function_spec.analytical_derivative is not None:
        exact_grid = function_spec.analytical_derivative(grid)
        exact_values = np.asarray(exact_grid[slicer], dtype=float)

    metadata = {
        "function_expression": function_spec.expression,
        "derivative_expression": function_spec.derivative_expression or "Nao informada",
        "method_description": method_spec.description,
    }

    return DerivativeResult(
        function_spec=function_spec,
        method_spec=method_spec,
        axis_name=derivative_axis,
        axis_values=axis_values,
        sampled_values=sampled_values,
        approx_axis_values=approx_axis_values,
        approx_values=approx_values,
        exact_values=exact_values,
        slice_description=slice_description,
        metadata=metadata,
    )

