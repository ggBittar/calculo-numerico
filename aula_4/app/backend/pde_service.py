from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expression_eval import evaluate_expression
from .pde_methods import PDE_METHODS, PdeMethodSpec
from .pdes import BoundaryCondition, PDES, PdeSpec

try:
    from . import pde_kernels as cython_kernels
except Exception:
    cython_kernels = None


@dataclass(frozen=True)
class PdeResult:
    pde_spec: PdeSpec
    method_spec: PdeMethodSpec
    axes: dict[str, np.ndarray]
    solution: np.ndarray
    exact_solution: np.ndarray | None
    final_slice: np.ndarray
    final_exact_slice: np.ndarray | None
    spatial_axes: tuple[str, ...]
    error_max: float | None
    metadata: dict[str, str]


BOUNDARY_TYPE_CODES = {
    "dirichlet": 0,
    "neumann": 1,
    "robin": 2,
}


def _build_axis_from_step(start: float, stop: float, step: float) -> np.ndarray:
    points = max(2, int(round((stop - start) / step)) + 1)
    return np.linspace(start, stop, points)


def _build_axis_from_count(start: float, stop: float, count: int) -> np.ndarray:
    points = max(2, int(count) + 1)
    return np.linspace(start, stop, points)


def _build_axes(
    pde_spec: PdeSpec,
    method_id: str,
    discretization_mode: str | None,
    discretization_modes: dict[str, str] | None,
    steps: dict[str, float],
    counts: dict[str, int],
) -> dict[str, np.ndarray]:
    axes: dict[str, np.ndarray] = {}
    for variable in pde_spec.variables:
        lower, upper = pde_spec.domain[variable]
        mode = (discretization_modes or {}).get(variable, discretization_mode or "step")
        if mode == "step":
            axes[variable] = _build_axis_from_step(lower, upper, steps[variable])
        elif mode == "count":
            axes[variable] = _build_axis_from_count(lower, upper, counts[variable])
        else:
            raise ValueError(f"Modo de discretizacao invalido para {variable}: {mode}")
    return axes


def _mesh_for_variables(axes: dict[str, np.ndarray], variables: tuple[str, ...]) -> dict[str, np.ndarray]:
    mesh = np.meshgrid(*(axes[name] for name in variables), indexing="ij")
    return {name: values for name, values in zip(variables, mesh, strict=True)}


def _initial_condition_from_config(
    mode: str,
    initial_value: float,
    initial_expression: str,
    axes: dict[str, np.ndarray],
    variables: tuple[str, ...],
) -> np.ndarray:
    spatial_variables = tuple(name for name in variables if name != "t")
    spatial_grid = _mesh_for_variables(axes, spatial_variables)

    if mode == "value":
        shape = tuple(len(axes[name]) for name in spatial_variables)
        return np.full(shape, initial_value, dtype=float)

    values = evaluate_expression(initial_expression, spatial_grid)
    shape = tuple(len(axes[name]) for name in spatial_variables)
    array = np.asarray(values, dtype=float)
    if array.shape == ():
        return np.full(shape, float(array), dtype=float)
    if array.shape != shape:
        try:
            return np.broadcast_to(array, shape).astype(float)
        except ValueError as exc:
            raise ValueError("A expressao da condicao inicial nao gerou um array compativel com a malha.") from exc
    return array


def _boundary_to_text(boundary: BoundaryCondition) -> str:
    if boundary.condition_type == "dirichlet":
        return f"Dirichlet(value={boundary.value:.4f})"
    if boundary.condition_type == "neumann":
        return f"Neumann(flux={boundary.value:.4f})"
    return (
        "Robin("
        f"a={boundary.robin_alpha:.4f}, "
        f"b={boundary.robin_beta:.4f}, "
        f"g={boundary.robin_gamma:.4f})"
    )


def _boundaries_summary(boundaries: dict[str, BoundaryCondition]) -> str:
    return ", ".join(f"{side}: {_boundary_to_text(boundaries[side])}" for side in sorted(boundaries))


def _boundary_matches_exact(
    exact_boundaries: dict[str, BoundaryCondition] | None,
    current_boundaries: dict[str, BoundaryCondition],
) -> bool:
    if exact_boundaries is None or exact_boundaries.keys() != current_boundaries.keys():
        return False

    for side, exact in exact_boundaries.items():
        current = current_boundaries[side]
        if exact.condition_type != current.condition_type:
            return False
        if not np.isclose(exact.value, current.value):
            return False
        if not np.isclose(exact.robin_alpha, current.robin_alpha):
            return False
        if not np.isclose(exact.robin_beta, current.robin_beta):
            return False
        if not np.isclose(exact.robin_gamma, current.robin_gamma):
            return False
    return True


def _evaluate_boundary(
    side: str,
    boundary: BoundaryCondition,
    neighbor: np.ndarray,
    delta: float,
) -> np.ndarray:
    if boundary.condition_type == "dirichlet":
        return np.full_like(neighbor, boundary.value, dtype=float)

    if boundary.condition_type == "neumann":
        return neighbor + boundary.value * delta

    if boundary.condition_type == "robin":
        denom = boundary.robin_alpha + (boundary.robin_beta / delta)
        if np.isclose(denom, 0.0):
            raise ValueError(f"Condicao de Robin invalida em {side}: a + b/delta nao pode ser zero.")
        return (boundary.robin_gamma + (boundary.robin_beta / delta) * neighbor) / denom

    raise ValueError(f"Tipo de contorno nao suportado em {side}: {boundary.condition_type}")


def _boundary_to_kernel_params(boundary: BoundaryCondition) -> tuple[int, float, float, float, float]:
    condition_type = boundary.condition_type.lower()
    if condition_type not in BOUNDARY_TYPE_CODES:
        raise ValueError(f"Tipo de contorno nao suportado: {boundary.condition_type}")
    return (
        BOUNDARY_TYPE_CODES[condition_type],
        float(boundary.value),
        float(boundary.robin_alpha),
        float(boundary.robin_beta),
        float(boundary.robin_gamma),
    )


def _apply_boundaries_1d(field: np.ndarray, dx: float, boundaries: dict[str, BoundaryCondition]) -> None:
    field[0] = _evaluate_boundary("x_min", boundaries["x_min"], np.asarray(field[1]), dx)
    field[-1] = _evaluate_boundary("x_max", boundaries["x_max"], np.asarray(field[-2]), dx)


def _apply_boundaries_2d(
    field: np.ndarray,
    dx: float,
    dy: float,
    boundaries: dict[str, BoundaryCondition],
) -> None:
    field[0, :] = _evaluate_boundary("x_min", boundaries["x_min"], field[1, :], dx)
    field[-1, :] = _evaluate_boundary("x_max", boundaries["x_max"], field[-2, :], dx)
    field[:, 0] = _evaluate_boundary("y_min", boundaries["y_min"], field[:, 1], dy)
    field[:, -1] = _evaluate_boundary("y_max", boundaries["y_max"], field[:, -2], dy)


def _build_save_indices(t_axis: np.ndarray, save_dt: float | None) -> np.ndarray:
    if save_dt is None:
        return np.arange(len(t_axis), dtype=int)
    if save_dt <= 0.0:
        raise ValueError("dt de salvamento deve ser maior que zero.")

    dt = float(t_axis[1] - t_axis[0])
    stride = max(1, int(round(save_dt / dt)))
    indices = np.arange(0, len(t_axis), stride, dtype=int)
    if indices[-1] != len(t_axis) - 1:
        indices = np.append(indices, len(t_axis) - 1)
    return indices


def _explicit_euler_heat_1d(
    initial: np.ndarray,
    axes: dict[str, np.ndarray],
    alpha: float,
    boundaries: dict[str, BoundaryCondition],
) -> np.ndarray:
    x = axes["x"]
    t = axes["t"]
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    solution = np.zeros((len(t), len(x)), dtype=float)
    solution[0] = initial
    _apply_boundaries_1d(solution[0], dx, boundaries)

    factor = alpha * dt / (dx**2)
    for n in range(len(t) - 1):
        next_state = solution[n].copy()
        next_state[1:-1] = solution[n, 1:-1] + factor * (
            solution[n, 2:] - 2.0 * solution[n, 1:-1] + solution[n, :-2]
        )
        _apply_boundaries_1d(next_state, dx, boundaries)
        solution[n + 1] = next_state

    return solution


def _explicit_euler_heat_1d_sparse(
    initial: np.ndarray,
    axes: dict[str, np.ndarray],
    alpha: float,
    boundaries: dict[str, BoundaryCondition],
    save_indices: np.ndarray,
) -> np.ndarray:
    x = axes["x"]
    t = axes["t"]
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    solution = np.zeros((len(save_indices), len(x)), dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    _apply_boundaries_1d(state, dx, boundaries)
    save_ptr = 0
    if save_indices[0] == 0:
        solution[0] = state
        save_ptr = 1

    factor = alpha * dt / (dx**2)
    for n in range(len(t) - 1):
        next_state = state.copy()
        next_state[1:-1] = state[1:-1] + factor * (state[2:] - 2.0 * state[1:-1] + state[:-2])
        _apply_boundaries_1d(next_state, dx, boundaries)
        state = next_state
        if save_ptr < len(save_indices) and (n + 1) == int(save_indices[save_ptr]):
            solution[save_ptr] = state
            save_ptr += 1

    return solution


def _explicit_euler_heat_2d(
    initial: np.ndarray,
    axes: dict[str, np.ndarray],
    alpha: float,
    boundaries: dict[str, BoundaryCondition],
) -> np.ndarray:
    x = axes["x"]
    y = axes["y"]
    t = axes["t"]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t[1] - t[0]

    solution = np.zeros((len(t), len(x), len(y)), dtype=float)
    solution[0] = initial
    _apply_boundaries_2d(solution[0], dx, dy, boundaries)

    fx = alpha * dt / (dx**2)
    fy = alpha * dt / (dy**2)

    for n in range(len(t) - 1):
        next_state = solution[n].copy()
        next_state[1:-1, 1:-1] = solution[n, 1:-1, 1:-1] + fx * (
            solution[n, 2:, 1:-1] - 2.0 * solution[n, 1:-1, 1:-1] + solution[n, :-2, 1:-1]
        ) + fy * (
            solution[n, 1:-1, 2:] - 2.0 * solution[n, 1:-1, 1:-1] + solution[n, 1:-1, :-2]
        )
        _apply_boundaries_2d(next_state, dx, dy, boundaries)
        solution[n + 1] = next_state

    return solution


def _explicit_euler_heat_2d_sparse(
    initial: np.ndarray,
    axes: dict[str, np.ndarray],
    alpha: float,
    boundaries: dict[str, BoundaryCondition],
    save_indices: np.ndarray,
) -> np.ndarray:
    x = axes["x"]
    y = axes["y"]
    t = axes["t"]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t[1] - t[0]

    solution = np.zeros((len(save_indices), len(x), len(y)), dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    _apply_boundaries_2d(state, dx, dy, boundaries)
    save_ptr = 0
    if save_indices[0] == 0:
        solution[0] = state
        save_ptr = 1

    fx = alpha * dt / (dx**2)
    fy = alpha * dt / (dy**2)

    for n in range(len(t) - 1):
        next_state = state.copy()
        next_state[1:-1, 1:-1] = state[1:-1, 1:-1] + fx * (
            state[2:, 1:-1] - 2.0 * state[1:-1, 1:-1] + state[:-2, 1:-1]
        ) + fy * (state[1:-1, 2:] - 2.0 * state[1:-1, 1:-1] + state[1:-1, :-2])
        _apply_boundaries_2d(next_state, dx, dy, boundaries)
        state = next_state
        if save_ptr < len(save_indices) and (n + 1) == int(save_indices[save_ptr]):
            solution[save_ptr] = state
            save_ptr += 1

    return solution


def _solve_explicit_euler(
    pde_spec: PdeSpec,
    method_id: str,
    axes: dict[str, np.ndarray],
    initial: np.ndarray,
    boundaries: dict[str, BoundaryCondition],
    save_indices: np.ndarray,
) -> np.ndarray:
    alpha = pde_spec.parameters["alpha"]
    can_use_dense = len(save_indices) == len(axes["t"])
    if cython_kernels is not None and can_use_dense:
        if pde_spec.pde_id == "heat_1d":
            left = _boundary_to_kernel_params(boundaries["x_min"])
            right = _boundary_to_kernel_params(boundaries["x_max"])
            return cython_kernels.explicit_euler_heat_1d(
                np.asarray(initial, dtype=np.float64),
                np.asarray(axes["x"], dtype=np.float64),
                np.asarray(axes["t"], dtype=np.float64),
                float(alpha),
                *left,
                *right,
            )
        if pde_spec.pde_id == "heat_2d":
            x_min = _boundary_to_kernel_params(boundaries["x_min"])
            x_max = _boundary_to_kernel_params(boundaries["x_max"])
            y_min = _boundary_to_kernel_params(boundaries["y_min"])
            y_max = _boundary_to_kernel_params(boundaries["y_max"])
            return cython_kernels.explicit_euler_heat_2d(
                np.asarray(initial, dtype=np.float64),
                np.asarray(axes["x"], dtype=np.float64),
                np.asarray(axes["y"], dtype=np.float64),
                np.asarray(axes["t"], dtype=np.float64),
                float(alpha),
                *x_min,
                *x_max,
                *y_min,
                *y_max,
            )

    if pde_spec.pde_id == "heat_1d":
        if can_use_dense:
            return _explicit_euler_heat_1d(initial, axes, alpha, boundaries)
        return _explicit_euler_heat_1d_sparse(initial, axes, alpha, boundaries, save_indices)
    if pde_spec.pde_id == "heat_2d":
        if can_use_dense:
            return _explicit_euler_heat_2d(initial, axes, alpha, boundaries)
        return _explicit_euler_heat_2d_sparse(initial, axes, alpha, boundaries, save_indices)
    raise ValueError(f"Solver Euler ainda nao implementado para {pde_spec.pde_id}.")


def _ghost_left(boundary: BoundaryCondition, first_value: np.ndarray, delta: float) -> np.ndarray:
    if boundary.condition_type == "dirichlet":
        return (2.0 * boundary.value) - first_value

    if boundary.condition_type == "neumann":
        return first_value - (boundary.value * delta)

    if boundary.condition_type == "robin":
        coeff_ghost = (boundary.robin_alpha / 2.0) - (boundary.robin_beta / delta)
        coeff_inner = (boundary.robin_alpha / 2.0) + (boundary.robin_beta / delta)
        if np.isclose(coeff_ghost, 0.0):
            raise ValueError("Condicao de Robin invalida no contorno esquerdo da malha centrada.")
        return (boundary.robin_gamma - (coeff_inner * first_value)) / coeff_ghost

    raise ValueError(f"Tipo de contorno nao suportado: {boundary.condition_type}")


def _ghost_right(boundary: BoundaryCondition, last_value: np.ndarray, delta: float) -> np.ndarray:
    if boundary.condition_type == "dirichlet":
        return (2.0 * boundary.value) - last_value

    if boundary.condition_type == "neumann":
        return last_value + (boundary.value * delta)

    if boundary.condition_type == "robin":
        coeff_ghost = (boundary.robin_alpha / 2.0) + (boundary.robin_beta / delta)
        coeff_inner = (boundary.robin_alpha / 2.0) - (boundary.robin_beta / delta)
        if np.isclose(coeff_ghost, 0.0):
            raise ValueError("Condicao de Robin invalida no contorno direito da malha centrada.")
        return (boundary.robin_gamma - (coeff_inner * last_value)) / coeff_ghost

    raise ValueError(f"Tipo de contorno nao suportado: {boundary.condition_type}")


def _ghost_pad_1d(field: np.ndarray, dx: float, boundaries: dict[str, BoundaryCondition]) -> np.ndarray:
    padded = np.empty(len(field) + 2, dtype=float)
    padded[1:-1] = field
    padded[0] = _ghost_left(boundaries["x_min"], np.asarray(field[0]), dx)
    padded[-1] = _ghost_right(boundaries["x_max"], np.asarray(field[-1]), dx)
    return padded


def _ghost_pad_2d(field: np.ndarray, dx: float, dy: float, boundaries: dict[str, BoundaryCondition]) -> np.ndarray:
    padded = np.empty((field.shape[0] + 2, field.shape[1] + 2), dtype=float)
    padded[1:-1, 1:-1] = field

    padded[0, 1:-1] = _ghost_left(boundaries["x_min"], field[0, :], dx)
    padded[-1, 1:-1] = _ghost_right(boundaries["x_max"], field[-1, :], dx)
    padded[1:-1, 0] = _ghost_left(boundaries["y_min"], field[:, 0], dy)
    padded[1:-1, -1] = _ghost_right(boundaries["y_max"], field[:, -1], dy)

    padded[0, 0] = 0.5 * (padded[0, 1] + padded[1, 0])
    padded[0, -1] = 0.5 * (padded[0, -2] + padded[1, -1])
    padded[-1, 0] = 0.5 * (padded[-1, 1] + padded[-2, 0])
    padded[-1, -1] = 0.5 * (padded[-1, -2] + padded[-2, -1])
    return padded


def _ghost_cells_heat_1d(
    initial: np.ndarray,
    axes: dict[str, np.ndarray],
    alpha: float,
    boundaries: dict[str, BoundaryCondition],
) -> np.ndarray:
    x = axes["x"]
    t = axes["t"]
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    factor = alpha * dt / (dx**2)

    solution = np.zeros((len(t), len(x)), dtype=float)
    solution[0] = np.asarray(initial, dtype=float)
    _apply_boundaries_1d(solution[0], dx, boundaries)

    for n in range(len(t) - 1):
        padded = _ghost_pad_1d(solution[n], dx, boundaries)
        next_state = solution[n].copy()
        next_state[1:-1] = solution[n, 1:-1] + factor * (
            padded[3:-1] - 2.0 * padded[2:-2] + padded[1:-3]
        )
        _apply_boundaries_1d(next_state, dx, boundaries)
        solution[n + 1] = next_state

    return solution


def _ghost_cells_heat_1d_sparse(
    initial: np.ndarray,
    axes: dict[str, np.ndarray],
    alpha: float,
    boundaries: dict[str, BoundaryCondition],
    save_indices: np.ndarray,
) -> np.ndarray:
    x = axes["x"]
    t = axes["t"]
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    factor = alpha * dt / (dx**2)

    solution = np.zeros((len(save_indices), len(x)), dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    _apply_boundaries_1d(state, dx, boundaries)
    save_ptr = 0
    if save_indices[0] == 0:
        solution[0] = state
        save_ptr = 1

    for n in range(len(t) - 1):
        padded = _ghost_pad_1d(state, dx, boundaries)
        next_state = state.copy()
        next_state[1:-1] = state[1:-1] + factor * (padded[3:-1] - 2.0 * padded[2:-2] + padded[1:-3])
        _apply_boundaries_1d(next_state, dx, boundaries)
        state = next_state
        if save_ptr < len(save_indices) and (n + 1) == int(save_indices[save_ptr]):
            solution[save_ptr] = state
            save_ptr += 1

    return solution


def _ghost_cells_heat_2d(
    initial: np.ndarray,
    axes: dict[str, np.ndarray],
    alpha: float,
    boundaries: dict[str, BoundaryCondition],
) -> np.ndarray:
    x = axes["x"]
    y = axes["y"]
    t = axes["t"]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t[1] - t[0]
    fx = alpha * dt / (dx**2)
    fy = alpha * dt / (dy**2)

    solution = np.zeros((len(t), len(x), len(y)), dtype=float)
    solution[0] = np.asarray(initial, dtype=float)
    _apply_boundaries_2d(solution[0], dx, dy, boundaries)

    for n in range(len(t) - 1):
        padded = _ghost_pad_2d(solution[n], dx, dy, boundaries)
        next_state = solution[n].copy()
        next_state[1:-1, 1:-1] = solution[n, 1:-1, 1:-1] + fx * (
            padded[3:-1, 2:-2] - 2.0 * padded[2:-2, 2:-2] + padded[1:-3, 2:-2]
        ) + fy * (
            padded[2:-2, 3:-1] - 2.0 * padded[2:-2, 2:-2] + padded[2:-2, 1:-3]
        )
        _apply_boundaries_2d(next_state, dx, dy, boundaries)
        solution[n + 1] = next_state

    return solution


def _ghost_cells_heat_2d_sparse(
    initial: np.ndarray,
    axes: dict[str, np.ndarray],
    alpha: float,
    boundaries: dict[str, BoundaryCondition],
    save_indices: np.ndarray,
) -> np.ndarray:
    x = axes["x"]
    y = axes["y"]
    t = axes["t"]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t[1] - t[0]
    fx = alpha * dt / (dx**2)
    fy = alpha * dt / (dy**2)

    solution = np.zeros((len(save_indices), len(x), len(y)), dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    _apply_boundaries_2d(state, dx, dy, boundaries)
    save_ptr = 0
    if save_indices[0] == 0:
        solution[0] = state
        save_ptr = 1

    for n in range(len(t) - 1):
        padded = _ghost_pad_2d(state, dx, dy, boundaries)
        next_state = state.copy()
        next_state[1:-1, 1:-1] = state[1:-1, 1:-1] + fx * (
            padded[3:-1, 2:-2] - 2.0 * padded[2:-2, 2:-2] + padded[1:-3, 2:-2]
        ) + fy * (
            padded[2:-2, 3:-1] - 2.0 * padded[2:-2, 2:-2] + padded[2:-2, 1:-3]
        )
        _apply_boundaries_2d(next_state, dx, dy, boundaries)
        state = next_state
        if save_ptr < len(save_indices) and (n + 1) == int(save_indices[save_ptr]):
            solution[save_ptr] = state
            save_ptr += 1

    return solution


def _solve_ghost_cells(
    pde_spec: PdeSpec,
    axes: dict[str, np.ndarray],
    initial: np.ndarray,
    boundaries: dict[str, BoundaryCondition],
    save_indices: np.ndarray,
) -> np.ndarray:
    alpha = pde_spec.parameters["alpha"]
    can_use_dense = len(save_indices) == len(axes["t"])

    if pde_spec.pde_id == "heat_1d":
        if can_use_dense:
            return _ghost_cells_heat_1d(initial, axes, alpha, boundaries)
        return _ghost_cells_heat_1d_sparse(initial, axes, alpha, boundaries, save_indices)

    if pde_spec.pde_id == "heat_2d":
        if can_use_dense:
            return _ghost_cells_heat_2d(initial, axes, alpha, boundaries)
        return _ghost_cells_heat_2d_sparse(initial, axes, alpha, boundaries, save_indices)

    raise ValueError(f"Solver com celulas fantasmas ainda nao implementado para {pde_spec.pde_id}.")


def _validate_stability(pde_spec: PdeSpec, axes: dict[str, np.ndarray]) -> None:
    alpha = pde_spec.parameters["alpha"]
    dt = axes["t"][1] - axes["t"][0]

    if pde_spec.pde_id == "heat_1d":
        dx = axes["x"][1] - axes["x"][0]
        limit = (dx**2) / (2.0 * alpha)
        if dt > limit:
            raise ValueError(f"Passo temporal instavel para Euler explicito. Use dt <= {limit:.5e}.")
        return

    if pde_spec.pde_id == "heat_2d":
        dx = axes["x"][1] - axes["x"][0]
        dy = axes["y"][1] - axes["y"][0]
        limit = 1.0 / (2.0 * alpha * ((1.0 / (dx**2)) + (1.0 / (dy**2))))
        if dt > limit:
            raise ValueError(f"Passo temporal instavel para Euler explicito. Use dt <= {limit:.5e}.")


def _build_exact_solution(
    pde_spec: PdeSpec,
    axes: dict[str, np.ndarray],
    boundaries: dict[str, BoundaryCondition],
) -> np.ndarray | None:
    if pde_spec.exact_solution is None:
        return None
    if not _boundary_matches_exact(pde_spec.exact_boundary_conditions, boundaries):
        return None

    grid = _mesh_for_variables(axes, pde_spec.variables)
    exact = pde_spec.exact_solution(grid, pde_spec.parameters)
    exact_array = np.asarray(exact, dtype=float)
    time_index = pde_spec.variables.index("t")
    transpose_order = [time_index] + [idx for idx in range(len(pde_spec.variables)) if idx != time_index]
    return np.transpose(exact_array, axes=transpose_order)


def solve_pde(
    pde_id: str,
    method_id: str,
    discretization_mode: str | None,
    steps: dict[str, float],
    counts: dict[str, int],
    initial_mode: str,
    initial_value: float,
    initial_expression: str,
    boundaries: dict[str, BoundaryCondition],
    discretization_modes: dict[str, str] | None = None,
    save_dt: float | None = None,
) -> PdeResult:
    pde_spec = PDES[pde_id]
    method_spec = PDE_METHODS[method_id]

    axes = _build_axes(pde_spec, method_id, discretization_mode, discretization_modes, steps, counts)
    _validate_stability(pde_spec, axes)
    initial = _initial_condition_from_config(
        mode=initial_mode,
        initial_value=initial_value,
        initial_expression=initial_expression,
        axes=axes,
        variables=pde_spec.variables,
    )
    save_indices = _build_save_indices(axes["t"], save_dt)
    saved_axes = dict(axes)
    saved_axes["t"] = axes["t"][save_indices]

    if method_id == "explicit_euler":
        solution = _solve_explicit_euler(pde_spec, method_id, axes, initial, boundaries, save_indices)
    elif method_id == "ghost_cells":
        solution = _solve_ghost_cells(pde_spec, axes, initial, boundaries, save_indices)
    else:
        raise ValueError(f"Metodo numerico nao suportado: {method_id}")
    exact_solution = _build_exact_solution(pde_spec, saved_axes, boundaries)
    exact_status = "disponivel"
    if pde_spec.exact_solution is None:
        exact_status = "nao cadastrada"
    elif exact_solution is None:
        exact_status = "indisponivel para o contorno configurado"

    spatial_axes = tuple(name for name in pde_spec.variables if name != "t")
    final_slice = solution[-1]
    final_exact_slice = exact_solution[-1] if exact_solution is not None else None
    error_max = None
    if final_exact_slice is not None:
        error_max = float(np.max(np.abs(final_slice - final_exact_slice)))

    effective_save_dt = "N/D"
    if len(saved_axes["t"]) > 1:
        effective_save_dt = f"{float(saved_axes['t'][1] - saved_axes['t'][0]):.5f}"

    metadata = {
        "equation": pde_spec.equation,
        "boundary": _boundaries_summary(boundaries),
        "exact": pde_spec.exact_expression or "Nao informada",
        "exact_status": exact_status,
        "initial_mode": "valor constante" if initial_mode == "value" else "funcao das variaveis espaciais",
        "save_dt": effective_save_dt,
        "mesh_type": method_spec.name,
    }

    return PdeResult(
        pde_spec=pde_spec,
        method_spec=method_spec,
        axes=saved_axes,
        solution=solution,
        exact_solution=exact_solution,
        final_slice=final_slice,
        final_exact_slice=final_exact_slice,
        spatial_axes=spatial_axes,
        error_max=error_max,
        metadata=metadata,
    )
