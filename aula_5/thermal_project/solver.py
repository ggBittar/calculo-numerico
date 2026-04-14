from __future__ import annotations

import numpy as np

from .models import BoundaryConfig, SimulationResult, ThermalConfig


def _nonlinear_flux(
    boundary_temperature: float,
    first_inner_temperature: float,
    boundary_cfg: BoundaryConfig,
    cfg: ThermalConfig,
    dx: float,
    position_factor: float = 1.0,
) -> float:
    return (
        boundary_cfg.emissividade
        * boundary_cfg.sigma
        * (boundary_cfg.temperatura_vizinhanca**4 - boundary_temperature**4)
        + boundary_cfg.h * (boundary_cfg.temperatura_infinito - boundary_temperature)
        + (boundary_cfg.fluxo_superficial / cfg.area)
        + ((cfg.condutividade * boundary_cfg.contato / dx) * position_factor * (boundary_cfg.temperatura_contorno - first_inner_temperature))
    )


def left_ghost_temperature(reference_state: np.ndarray, cfg: ThermalConfig, dx: float) -> float:
    boundary_temperature = reference_state[1]
    first_inner_temperature = reference_state[2]
    flux = _nonlinear_flux(boundary_temperature, first_inner_temperature, cfg.esquerda, cfg, dx)
    return float(first_inner_temperature + ((2.0 * dx / cfg.condutividade) * flux))


def right_ghost_temperature(reference_state: np.ndarray, cfg: ThermalConfig, dx: float) -> float:
    boundary_temperature = reference_state[-2]
    first_inner_temperature = reference_state[-3]
    flux = _nonlinear_flux(boundary_temperature, first_inner_temperature, cfg.direita, cfg, dx, -1)
    # Na fronteira direita, o sentido normal e' oposto ao da esquerda.
    return float(first_inner_temperature - ((2.0 * dx / cfg.condutividade) * flux))


def apply_ghost_cells(state: np.ndarray, reference_state: np.ndarray, cfg: ThermalConfig, dx: float) -> None:
    state[0] = left_ghost_temperature(reference_state, cfg, dx)
    state[-1] = right_ghost_temperature(reference_state, cfg, dx)


def solve_problem(cfg: ThermalConfig) -> SimulationResult:
    if cfg.nx < 3:
        raise ValueError("Use pelo menos 3 pontos espaciais para o problema com células fantasmas.")
    if cfg.x_final <= cfg.x_inicial:
        raise ValueError("O intervalo espacial deve satisfazer xf > xo.")
    if cfg.tempo_final <= cfg.tempo_inicial:
        raise ValueError("O intervalo temporal deve satisfazer tf > to.")
    if cfg.co <= 0.0:
        raise ValueError("Co deve ser maior que zero.")
    if cfg.co > 0.5:
        raise ValueError("Co deve ser menor ou igual a 0.5 para estabilidade no esquema explicito.")
    if cfg.condutividade <= 0.0 or np.isclose(cfg.condutividade, 0.0):
        raise ValueError("A condutividade k deve ser maior que zero.")
    if cfg.area <= 0.0 or np.isclose(cfg.area, 0.0):
        raise ValueError("A área A deve ser maior que zero.")

    x_axis = np.linspace(cfg.x_inicial, cfg.x_final, cfg.nx)
    dx = float(x_axis[1] - x_axis[0])
    dt_from_co = cfg.co * (dx**2) / (cfg.condutividade * cfg.area)
    if dt_from_co <= 0.0 or np.isclose(dt_from_co, 0.0):
        raise ValueError("dt calculado por Co e invalido. Verifique Co, k, A e malha espacial.")

    nt = max(2, int(round((cfg.tempo_final - cfg.tempo_inicial) / dt_from_co)) + 1)
    t_axis = np.linspace(cfg.tempo_inicial, cfg.tempo_final, nt)
    dt_effective = float(t_axis[1] - t_axis[0])
    factor = cfg.condutividade * cfg.area * dt_effective / (dx**2)
    max_stable_factor = 0.5
    if factor > max_stable_factor:
        raise ValueError(
            "Instabilidade numerica detectada no esquema explicito: "
            f"k*A*dt/dx^2 = {factor:.6g} excede {max_stable_factor:.3g}. "
            "Reduza Co, aumente nx (reduzindo dx) ou ajuste os parametros fisicos."
        )

    state = np.full(cfg.nx + 2, cfg.temperatura_inicial, dtype=float)
    apply_ghost_cells(state, state.copy(), cfg, dx)

    solution = np.zeros((nt, cfg.nx), dtype=float)
    solution[0] = state[1:-1]

    for time_index in range(1, nt):
        previous_state = state.copy()
        state = previous_state.copy()
        for spatial_index in range(1, cfg.nx + 1):
            state[spatial_index] = previous_state[spatial_index] + factor * (
                previous_state[spatial_index + 1]
                - (2.0 * previous_state[spatial_index])
                + previous_state[spatial_index - 1]
            )
        apply_ghost_cells(state, previous_state, cfg, dx)
        solution[time_index] = state[1:-1]

    left_formula = (
        "T_ghost,E = T_2 + (2*dx/k) * ["
        "eE*sigmaE*(TvizE^4 - T_1^4) + hE*(TinfE - T_1) + qsE/A + (k*CE/dx)*(TrefE - T_2)]"
    )
    right_formula = (
        "T_ghost,D = T_(N-1) - (2*dx/k) * ["
        "eD*sigmaD*(TvizD^4 - T_N^4) + hD*(TinfD - T_N) + qsD/A + (k*CD/dx)*(TrefD - T_(N-1))]"
    )

    global_min = float(np.min(solution))
    global_max = float(np.max(solution))
    if np.isclose(global_min, global_max):
        margin = max(1.0, 0.05 * max(abs(global_min), 1.0))
    else:
        margin = max(0.5, 0.05 * (global_max - global_min))
    plot_y_limits = (global_min - margin, global_max + margin)

    return SimulationResult(
        config=cfg,
        x_axis=x_axis,
        t_axis=t_axis,
        solution=solution,
        dx=dx,
        dt=dt_effective,
        factor=factor,
        left_ghost_formula=left_formula,
        right_ghost_formula=right_formula,
        plot_y_limits=plot_y_limits,
    )


def profile_to_state(profile: np.ndarray) -> np.ndarray:
    state = np.zeros(profile.shape[0] + 2, dtype=float)
    state[1:-1] = profile
    return state
