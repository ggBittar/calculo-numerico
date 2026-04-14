from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryConfig:
    emissividade: float
    sigma: float
    h: float
    fluxo_superficial: float
    contato: float
    temperatura_vizinhanca: float
    temperatura_infinito: float
    temperatura_contorno: float


@dataclass(frozen=True)
class ThermalConfig:
    area: float
    condutividade: float
    temperatura_inicial: float
    tempo_inicial: float
    tempo_final: float
    co: float
    x_inicial: float
    x_final: float
    nx: int
    esquerda: BoundaryConfig
    direita: BoundaryConfig


@dataclass(frozen=True)
class SimulationResult:
    config: ThermalConfig
    x_axis: np.ndarray
    t_axis: np.ndarray
    solution: np.ndarray
    dx: float
    dt: float
    factor: float
    left_ghost_formula: str
    right_ghost_formula: str
    plot_y_limits: tuple[float, float]
