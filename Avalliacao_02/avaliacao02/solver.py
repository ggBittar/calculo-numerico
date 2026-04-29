"""Rotina principal de simulação."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .backend import to_scalar
from .config import SimulationConfig
from .spatial_operator import (
    create_initial_field,
    enforce_boundary_values_inplace,
    nearest_index,
    rhs_temperature,
    stable_dt,
)
from .time_methods import AdamsBashforthStepper, STEP_METHODS


@dataclass
class SimulationResult:
    """Resultado reduzido de uma simulação.

    Armazena apenas as séries temporais nos pontos pedidos pelo enunciado, evitando
    gravar todos os campos em memória para malhas grandes.
    """

    method: str
    Nx: int
    Ny: int
    C: float
    dt: float
    using_cuda: bool
    times: list[float]
    T_center: list[float]
    T_quarter: list[float]

    def to_dataframe(self) -> pd.DataFrame:
        """Converte o resultado para DataFrame."""
        return pd.DataFrame(
            {
                "t_s": self.times,
                "t_h": [t / 3600.0 for t in self.times],
                "T_Lx2_Ly2_K": self.T_center,
                "T_Lx4_Ly4_K": self.T_quarter,
                "metodo": self.method,
                "Nx": self.Nx,
                "Ny": self.Ny,
                "C": self.C,
                "dt_s": self.dt,
                "cuda": self.using_cuda,
            }
        )

    def save_csv(self, output_dir: Path) -> Path:
        """Salva as séries em CSV."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"serie_{self.method}_Nx{self.Nx}_Ny{self.Ny}_C{self.C:g}.csv"
        self.to_dataframe().to_csv(path, index=False)
        return path


def _make_stepper(method: str):
    """Cria função/objeto de avanço temporal a partir do nome."""
    if method in STEP_METHODS:
        return STEP_METHODS[method]
    if method == "ab2":
        return AdamsBashforthStepper(order=2)
    if method == "ab4":
        return AdamsBashforthStepper(order=4)
    raise ValueError(f"Método desconhecido: {method}")


def run_simulation(
    *,
    method: str,
    Nx: int,
    Ny: int,
    C: float,
    xp: Any,
    using_cuda: bool,
    cfg: SimulationConfig,
) -> SimulationResult:
    """Executa uma simulação para um método, malha e coeficiente C."""
    dt_base = stable_dt(Nx, Ny, C, cfg)
    n_steps = int(cfg.t_final // dt_base) + 1
    dt = cfg.t_final / n_steps  # ajusta para terminar exatamente em 4 h

    T = create_initial_field(Nx, Ny, xp, cfg)

    ix_center = nearest_index(cfg.geometry.Lx / 2.0, cfg.geometry.Lx, Nx)
    iy_center = nearest_index(cfg.geometry.Ly / 2.0, cfg.geometry.Ly, Ny)
    ix_quarter = nearest_index(cfg.geometry.Lx / 4.0, cfg.geometry.Lx, Nx)
    iy_quarter = nearest_index(cfg.geometry.Ly / 4.0, cfg.geometry.Ly, Ny)

    save_stride = max(1, int(round(cfg.save_every_seconds / dt)))

    times: list[float] = []
    T_center: list[float] = []
    T_quarter: list[float] = []

    def rhs(local_T):
        return rhs_temperature(local_T, Nx, Ny, xp, cfg)

    stepper = _make_stepper(method)

    def save_sample(step: int, current_T) -> None:
        t = step * dt
        times.append(float(t))
        T_center.append(float(to_scalar(current_T[ix_center, iy_center])))
        T_quarter.append(float(to_scalar(current_T[ix_quarter, iy_quarter])))

    save_sample(0, T)

    for step in range(1, n_steps + 1):
        if isinstance(stepper, AdamsBashforthStepper):
            T = stepper.step(T, dt, rhs)
        else:
            T = stepper(T, dt, rhs)
        enforce_boundary_values_inplace(T, cfg)

        # Detecta instabilidade de forma simples.
        if bool(to_scalar(xp.any(~xp.isfinite(T)))):
            raise FloatingPointError(
                f"Solução não finita em method={method}, Nx={Nx}, Ny={Ny}, C={C}, step={step}."
            )

        if step % save_stride == 0 or step == n_steps:
            save_sample(step, T)

    return SimulationResult(
        method=method,
        Nx=Nx,
        Ny=Ny,
        C=C,
        dt=dt,
        using_cuda=using_cuda,
        times=times,
        T_center=T_center,
        T_quarter=T_quarter,
    )
