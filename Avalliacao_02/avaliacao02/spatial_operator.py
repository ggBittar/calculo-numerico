"""Operador espacial por diferenças finitas para a condução 2D transiente.

Modelo resolvido:
    dT/dt = alpha * (d²T/dx² + d²T/dy²)

Condições de contorno:
    x=0   : T = 400 K
    x=Lx  : adiabático, dT/dx = 0
    y=0   : convecção, k*dT/dy = h*(T - T_inf)
    y=Ly  : convecção + radiação, -k*dT/dy = h*(T - T_inf) + eps*sigma*(T^4 - Tviz^4)

As condições de convecção/radiação são incorporadas por pontos fantasmas
(ghost points), mantendo segunda ordem espacial no interior.
"""

from __future__ import annotations

from typing import Any

from .config import SimulationConfig


def dx_dy(Nx: int, Ny: int, cfg: SimulationConfig) -> tuple[float, float]:
    """Calcula os espaçamentos de malha."""
    return cfg.geometry.Lx / (Nx - 1), cfg.geometry.Ly / (Ny - 1)


def stable_dt(Nx: int, Ny: int, C: float, cfg: SimulationConfig) -> float:
    """Passo temporal explícito usado no projeto.

    Para problemas difusivos 2D, uma escolha conservadora é:
        dt = C * min(dx, dy)^2 / alpha

    Observação: para Euler explícito na equação de calor 2D, C <= 0,25 costuma
    ser a restrição clássica quando dx=dy. O enunciado pede testes com C=0,5
    para RK4 e AB4; nesses casos, o código permite o valor pedido para estudo.
    """
    dx, dy = dx_dy(Nx, Ny, cfg)
    return C * min(dx, dy) ** 2 / cfg.material.alpha


def create_initial_field(Nx: int, Ny: int, xp: Any, cfg: SimulationConfig):
    """Cria o campo inicial de temperatura e aplica as condições essenciais."""
    T = xp.full((Nx, Ny), cfg.bc.T_initial, dtype=xp.float64)
    enforce_boundary_values_inplace(T, cfg)
    return T


def enforce_boundary_values_inplace(T, cfg: SimulationConfig):
    """Aplica contornos diretamente no próprio array.

    - x=0: temperatura fixa.
    - x=Lx: adiabático aproximado por T[-1, :] = T[-2, :].

    Os contornos em y=0 e y=Ly entram no operador espacial por pontos fantasmas.
    """
    T[0, :] = cfg.bc.T_left
    T[-1, :] = T[-2, :]
    return T


def enforce_boundary_values(T, cfg: SimulationConfig):
    """Retorna uma cópia do campo com os contornos essenciais reaplicados."""
    T_copy = T.copy()
    enforce_boundary_values_inplace(T_copy, cfg)
    return T_copy


def rhs_temperature(T, Nx: int, Ny: int, xp: Any, cfg: SimulationConfig):
    """Calcula F(T)=dT/dt para o campo de temperatura.

    O retorno tem o mesmo formato de T. Nos contornos de Dirichlet/Neumann em x,
    a derivada temporal é zerada e os valores são reaplicados após cada avanço.
    """
    enforce_boundary_values_inplace(T, cfg)

    dx, dy = dx_dy(Nx, Ny, cfg)
    alpha = cfg.material.alpha
    k = cfg.material.k
    bc = cfg.bc

    dTdt = xp.zeros_like(T)

    # Pontos internos em x. Incluímos y=0 e y=Ny-1 em blocos específicos abaixo.
    i = slice(1, Nx - 1)

    with xp.errstate(over="ignore", invalid="ignore"):
        # Região estritamente interna: 1 <= y <= Ny-2.
        j = slice(1, Ny - 1)
        d2Tdx2 = (T[2:Nx, j] - 2.0 * T[1:Nx - 1, j] + T[0:Nx - 2, j]) / dx**2
        d2Tdy2 = (T[i, 2:Ny] - 2.0 * T[i, 1:Ny - 1] + T[i, 0:Ny - 2]) / dy**2
        dTdt[i, j] = alpha * (d2Tdx2 + d2Tdy2)

        # Contorno inferior y=0: convecção.
        # k*dT/dy = h*(T_s - T_inf)
        # Laplaciano em y usando ghost point:
        # d²T/dy² ≈ 2*(T_1 - T_0)/dy² - 2*h*(T_0 - T_inf)/(k*dy)
        j0 = 0
        d2Tdx2_bottom = (T[2:Nx, j0] - 2.0 * T[1:Nx - 1, j0] + T[0:Nx - 2, j0]) / dx**2
        d2Tdy2_bottom = (
            2.0 * (T[i, 1] - T[i, 0]) / dy**2
            - 2.0 * bc.h_bottom * (T[i, 0] - bc.Tinf_bottom) / (k * dy)
        )
        dTdt[i, j0] = alpha * (d2Tdx2_bottom + d2Tdy2_bottom)

        # Contorno superior y=Ly: convecção + radiação.
        # q = h*(T_s - T_inf) + eps*sigma*(T_s^4 - Tviz^4)
        # -k*dT/dy = q
        # d²T/dy² ≈ 2*(T_{Ny-2} - T_{Ny-1})/dy² - 2*q/(k*dy)
        jt = Ny - 1
        Ttop = T[i, jt]
        q_top = bc.h_top * (Ttop - bc.Tinf_top) + bc.emissivity * bc.sigma * (Ttop**4 - bc.T_sur**4)
        d2Tdx2_top = (T[2:Nx, jt] - 2.0 * T[1:Nx - 1, jt] + T[0:Nx - 2, jt]) / dx**2
        d2Tdy2_top = 2.0 * (T[i, Ny - 2] - T[i, jt]) / dy**2 - 2.0 * q_top / (k * dy)
        dTdt[i, jt] = alpha * (d2Tdx2_top + d2Tdy2_top)

    # Garante que x=0 e x=Lx não sejam integrados como incógnitas livres.
    dTdt[0, :] = 0.0
    dTdt[-1, :] = 0.0
    return dTdt


def nearest_index(position: float, length: float, N: int) -> int:
    """Índice do ponto de malha mais próximo de uma coordenada física."""
    if N <= 1:
        raise ValueError("N deve ser maior que 1.")
    idx = round(position / length * (N - 1))
    return int(max(0, min(N - 1, idx)))
