"""
Trabalho de Camada Limite - Placa plana
Aluno: Gustavo Bittar Goncalves

Resolve a equacao de Blasius por shooting + Runge-Kutta de 4a ordem.
Gera a distribuicao de velocidade u/U = f'(eta) e um exemplo dimensional u(y).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt


@dataclass
class BlasiusSolution:
    eta: list[float]
    f: list[float]
    fp: list[float]
    fpp: list[float]
    alpha: float


def rhs(_eta: float, state: tuple[float, float, float]) -> tuple[float, float, float]:
    """Sistema de 1a ordem para a equacao de Blasius.

    y1 = f
    y2 = f'
    y3 = f''

    y1' = y2
    y2' = y3
    y3' = -0.5*y1*y3
    """
    y1, y2, y3 = state
    return (y2, y3, -0.5 * y1 * y3)


def rk4_step(
    fun: Callable[[float, tuple[float, float, float]], tuple[float, float, float]],
    eta: float,
    state: tuple[float, float, float],
    h: float,
) -> tuple[float, float, float]:
    """Executa um passo de Runge-Kutta classico de 4a ordem."""
    k1 = fun(eta, state)
    s2 = tuple(state[i] + 0.5 * h * k1[i] for i in range(3))
    k2 = fun(eta + 0.5 * h, s2)
    s3 = tuple(state[i] + 0.5 * h * k2[i] for i in range(3))
    k3 = fun(eta + 0.5 * h, s3)
    s4 = tuple(state[i] + h * k3[i] for i in range(3))
    k4 = fun(eta + h, s4)
    return tuple(state[i] + (h / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(3))


def integrate(alpha: float, eta_max: float = 10.0, h: float = 0.01) -> BlasiusSolution:
    """Integra o sistema para f''(0)=alpha."""
    n = int(round(eta_max / h))
    eta_values = [0.0]
    f_values = [0.0]
    fp_values = [0.0]
    fpp_values = [alpha]

    eta = 0.0
    state = (0.0, 0.0, alpha)
    for _ in range(n):
        state = rk4_step(rhs, eta, state, h)
        eta += h
        eta_values.append(eta)
        f_values.append(state[0])
        fp_values.append(state[1])
        fpp_values.append(state[2])

    return BlasiusSolution(eta_values, f_values, fp_values, fpp_values, alpha)


def shooting_secant(
    a0: float = 0.30,
    a1: float = 0.35,
    eta_max: float = 10.0,
    h: float = 0.01,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> BlasiusSolution:
    """Ajusta alpha=f''(0) para satisfazer f'(eta_max)=1."""
    sol0 = integrate(a0, eta_max, h)
    sol1 = integrate(a1, eta_max, h)
    r0 = sol0.fp[-1] - 1.0
    r1 = sol1.fp[-1] - 1.0

    for _ in range(max_iter):
        if abs(r1) < tol:
            return sol1
        if abs(r1 - r0) < 1e-15:
            raise RuntimeError("Falha no shooting: residuos quase iguais.")

        a2 = a1 - r1 * (a1 - a0) / (r1 - r0)
        sol2 = integrate(a2, eta_max, h)
        r2 = sol2.fp[-1] - 1.0

        a0, r0 = a1, r1
        a1, r1 = a2, r2
        sol1 = sol2

    raise RuntimeError("Shooting nao convergiu.")


def boundary_layer_thickness_x(x: float, U: float, nu: float, eta_99: float = 5.0) -> float:
    """Estimativa delta_99 ~= eta_99*sqrt(nu*x/U)."""
    return eta_99 * math.sqrt(nu * x / U)


def main() -> None:
    sol = shooting_secant(eta_max=10.0, h=0.01)
    print(f"f''(0) = {sol.alpha:.10f}")
    print(f"f'(eta_max) = {sol.fp[-1]:.10f}")

    # Grafico adimensional: u/U contra eta.
    plt.figure(figsize=(6, 4))
    plt.plot(sol.fp, sol.eta, linewidth=2)
    plt.xlabel(r"$u/U = f'(\eta)$")
    plt.ylabel(r"$\eta = y\sqrt{U/(\nu x)}$")
    plt.title("Perfil de velocidade - solucao de Blasius")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/data/perfil_velocidade_adimensional.png", dpi=200)

    # Exemplo dimensional para x=1 m, U=1 m/s, nu=1e-6 m2/s.
    x = 1.0
    U = 1.0
    nu = 1.0e-6
    y = [eta * math.sqrt(nu * x / U) for eta in sol.eta]
    u = [U * fp for fp in sol.fp]

    plt.figure(figsize=(6, 4))
    plt.plot(u, y, linewidth=2)
    plt.xlabel(r"$u$ [m/s]")
    plt.ylabel(r"$y$ [m]")
    plt.title(r"Perfil dimensional exemplo: $x=1$ m, $U=1$ m/s, $\nu=10^{-6}$ m$^2$/s")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/data/perfil_velocidade_dimensional.png", dpi=200)

    delta99 = boundary_layer_thickness_x(x, U, nu)
    print(f"delta_99 aproximado em x=1 m = {delta99:.6e} m")


if __name__ == "__main__":
    main()
