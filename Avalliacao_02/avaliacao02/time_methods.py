"""Métodos explícitos de avanço temporal pedidos na Avaliação 02."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

Array = Any
RhsFunction = Callable[[Array], Array]


def euler_step(T: Array, dt: float, f: RhsFunction) -> Array:
    """Euler explícito de primeira ordem."""
    return T + dt * f(T)


def rk2_midpoint_step(T: Array, dt: float, f: RhsFunction) -> Array:
    """Runge-Kutta de 2ª ordem: método do ponto médio.

    k1 = f(T_n)
    k2 = f(T_n + dt*k1/2)
    T_{n+1} = T_n + dt*k2
    """
    k1 = f(T)
    k2 = f(T + 0.5 * dt * k1)
    return T + dt * k2


def rk2_modified_euler_step(T: Array, dt: float, f: RhsFunction) -> Array:
    """Runge-Kutta de 2ª ordem: Euler modificado/Ralston.

    Aqui foi usada a forma clássica de Ralston:
    k1 = f(T_n)
    k2 = f(T_n + 2*dt*k1/3)
    T_{n+1} = T_n + dt*(k1/4 + 3*k2/4)
    """
    k1 = f(T)
    k2 = f(T + (2.0 / 3.0) * dt * k1)
    return T + dt * (0.25 * k1 + 0.75 * k2)


def rk2_heun_step(T: Array, dt: float, f: RhsFunction) -> Array:
    """Runge-Kutta de 2ª ordem: método de Heun/trapézio explícito."""
    k1 = f(T)
    k2 = f(T + dt * k1)
    return T + 0.5 * dt * (k1 + k2)


def rk4_step(T: Array, dt: float, f: RhsFunction) -> Array:
    """Runge-Kutta clássico de 4ª ordem."""
    k1 = f(T)
    k2 = f(T + 0.5 * dt * k1)
    k3 = f(T + 0.5 * dt * k2)
    k4 = f(T + dt * k3)
    return T + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


STEP_METHODS = {
    "euler": euler_step,
    "rk2_ponto_medio": rk2_midpoint_step,
    "rk2_euler_modificado": rk2_modified_euler_step,
    "rk2_heun": rk2_heun_step,
    "rk4": rk4_step,
}


class AdamsBashforthStepper:
    """Integrador explícito de Adams-Bashforth de ordem 2 ou 4.

    Como AB é multípasso, os primeiros passos são inicializados com RK4 para
    gerar histórico de derivadas com boa precisão.
    """

    def __init__(self, order: int):
        if order not in (2, 4):
            raise ValueError("A ordem de Adams-Bashforth deve ser 2 ou 4.")
        self.order = order
        self.derivatives = deque(maxlen=order)

    @property
    def name(self) -> str:
        return f"ab{self.order}"

    def reset(self) -> None:
        """Limpa o histórico de derivadas."""
        self.derivatives.clear()

    def step(self, T: Array, dt: float, f: RhsFunction) -> Array:
        """Executa um passo AB2 ou AB4, inicializando com RK4 se necessário."""
        fn = f(T)
        self.derivatives.appendleft(fn)

        # Enquanto não há histórico suficiente, usa RK4 como método de partida.
        if len(self.derivatives) < self.order:
            return rk4_step(T, dt, f)

        if self.order == 2:
            f_n, f_nm1 = self.derivatives[0], self.derivatives[1]
            return T + dt * (1.5 * f_n - 0.5 * f_nm1)

        # AB4: T_{n+1}=T_n+dt/24*(55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})
        f_n, f_nm1, f_nm2, f_nm3 = self.derivatives[0], self.derivatives[1], self.derivatives[2], self.derivatives[3]
        return T + (dt / 24.0) * (55.0 * f_n - 59.0 * f_nm1 + 37.0 * f_nm2 - 9.0 * f_nm3)


ALL_METHOD_NAMES = (
    "euler",
    "rk2_ponto_medio",
    "rk2_euler_modificado",
    "rk2_heun",
    "rk4",
    "ab2",
    "ab4",
)
