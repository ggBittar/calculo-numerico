"""Heurísticas conservadoras de estabilidade para os métodos temporais."""

from __future__ import annotations


MAX_STABLE_C = {
    "euler": 0.25,
    "rk2_ponto_medio": 0.25,
    "rk2_euler_modificado": 0.25,
    "rk2_heun": 0.25,
    "rk4": 0.34,
    "ab2": 0.12,
    "ab4": 0.03,
}


def max_stable_c(method: str) -> float:
    """Retorna um teto conservador para o coeficiente C do método."""
    try:
        return MAX_STABLE_C[method]
    except KeyError as exc:
        raise ValueError(f"Método sem limite de estabilidade cadastrado: {method}") from exc


def limit_c(method: str, requested_c: float) -> tuple[float, bool]:
    """Limita C quando o valor solicitado excede a faixa estável do método."""
    used_c = min(requested_c, max_stable_c(method))
    return used_c, used_c < requested_c


def scale_stability_coefficients(method: str, requested_values: list[float] | tuple[float, ...]) -> list[tuple[float, float]]:
    """Reescala uma sequência de C para a faixa estável, preservando proporções."""
    values = list(requested_values)
    if not values:
        return []

    scale = min(1.0, max_stable_c(method) / max(values))
    return [(requested, requested * scale) for requested in values]
