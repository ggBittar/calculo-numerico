from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    name: str
    description: str
    estimator: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]


def finite_difference_central(values: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(1, len(values) - 1, dtype=float)
    approx = (values[2:] - values[:-2]) / (2.0 * delta)
    return coords, approx


def finite_difference_forward(values: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(0, len(values) - 1, dtype=float)
    approx = (values[1:] - values[:-1]) / delta
    return coords, approx


def finite_difference_backward(values: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(1, len(values), dtype=float)
    approx = (values[1:] - values[:-1]) / delta
    return coords, approx


METHODS: dict[str, MethodSpec] = {
    "central": MethodSpec(
        method_id="central",
        name="Diferença central",
        description="Aproxima a derivada usando pontos anterior e posterior.",
        estimator=finite_difference_central,
    ),
    "forward": MethodSpec(
        method_id="forward",
        name="Diferença progressiva",
        description="Aproxima a derivada usando o ponto atual e o próximo.",
        estimator=finite_difference_forward,
    ),
    "backward": MethodSpec(
        method_id="backward",
        name="Diferença regressiva",
        description="Aproxima a derivada usando o ponto atual e o anterior.",
        estimator=finite_difference_backward,
    ),
}

