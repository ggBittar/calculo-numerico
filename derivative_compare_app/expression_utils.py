from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

VARIABLES = ["x", "y", "z", "t"]
SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "where": np.where,
}


def parse_n_values(text: str) -> List[int]:
    values = []
    for token in [p.strip() for p in text.split(",") if p.strip()]:
        n = int(token)
        if n < 2:
            raise ValueError("Cada refinamento precisa ter pelo menos 2 pontos.")
        values.append(n)
    if not values:
        raise ValueError("Informe ao menos um número de elementos para os refinamentos.")
    return sorted(set(values))


def parse_positions(text: str, axis_values: np.ndarray) -> List[int]:
    max_index = len(axis_values) - 1
    raw_positions = [p.strip() for p in text.split(",") if p.strip()]
    positions: List[int] = []
    for token in raw_positions:
        if any(ch in token for ch in ".eE"):
            coord = float(token)
            idx = int(np.argmin(np.abs(axis_values - coord)))
        else:
            idx_int = int(token)
            if 0 <= idx_int <= max_index:
                idx = idx_int
            else:
                coord = float(token)
                idx = int(np.argmin(np.abs(axis_values - coord)))
        positions.append(idx)
    if not positions:
        positions = [0, max_index // 2, max_index]
    return sorted(set(min(max(0, p), max_index) for p in positions))


def evaluate_expression(expr: str, local_vars: Dict[str, np.ndarray]) -> np.ndarray:
    result = eval(expr, SAFE_GLOBALS, local_vars)
    return np.asarray(result, dtype=float)


def build_grid(configs: Dict[str, dict], n_value: int) -> Tuple[List[str], Dict[str, np.ndarray], List[float], Dict[str, np.ndarray]]:
    axis_order: List[str] = []
    axes_vectors: Dict[str, np.ndarray] = {}
    spacings: List[float] = []
    for var in VARIABLES:
        cfg = configs[var]
        if not cfg["enabled"]:
            continue
        if cfg["max"] <= cfg["min"]:
            raise ValueError(f"No domínio de {var}, o máximo deve ser maior que o mínimo.")
        vec = np.linspace(cfg["min"], cfg["max"], n_value)
        axis_order.append(var)
        axes_vectors[var] = vec
        spacings.append(vec[1] - vec[0])
    if not axis_order:
        raise ValueError("Habilite ao menos uma variável do domínio.")

    mesh = np.meshgrid(*[axes_vectors[v] for v in axis_order], indexing="ij")
    locals_map = {v: np.array(0.0) for v in VARIABLES}
    for var, grid in zip(axis_order, mesh):
        locals_map[var] = grid
    return axis_order, axes_vectors, spacings, locals_map


def expand_scalar(arr: np.ndarray, axis_order: List[str], axes_vectors: Dict[str, np.ndarray]) -> np.ndarray:
    if arr.shape != ():
        return arr
    shape = [len(axes_vectors[v]) for v in axis_order]
    return np.full(shape, float(arr), dtype=float)


def extract_line(array: np.ndarray, axis_order: List[str], plot_var: str, configs: Dict[str, dict], n_value: int) -> np.ndarray:
    selectors = []
    for var in axis_order:
        if var == plot_var:
            selectors.append(slice(None))
            continue
        ratio = configs[var]["slice_ratio"]
        idx = min(max(0, int(round(ratio * (n_value - 1)))), n_value - 1)
        selectors.append(idx)
    return array[tuple(selectors)]
