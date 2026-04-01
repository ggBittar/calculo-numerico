from __future__ import annotations

from typing import Any

import numpy as np


SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "pi": np.pi,
}


def evaluate_expression(expression: str, variables: dict[str, Any]) -> Any:
    cleaned = expression.strip()
    if not cleaned:
        raise ValueError("A expressao nao pode ser vazia.")

    try:
        return eval(cleaned, SAFE_GLOBALS, variables)
    except Exception as exc:
        raise ValueError(f"Expressao invalida: {expression}") from exc
