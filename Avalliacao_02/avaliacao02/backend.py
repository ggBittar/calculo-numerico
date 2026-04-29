"""Seleção do backend numérico: CuPy/CUDA quando disponível; NumPy como fallback.

A ideia é escrever o restante do código usando `xp`, que pode ser:
- cupy: executa arrays e operações na GPU NVIDIA com CUDA;
- numpy: executa na CPU, útil para depuração ou máquinas sem GPU/CuPy.
"""

from __future__ import annotations

import os
from typing import Any


def get_array_module(prefer_cuda: bool = True) -> tuple[Any, bool]:
    """Retorna o módulo de arrays e um booleano indicando se CUDA está ativo.

    Parameters
    ----------
    prefer_cuda:
        Se True, tenta importar CuPy. Se False, força NumPy.

    Returns
    -------
    xp:
        Módulo cupy ou numpy.
    using_cuda:
        True quando `xp` é cupy; False quando `xp` é numpy.
    """
    force_cpu = os.getenv("AVALIACAO02_FORCE_CPU", "0") == "1"
    if prefer_cuda and not force_cpu:
        try:
            import cupy as cp  # type: ignore

            # Pequeno teste para confirmar que há dispositivo CUDA acessível.
            _ = cp.cuda.runtime.getDeviceCount()
            return cp, True
        except Exception as exc:  # pragma: no cover - depende do ambiente local
            print("[aviso] CuPy/CUDA indisponível; usando NumPy na CPU.")
            print(f"        Motivo: {exc}")

    import numpy as np

    return np, False


def to_numpy(array: Any):
    """Converte array CuPy/NumPy para NumPy, necessário para salvar CSV e plotar."""
    try:
        import cupy as cp  # type: ignore

        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except Exception:
        pass
    return array


def to_scalar(value: Any):
    """Converte um escalar CuPy/NumPy/Python para tipo nativo do Python."""
    try:
        return value.item()
    except Exception:
        return value
