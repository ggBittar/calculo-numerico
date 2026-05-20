"""Seleção do backend numérico: CuPy/CUDA quando disponível; NumPy como fallback.

A ideia é escrever o restante do código usando `xp`, que pode ser:
- cupy: executa arrays e operações na GPU NVIDIA com CUDA;
- numpy: executa na CPU, útil para depuração ou máquinas sem GPU/CuPy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_CUDA_PROBE_TIMEOUT_SECONDS = 10.0


def _cuda_probe_timeout_seconds() -> float:
    """Lê o timeout do teste CUDA a partir do ambiente."""
    raw_value = os.getenv(
        "AVALIACAO02_CUDA_PROBE_TIMEOUT",
        str(DEFAULT_CUDA_PROBE_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_value)
    except ValueError:
        print(
            "[aviso] AVALIACAO02_CUDA_PROBE_TIMEOUT inválido; "
            f"usando {DEFAULT_CUDA_PROBE_TIMEOUT_SECONDS:g}s."
        )
        return DEFAULT_CUDA_PROBE_TIMEOUT_SECONDS

    return max(0.5, timeout)


def _probe_cupy_cuda(timeout_seconds: float) -> tuple[bool, str]:
    """Testa CuPy/CUDA em outro processo e força uma pequena operação na GPU."""
    probe_code = "\n".join(
        [
            "import cupy as cp",
            "device_count = cp.cuda.runtime.getDeviceCount()",
            "if device_count < 1:",
            "    raise RuntimeError('nenhum dispositivo CUDA encontrado')",
            "x = cp.arange(16, dtype=cp.float32)",
            "float(x.sum().get())",
            "cp.cuda.Stream.null.synchronize()",
            "print(device_count)",
        ]
    )

    process = subprocess.Popen(
        [sys.executable, "-c", probe_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        return (
            False,
            f"teste de CUDA excedeu {timeout_seconds:g}s. "
            "Aumente AVALIACAO02_CUDA_PROBE_TIMEOUT se a inicialização da GPU for lenta.",
        )
    except KeyboardInterrupt:
        process.kill()
        raise

    if process.returncode != 0:
        details = (stderr or stdout).strip()
        last_line = (
            details.splitlines()[-1]
            if details
            else "falha desconhecida ao inicializar CuPy/CUDA"
        )
        return False, last_line

    device_count = stdout.strip().splitlines()[-1] if stdout.strip() else "0"
    return True, f"{device_count} dispositivo(s) CUDA encontrados."


def _run_nvidia_modprobe(timeout_seconds: float) -> None:
    """Pede ao driver NVIDIA para criar /dev/nvidia* quando o udev não criou."""
    if shutil.which("nvidia-modprobe") is None:
        return

    commands = (
        ["nvidia-modprobe", "-c=0"],
        ["nvidia-modprobe", "-u"],
    )
    for command in commands:
        try:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def _raise_cuda_required(reason: str) -> None:
    raise RuntimeError(
        "CUDA foi solicitado explicitamente com --cuda, mas a GPU não passou no teste.\n"
        f"Motivo: {reason}\n"
        "Confira se `nvidia-smi` funciona e se existem /dev/nvidia0, /dev/nvidiactl e /dev/nvidia-uvm."
    )


def get_array_module(prefer_cuda: bool = True, require_cuda: bool = False) -> tuple[Any, bool]:
    """Retorna o módulo de arrays e um booleano indicando se CUDA está ativo.

    Parameters
    ----------
    prefer_cuda:
        Se True, tenta importar CuPy. Se False, força NumPy.
    require_cuda:
        Se True, falha quando CUDA não estiver funcional em vez de cair para CPU.

    Returns
    -------
    xp:
        Módulo cupy ou numpy.
    using_cuda:
        True quando `xp` é cupy; False quando `xp` é numpy.
    """
    force_cpu = os.getenv("AVALIACAO02_FORCE_CPU", "0") == "1"
    if force_cpu and require_cuda:
        _raise_cuda_required("AVALIACAO02_FORCE_CPU=1 está ativo.")

    if prefer_cuda and not force_cpu:
        timeout_seconds = _cuda_probe_timeout_seconds()
        _run_nvidia_modprobe(timeout_seconds=timeout_seconds)
        cuda_ok, reason = _probe_cupy_cuda(timeout_seconds)
        if not cuda_ok:
            if require_cuda:
                _raise_cuda_required(reason)
            print("[aviso] CuPy/CUDA indisponível; usando NumPy na CPU.")
            print(f"        Motivo: {reason}")
        else:
            try:
                import cupy as cp  # type: ignore

                return cp, True
            except Exception as exc:  # pragma: no cover - depende do ambiente local
                if require_cuda:
                    _raise_cuda_required(str(exc))
                print("[aviso] CuPy/CUDA passou no teste, mas falhou ao importar no processo principal.")
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
