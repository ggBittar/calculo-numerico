"""Orquestra a execução sequencial ou paralela das simulações."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

from .config import SimulationConfig
from .solver import SimulationResult, run_simulation


@dataclass(frozen=True)
class SimulationTask:
    """Define um caso independente da avaliação."""

    label: str
    method: str
    Nx: int
    Ny: int
    C: float


@dataclass(frozen=True)
class SimulationFailure:
    """Representa um caso que falhou durante a execução."""

    label: str
    method: str
    Nx: int
    Ny: int
    C: float
    error_type: str
    error_message: str


@dataclass(frozen=True)
class BatchExecutionResult:
    """Agrupa os casos concluídos e os casos que falharam."""

    results: list[SimulationResult]
    failures: list[SimulationFailure]


def default_worker_count() -> int:
    """Escolhe uma quantidade conservadora de workers para CPU."""
    cpu_total = os.cpu_count() or 1
    return max(1, cpu_total - 1)


def _run_cpu_task(task: SimulationTask, cfg: SimulationConfig) -> SimulationResult:
    """Executa um caso isolado em NumPy, adequado para subprocessos."""
    import numpy as np

    return run_simulation(
        method=task.method,
        Nx=task.Nx,
        Ny=task.Ny,
        C=task.C,
        xp=np,
        using_cuda=False,
        cfg=cfg,
    )


def run_simulation_tasks(
    tasks: Iterable[SimulationTask],
    *,
    cfg: SimulationConfig,
    xp,
    using_cuda: bool,
    workers: int | None = None,
) -> BatchExecutionResult:
    """Executa uma lista de casos preservando a ordem original.

    Em CPU/NumPy, cada caso é independente e pode rodar em um processo separado.
    Em CUDA/CuPy, mantemos execução sequencial para evitar contenção do dispositivo
    e overhead de múltiplos contextos.
    """
    task_list = list(tasks)
    if not task_list:
        return BatchExecutionResult(results=[], failures=[])

    if using_cuda:
        results: list[SimulationResult] = []
        failures: list[SimulationFailure] = []
        for task in task_list:
            print(task.label)
            try:
                results.append(
                    run_simulation(
                        method=task.method,
                        Nx=task.Nx,
                        Ny=task.Ny,
                        C=task.C,
                        xp=xp,
                        using_cuda=True,
                        cfg=cfg,
                    )
                )
            except Exception as exc:
                failures.append(
                    SimulationFailure(
                        label=task.label,
                        method=task.method,
                        Nx=task.Nx,
                        Ny=task.Ny,
                        C=task.C,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                print(f"{task.label} [falhou: {type(exc).__name__}]")
        return BatchExecutionResult(results=results, failures=failures)

    max_workers = workers if workers is not None else default_worker_count()
    max_workers = max(1, min(max_workers, len(task_list)))

    if max_workers == 1:
        results = []
        failures: list[SimulationFailure] = []
        for task in task_list:
            print(task.label)
            try:
                results.append(_run_cpu_task(task, cfg))
            except Exception as exc:
                failures.append(
                    SimulationFailure(
                        label=task.label,
                        method=task.method,
                        Nx=task.Nx,
                        Ny=task.Ny,
                        C=task.C,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                print(f"{task.label} [falhou: {type(exc).__name__}]")
        return BatchExecutionResult(results=results, failures=failures)

    ordered_results: list[SimulationResult | None] = [None] * len(task_list)
    failures: list[SimulationFailure] = []
    print(f"Executando {len(task_list)} casos em paralelo na CPU com {max_workers} workers.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_run_cpu_task, task, cfg): index
            for index, task in enumerate(task_list)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            task = task_list[index]
            try:
                result = future.result()
                ordered_results[index] = result
                print(f"{task.label} [concluido]")
            except Exception as exc:
                failures.append(
                    SimulationFailure(
                        label=task.label,
                        method=task.method,
                        Nx=task.Nx,
                        Ny=task.Ny,
                        C=task.C,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                print(f"{task.label} [falhou: {type(exc).__name__}]")

    return BatchExecutionResult(
        results=[result for result in ordered_results if result is not None],
        failures=failures,
    )
