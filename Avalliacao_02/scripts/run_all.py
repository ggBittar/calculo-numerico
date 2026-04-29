"""Executa todas as simulações principais da Avaliação 02.

Uso a partir da raiz do projeto:
    python scripts/run_all.py --cuda
    python scripts/run_all.py --cpu

Saídas:
    resultados/dados/*.csv
    resultados/dados/resultados_consolidados.csv
    resultados/figuras/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Permite executar o script diretamente sem instalar o pacote.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from avaliacao02.backend import get_array_module
from avaliacao02.config import MESHES, STABILITY_COEFFICIENTS, SimulationConfig
from avaliacao02.parallel import SimulationTask, default_worker_count, run_simulation_tasks
from avaliacao02.plots import plot_mesh_comparison, plot_stability_comparison
from avaliacao02.time_methods import ALL_METHOD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulações da Avaliação 02 com CUDA/CuPy quando disponível.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cuda", action="store_true", help="Tenta usar CUDA via CuPy.")
    group.add_argument("--cpu", action="store_true", help="Força NumPy/CPU.")
    parser.add_argument("--C", type=float, default=None, help="Coeficiente C para a questão 1. Padrão: config.default_C.")
    parser.add_argument("--save-every", type=float, default=60.0, help="Intervalo de amostragem do histórico [s].")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de processos em CPU. Padrão: núcleos lógicos menos um. Em CUDA, é ignorado.",
    )
    return parser.parse_args()


def build_tasks(C_default: float) -> list[SimulationTask]:
    """Monta todos os casos pedidos no enunciado."""
    tasks: list[SimulationTask] = []

    for method in ALL_METHOD_NAMES:
        for N in MESHES:
            tasks.append(
                SimulationTask(
                    label=f"[Q1] método={method:22s} Nx=Ny={N:2d} C={C_default:g}",
                    method=method,
                    Nx=N,
                    Ny=N,
                    C=C_default,
                )
            )

    for C in STABILITY_COEFFICIENTS:
        for N in MESHES:
            tasks.append(
                SimulationTask(
                    label=f"[Q2] método=rk4                    Nx=Ny={N:2d} C={C:g}",
                    method="rk4",
                    Nx=N,
                    Ny=N,
                    C=C,
                )
            )

    for C in STABILITY_COEFFICIENTS:
        for N in MESHES:
            tasks.append(
                SimulationTask(
                    label=f"[Q3] método=ab4                    Nx=Ny={N:2d} C={C:g}",
                    method="ab4",
                    Nx=N,
                    Ny=N,
                    C=C,
                )
            )

    return tasks


def main() -> None:
    args = parse_args()
    prefer_cuda = not args.cpu
    xp, using_cuda = get_array_module(prefer_cuda=prefer_cuda)

    cfg = SimulationConfig(save_every_seconds=args.save_every)
    C_default = cfg.default_C if args.C is None else args.C

    dados_dir = PROJECT_ROOT / "resultados" / "dados"
    figuras_dir = PROJECT_ROOT / "resultados" / "figuras"
    dados_dir.mkdir(parents=True, exist_ok=True)
    figuras_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend numérico: {'CUDA/CuPy' if using_cuda else 'CPU/NumPy'}")
    print(f"Coeficiente C padrão para a questão 1: {C_default:g}")
    if using_cuda:
        print("Execução em CUDA mantida em série para evitar contenção na GPU.")
    else:
        workers = args.workers if args.workers is not None else default_worker_count()
        print(f"Execução em CPU configurada com até {workers} workers.")

    tasks = build_tasks(C_default)
    batch = run_simulation_tasks(tasks, cfg=cfg, xp=xp, using_cuda=using_cuda, workers=args.workers)

    all_frames: list[pd.DataFrame] = []
    for result in batch.results:
        result.save_csv(dados_dir)
        all_frames.append(result.to_dataframe())

    if batch.failures:
        failures_df = pd.DataFrame(
            [
                {
                    "label": failure.label,
                    "metodo": failure.method,
                    "Nx": failure.Nx,
                    "Ny": failure.Ny,
                    "C": failure.C,
                    "tipo_erro": failure.error_type,
                    "mensagem_erro": failure.error_message,
                }
                for failure in batch.failures
            ]
        )
        failures_path = dados_dir / "falhas_execucao.csv"
        failures_df.to_csv(failures_path, index=False)
        print(f"Falhas registradas em: {failures_path}")

    if not all_frames:
        raise RuntimeError("Nenhuma simulação terminou com sucesso; verifique falhas_execucao.csv.")

    consolidated = pd.concat(all_frames, ignore_index=True)
    consolidated_path = dados_dir / "resultados_consolidados.csv"
    consolidated.to_csv(consolidated_path, index=False)
    print(f"CSV consolidado salvo em: {consolidated_path}")

    # Figuras da Questão 1.
    for method in ALL_METHOD_NAMES:
        plot_mesh_comparison(consolidated, method=method, C=C_default, output_dir=figuras_dir)

    # Figuras das Questões 2 e 3.
    for method in ("rk4", "ab4"):
        for C in STABILITY_COEFFICIENTS:
            plot_mesh_comparison(consolidated, method=method, C=C, output_dir=figuras_dir)
        plot_stability_comparison(consolidated, method=method, Nx=20, Ny=20, output_dir=figuras_dir)

    print(f"Figuras salvas em: {figuras_dir}")


if __name__ == "__main__":
    main()
