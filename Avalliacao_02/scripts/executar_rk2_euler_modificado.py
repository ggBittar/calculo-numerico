"""Programa separado para executar o método: rk2_euler_modificado.

Executa as malhas Nx=Ny=10,20,40,80 usando C=0.25 por padrão.
Use --C para alterar o coeficiente.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from avaliacao02.backend import get_array_module
from avaliacao02.config import MESHES, SimulationConfig
from avaliacao02.solver import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o método rk2_euler_modificado para as malhas da avaliação.")
    parser.add_argument("--C", type=float, default=0.25, help="Coeficiente C de estabilidade.")
    parser.add_argument("--cpu", action="store_true", help="Força execução em CPU/NumPy.")
    args = parser.parse_args()

    xp, using_cuda = get_array_module(prefer_cuda=not args.cpu)
    cfg = SimulationConfig(save_every_seconds=60.0)
    output_dir = PROJECT_ROOT / "resultados" / "dados"

    for N in MESHES:
        print(f"Executando método=rk2_euler_modificado, Nx=Ny={N}, C={args.C:g}")
        result = run_simulation(method="rk2_euler_modificado", Nx=N, Ny=N, C=args.C, xp=xp, using_cuda=using_cuda, cfg=cfg)
        path = result.save_csv(output_dir)
        print(f"  salvo: {path}")

    print(f"Backend: {'CUDA/CuPy' if using_cuda else 'CPU/NumPy'}")


if __name__ == "__main__":
    main()
