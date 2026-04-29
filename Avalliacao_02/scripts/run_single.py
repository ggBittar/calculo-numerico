"""Executa uma única simulação, útil para testes rápidos e depuração.

Exemplos:
    python scripts/run_single.py --method rk4 --N 20 --C 0.25 --cuda
    python scripts/run_single.py --method ab4 --N 80 --C 0.125 --cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from avaliacao02.backend import get_array_module
from avaliacao02.config import SimulationConfig
from avaliacao02.solver import run_simulation
from avaliacao02.time_methods import ALL_METHOD_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa uma simulação isolada.")
    parser.add_argument("--method", choices=ALL_METHOD_NAMES, default="rk4", help="Método temporal.")
    parser.add_argument("--N", type=int, default=20, help="Usa Nx=Ny=N.")
    parser.add_argument("--C", type=float, default=0.25, help="Coeficiente C de estabilidade.")
    parser.add_argument("--cpu", action="store_true", help="Força CPU/NumPy.")
    parser.add_argument("--cuda", action="store_true", help="Tenta usar CUDA/CuPy.")
    args = parser.parse_args()

    xp, using_cuda = get_array_module(prefer_cuda=not args.cpu)
    cfg = SimulationConfig(save_every_seconds=60.0)

    result = run_simulation(
        method=args.method,
        Nx=args.N,
        Ny=args.N,
        C=args.C,
        xp=xp,
        using_cuda=using_cuda,
        cfg=cfg,
    )

    output_dir = PROJECT_ROOT / "resultados" / "dados"
    path = result.save_csv(output_dir)
    print(result.to_dataframe().tail())
    print(f"Arquivo salvo em: {path}")
    print(f"Backend: {'CUDA/CuPy' if using_cuda else 'CPU/NumPy'}")


if __name__ == "__main__":
    main()
