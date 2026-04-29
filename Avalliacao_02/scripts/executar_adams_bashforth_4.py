"""Programa separado para executar o método: ab4.

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
from avaliacao02.stability import limit_c, max_stable_c


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o método ab4 para as malhas da avaliação.")
    parser.add_argument("--C", type=float, default=None, help="Coeficiente C de estabilidade.")
    parser.add_argument("--cpu", action="store_true", help="Força execução em CPU/NumPy.")
    args = parser.parse_args()

    xp, using_cuda = get_array_module(prefer_cuda=not args.cpu)
    cfg = SimulationConfig(save_every_seconds=60.0)
    output_dir = PROJECT_ROOT / "resultados" / "dados"
    requested_c = max_stable_c("ab4") if args.C is None else args.C
    used_c, limited = limit_c("ab4", requested_c)
    if limited:
        print(f"[ajuste] método=ab4 C solicitado={requested_c:g} excede o limite; usando C={used_c:g}.")

    for N in MESHES:
        print(f"Executando método=ab4, Nx=Ny={N}, C={used_c:g}")
        result = run_simulation(
            method="ab4",
            Nx=N,
            Ny=N,
            C=used_c,
            C_requested=requested_c,
            stability_limited=limited,
            xp=xp,
            using_cuda=using_cuda,
            cfg=cfg,
        )
        path = result.save_csv(output_dir)
        print(f"  salvo: {path}")

    print(f"Backend: {'CUDA/CuPy' if using_cuda else 'CPU/NumPy'}")


if __name__ == "__main__":
    main()
