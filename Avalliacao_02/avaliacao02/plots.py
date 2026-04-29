"""Geração de gráficos solicitados na Avaliação 02."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


POINT_COLUMNS = {
    "centro": ("T_Lx2_Ly2_K", "x=Lx/2, y=Ly/2"),
    "quarto": ("T_Lx4_Ly4_K", "x=Lx/4, y=Ly/4"),
}


def plot_mesh_comparison(df: pd.DataFrame, method: str, C: float, output_dir: Path) -> list[Path]:
    """Gera, para um método e C, os gráficos comparando as malhas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    subset = df[(df["metodo"] == method) & (df["C"] == C)].copy()
    if subset.empty:
        return paths

    for point_name, (col, label) in POINT_COLUMNS.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for Nx in sorted(subset["Nx"].unique()):
            local = subset[subset["Nx"] == Nx]
            ax.plot(local["t_h"], local[col], label=f"Nx=Ny={Nx}")
        ax.set_xlabel("Tempo [h]")
        ax.set_ylabel("Temperatura [K]")
        ax.set_title(f"Evolução temporal em {label}\nMétodo: {method}, C={C:g}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"evolucao_{method}_{point_name}_C{C:g}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_stability_comparison(df: pd.DataFrame, method: str, Nx: int, Ny: int, output_dir: Path) -> list[Path]:
    """Compara C=0,5; 0,25; 0,125 para um método e malha fixa."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    subset = df[(df["metodo"] == method) & (df["Nx"] == Nx) & (df["Ny"] == Ny)].copy()
    if subset.empty:
        return paths

    for point_name, (col, label) in POINT_COLUMNS.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for C in sorted(subset["C"].unique(), reverse=True):
            local = subset[subset["C"] == C]
            ax.plot(local["t_h"], local[col], label=f"C={C:g}")
        ax.set_xlabel("Tempo [h]")
        ax.set_ylabel("Temperatura [K]")
        ax.set_title(f"Comparação de coeficientes C em {label}\nMétodo: {method}, Nx=Ny={Nx}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"comparacao_C_{method}_{point_name}_Nx{Nx}_Ny{Ny}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return paths
