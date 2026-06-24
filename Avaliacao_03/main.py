from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resultados"
FIGURES = RESULTS / "figuras"
DATA = RESULTS / "dados"


@dataclass(frozen=True)
class HeatConfig:
    lx: float = 1.0
    ly: float = 1.0
    k: float = 237.0
    rho: float = 2700.0
    cp: float = 900.0
    t_initial: float = 300.0
    t_left: float = 400.0
    t_right: float = 500.0
    t_bottom: float = 600.0
    t_top: float = 200.0

    @property
    def alpha(self) -> float:
        return self.k / (self.rho * self.cp)


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)


def initialize_temperature(nx: int, ny: int, cfg: HeatConfig) -> np.ndarray:
    temp = np.full((ny, nx), cfg.t_initial, dtype=float)
    apply_heat_boundaries(temp, cfg)
    return temp


def apply_heat_boundaries(temp: np.ndarray, cfg: HeatConfig) -> None:
    temp[:, 0] = cfg.t_left
    temp[:, -1] = cfg.t_right
    temp[0, :] = cfg.t_bottom
    temp[-1, :] = cfg.t_top
    temp[0, 0] = 0.5 * (cfg.t_bottom + cfg.t_left)
    temp[0, -1] = 0.5 * (cfg.t_bottom + cfg.t_right)
    temp[-1, 0] = 0.5 * (cfg.t_top + cfg.t_left)
    temp[-1, -1] = 0.5 * (cfg.t_top + cfg.t_right)


def gauss_seidel_heat_step(
    previous: np.ndarray,
    rx: float,
    ry: float,
    cfg: HeatConfig,
    tol: float = 1.0e-8,
    max_iter: int = 8000,
) -> tuple[np.ndarray, int, float]:
    current = previous.copy()
    denom = 1.0 + 2.0 * rx + 2.0 * ry
    jj, ii = np.indices((current.shape[0] - 2, current.shape[1] - 2))
    red = (ii + jj) % 2 == 0
    black = ~red
    last_error = np.inf

    for iteration in range(1, max_iter + 1):
        old_interior = current[1:-1, 1:-1].copy()
        interior = current[1:-1, 1:-1]

        candidate = (
            previous[1:-1, 1:-1]
            + rx * (current[1:-1, :-2] + current[1:-1, 2:])
            + ry * (current[:-2, 1:-1] + current[2:, 1:-1])
        ) / denom
        interior[red] = candidate[red]

        candidate = (
            previous[1:-1, 1:-1]
            + rx * (current[1:-1, :-2] + current[1:-1, 2:])
            + ry * (current[:-2, 1:-1] + current[2:, 1:-1])
        ) / denom
        interior[black] = candidate[black]

        apply_heat_boundaries(current, cfg)
        last_error = float(np.max(np.abs(current[1:-1, 1:-1] - old_interior)))
        if last_error < tol:
            return current, iteration, last_error
    return current, max_iter, last_error


def simulate_heat(
    nx: int,
    ny: int,
    c_stability: float,
    target_times: list[float],
    cfg: HeatConfig,
) -> dict[str, object]:
    dx = cfg.lx / (nx - 1)
    dy = cfg.ly / (ny - 1)
    dt_base = c_stability * min(dx, dy) ** 2 / cfg.alpha
    final_time = max(target_times)
    n_steps = int(np.ceil(final_time / dt_base))
    dt = final_time / n_steps
    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2

    x = np.linspace(0.0, cfg.lx, nx)
    y = np.linspace(0.0, cfg.ly, ny)
    snapshots: dict[float, np.ndarray] = {0.0: initialize_temperature(nx, ny, cfg)}
    temp = snapshots[0.0].copy()
    target_times_sorted = sorted(set(target_times))
    target_index = 1 if target_times_sorted and target_times_sorted[0] == 0.0 else 0
    center_history: list[tuple[float, float, float]] = [(0.0, value_at(temp, x, y, cfg.lx / 2, cfg.ly / 2), value_at(temp, x, y, cfg.lx / 4, cfg.ly / 4))]
    gs_iterations: list[int] = []

    for step in range(1, n_steps + 1):
        temp, iterations, _ = gauss_seidel_heat_step(temp, rx, ry, cfg)
        gs_iterations.append(iterations)
        time = step * dt
        center_history.append((time, value_at(temp, x, y, cfg.lx / 2, cfg.ly / 2), value_at(temp, x, y, cfg.lx / 4, cfg.ly / 4)))
        while target_index < len(target_times_sorted) and time >= target_times_sorted[target_index] - 0.5 * dt:
            snapshots[target_times_sorted[target_index]] = temp.copy()
            target_index += 1

    return {
        "x": x,
        "y": y,
        "temperature": temp,
        "snapshots": snapshots,
        "history": np.array(center_history),
        "dt": dt,
        "rx": rx,
        "ry": ry,
        "mean_gs_iterations": float(np.mean(gs_iterations)) if gs_iterations else 0.0,
        "max_gs_iterations": int(np.max(gs_iterations)) if gs_iterations else 0,
    }


def value_at(field: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> float:
    i = int(np.argmin(np.abs(x - x0)))
    j = int(np.argmin(np.abs(y - y0)))
    return float(field[j, i])


def plot_heat_snapshots(result: dict[str, object], c_stability: float) -> None:
    x = result["x"]
    y = result["y"]
    snapshots = result["snapshots"]
    times = [0.0, 1800.0, 3600.0, 7200.0, 10800.0, 14400.0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    levels = np.linspace(200, 600, 41)
    for ax, time in zip(axes.ravel(), times):
        field = snapshots[time]
        contour = ax.contourf(x, y, field, levels=levels, cmap="inferno")
        ax.set_title(f"t = {time / 3600:.1f} h")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
    fig.colorbar(contour, ax=axes.ravel().tolist(), label="Temperatura (K)")
    fig.suptitle(f"Conducao 2D implicita - Nx=Ny=10, C={c_stability:g}")
    fig.savefig(FIGURES / f"calor_campos_C{c_stability:g}.png", dpi=180)
    plt.close(fig)


def nearest_profile(field: np.ndarray, x: np.ndarray, y: np.ndarray, *, x0: float | None = None, y0: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    if x0 is not None:
        i = int(np.argmin(np.abs(x - x0)))
        return y, field[:, i], x[i]
    if y0 is not None:
        j = int(np.argmin(np.abs(y - y0)))
        return x, field[j, :], y[j]
    raise ValueError("Informe x0 ou y0.")


def plot_heat_mesh_profiles(results: dict[int, dict[str, object]], cfg: HeatConfig) -> None:
    requests = [
        ("perfil_x_meio", "x = Lx/2", {"x0": cfg.lx / 2}, "y (m)"),
        ("perfil_x_quarto", "x = Lx/4", {"x0": cfg.lx / 4}, "y (m)"),
        ("perfil_y_meio", "y = Ly/2", {"y0": cfg.ly / 2}, "x (m)"),
        ("perfil_y_3quartos", "y = 3Ly/4", {"y0": 3 * cfg.ly / 4}, "x (m)"),
    ]
    for filename, title, selector, xlabel in requests:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for n, result in results.items():
            coord, values, actual = nearest_profile(result["temperature"], result["x"], result["y"], **selector)
            ax.plot(coord, values, marker="o", markersize=3, label=f"Nx=Ny={n}")
        ax.set_title(f"Perfil de temperatura em t=4 h - {title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Temperatura (K)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(FIGURES / f"calor_{filename}.png", dpi=180)
        plt.close(fig)


def plot_heat_temporal(results: dict[int, dict[str, object]]) -> None:
    for column, filename, title in [(1, "centro", "x=Lx/2, y=Ly/2"), (2, "quarto", "x=Lx/4, y=Ly/4")]:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for n, result in results.items():
            history = result["history"]
            ax.plot(history[:, 0] / 3600.0, history[:, column], label=f"Nx=Ny={n}")
        ax.set_title(f"Evolucao temporal da temperatura - {title}")
        ax.set_xlabel("Tempo (h)")
        ax.set_ylabel("Temperatura (K)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(FIGURES / f"calor_evolucao_{filename}.png", dpi=180)
        plt.close(fig)


@dataclass(frozen=True)
class WaveConfig:
    length: float = 1.0
    c: float = 1.0
    amplitude: float = 0.01
    final_time: float = 2.0
    nx: int = 101
    cfl: float = 0.4


def second_derivative_matrix(n_interior: int, dx: float) -> np.ndarray:
    diagonal = -2.0 * np.ones(n_interior)
    off = np.ones(n_interior - 1)
    return (np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)) / dx**2


def wave_initial_state(cfg: WaveConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, cfg.length, cfg.nx)
    y = cfg.amplitude * np.sin(np.pi * x / cfg.length)
    v = np.zeros_like(y)
    y[0] = y[-1] = 0.0
    return x, y, v


def wave_rhs(y_int: np.ndarray, v_int: np.ndarray, d2: np.ndarray, cfg: WaveConfig) -> tuple[np.ndarray, np.ndarray]:
    return v_int, cfg.c**2 * d2 @ y_int


def simulate_wave(method: str, cfg: WaveConfig) -> dict[str, object]:
    x, y_full, v_full = wave_initial_state(cfg)
    dx = cfg.length / (cfg.nx - 1)
    dt_base = cfg.cfl * dx / cfg.c
    n_steps = int(np.ceil(cfg.final_time / dt_base))
    dt = cfg.final_time / n_steps
    times = np.linspace(0.0, cfg.final_time, n_steps + 1)
    target_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    snapshots = {0.0: y_full.copy()}

    y = y_full[1:-1].copy()
    v = v_full[1:-1].copy()
    d2 = second_derivative_matrix(cfg.nx - 2, dx)
    identity = np.eye(cfg.nx - 2)
    if method == "euler_implicito":
        matrix = np.block(
            [
                [identity, -dt * identity],
                [-dt * cfg.c**2 * d2, identity],
            ]
        )
    center = int(np.argmin(np.abs(x - cfg.length / 2)))
    y_center = [float(y_full[center])]
    v_center = [float(v_full[center])]
    errors = [l2_wave_error(x, y_full, 0.0, cfg)]

    for step in range(1, n_steps + 1):
        if method == "euler_explicito":
            dy, dv = wave_rhs(y, v, d2, cfg)
            y = y + dt * dy
            v = v + dt * dv
        elif method == "rk4":
            k1y, k1v = wave_rhs(y, v, d2, cfg)
            k2y, k2v = wave_rhs(y + 0.5 * dt * k1y, v + 0.5 * dt * k1v, d2, cfg)
            k3y, k3v = wave_rhs(y + 0.5 * dt * k2y, v + 0.5 * dt * k2v, d2, cfg)
            k4y, k4v = wave_rhs(y + dt * k3y, v + dt * k3v, d2, cfg)
            y = y + dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
            v = v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        elif method == "euler_implicito":
            solution = np.linalg.solve(matrix, np.concatenate([y, v]))
            y = solution[: cfg.nx - 2]
            v = solution[cfg.nx - 2 :]
        else:
            raise ValueError(f"Metodo desconhecido: {method}")

        y_full = np.zeros(cfg.nx)
        v_full = np.zeros(cfg.nx)
        y_full[1:-1] = y
        v_full[1:-1] = v
        time = times[step]
        for target in target_times:
            if target not in snapshots and time >= target - 0.5 * dt:
                snapshots[target] = y_full.copy()
        y_center.append(float(y_full[center]))
        v_center.append(float(v_full[center]))
        errors.append(l2_wave_error(x, y_full, time, cfg))

    return {
        "x": x,
        "times": times,
        "snapshots": snapshots,
        "y_final": y_full,
        "y_center": np.array(y_center),
        "v_center": np.array(v_center),
        "errors": np.array(errors),
        "dt": dt,
    }


def exact_wave(x: np.ndarray, time: float, cfg: WaveConfig) -> np.ndarray:
    return cfg.amplitude * np.sin(np.pi * x / cfg.length) * np.cos(np.pi * cfg.c * time / cfg.length)


def l2_wave_error(x: np.ndarray, numerical: np.ndarray, time: float, cfg: WaveConfig) -> float:
    return float(np.sqrt(np.mean((numerical - exact_wave(x, time, cfg)) ** 2)))


def plot_wave_results(results: dict[str, dict[str, object]], cfg: WaveConfig) -> None:
    labels = {
        "euler_explicito": "Euler explicito",
        "rk4": "Runge-Kutta 4",
        "euler_implicito": "Euler implicito",
    }
    for method, result in results.items():
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for time, values in sorted(result["snapshots"].items()):
            ax.plot(result["x"], values, label=f"t={time:.1f} s")
        ax.set_title(f"Equacao da onda - {labels[method]}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(FIGURES / f"onda_deslocamento_{method}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for method, result in results.items():
        ax.plot(result["x"], result["y_final"], label=labels[method])
    ax.plot(result["x"], exact_wave(result["x"], cfg.final_time, cfg), "k--", label="Analitica")
    ax.set_title("Comparacao em t=2 s")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(FIGURES / "onda_comparacao_t2.png", dpi=180)
    plt.close(fig)

    for key, ylabel, filename in [("y_center", "y(L/2,t) (m)", "onda_evolucao_y_centro"), ("v_center", "v(L/2,t) (m/s)", "onda_evolucao_v_centro"), ("errors", "Erro L2 (m)", "onda_erro_l2")]:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for method, result in results.items():
            ax.plot(result["times"], result[key], label=labels[method])
        ax.set_title(ylabel)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(FIGURES / f"{filename}.png", dpi=180)
        plt.close(fig)


def save_summary(heat_c_results: dict[float, dict[str, object]], wave_results: dict[str, dict[str, object]]) -> None:
    with (DATA / "resumo.txt").open("w", encoding="utf-8") as file:
        file.write("Conducao termica 2D implicita\n")
        for c_value, result in heat_c_results.items():
            file.write(
                f"C={c_value:g}; dt={result['dt']:.8g} s; "
                f"rx={result['rx']:.6g}; ry={result['ry']:.6g}; "
                f"iteracoes medias GS={result['mean_gs_iterations']:.2f}; "
                f"iteracoes maximas GS={result['max_gs_iterations']}\n"
            )
        file.write("\nEquacao da onda 1D\n")
        for method, result in wave_results.items():
            file.write(f"{method}; dt={result['dt']:.8g} s; erro L2 final={result['errors'][-1]:.8e}\n")


def run_all() -> None:
    ensure_dirs()
    heat_cfg = HeatConfig()
    heat_times = [0.0, 1800.0, 3600.0, 7200.0, 10800.0, 14400.0]
    heat_c_results: dict[float, dict[str, object]] = {}
    for c_value in [0.25, 0.50, 1.0, 2.0]:
        result = simulate_heat(10, 10, c_value, heat_times, heat_cfg)
        heat_c_results[c_value] = result
        plot_heat_snapshots(result, c_value)

    heat_mesh_results: dict[int, dict[str, object]] = {}
    for n in [10, 20, 40]:
        heat_mesh_results[n] = simulate_heat(n, n, 1.0, [0.0, 14400.0], heat_cfg)
    plot_heat_mesh_profiles(heat_mesh_results, heat_cfg)
    plot_heat_temporal(heat_mesh_results)

    wave_cfg = WaveConfig()
    wave_results = {
        "euler_explicito": simulate_wave("euler_explicito", wave_cfg),
        "rk4": simulate_wave("rk4", wave_cfg),
        "euler_implicito": simulate_wave("euler_implicito", wave_cfg),
    }
    plot_wave_results(wave_results, wave_cfg)
    save_summary(heat_c_results, wave_results)
    print(f"Resultados gerados em: {RESULTS}")


if __name__ == "__main__":
    run_all()
