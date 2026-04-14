import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SOLUÇÃO NUMÉRICA DA CONDUÇÃO 2D TRANSIENTE EM UMA CHAPA
# Método: Euler explícito no tempo + diferenças finitas centrais
# Problema baseado no enunciado da avaliação.
# ============================================================

sigma = 5.670374419e-8  # Stefan-Boltzmann [W/m².K^4]

# ------------------------------
# Dados geométricos
# ------------------------------
Lx = 1.0           # [m]
Ly = 1.0           # [m]
e = 5.0e-3         # [m]

# ------------------------------
# Propriedades do alumínio puro
# ------------------------------
rho = 2700.0       # [kg/m³]
cp = 900.0         # [J/(kg.K)]
k = 237.0          # [W/(m.K)]
alpha = k / (rho * cp)

# ------------------------------
# Condições de contorno
# ------------------------------
T_left = 400.0
h_bottom = 100.0
Tinf_bottom = 350.0
h_top = 200.0
Tinf_top = 280.0
epsilon = 0.8
Tviz = 300.0

# ------------------------------
# Condição inicial e tempo final
# ------------------------------
T_init = 300.0
t_final = 4.0 * 3600.0
field_times = [0.0, 1800.0, 3600.0, 7200.0, 10800.0, 14400.0]
mesh_list = [10, 20, 40, 80]

# Pasta de saída
output_dir = "graficos_saida"
os.makedirs(output_dir, exist_ok=True)


def compute_stable_dt(dx, dy, alpha, safety=0.45):
    """Calcula um dt estável para o esquema explícito 2D."""
    dt_lim = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
    return safety * dt_lim


def radiative_h(Ts, Tviz, epsilon, sigma):
    """Coeficiente radiativo equivalente da linearização do termo T^4."""
    return epsilon * sigma * (Ts + Tviz) * (Ts**2 + Tviz**2)


def apply_boundary_conditions(T, dx, dy):
    """
    Aplica as condições de contorno na matriz T.
    T[j, i] com j em y e i em x.
    """
    Ny, Nx = T.shape

    # x = 0 -> temperatura prescrita
    T[:, 0] = T_left

    # x = Lx -> adiabático
    T[:, -1] = T[:, -2]

    # y = 0 -> convecção
    for i in range(1, Nx - 1):
        T[0, i] = ((k / dy) * T[1, i] + h_bottom * Tinf_bottom) / ((k / dy) + h_bottom)

    # y = Ly -> convecção + radiação
    for i in range(1, Nx - 1):
        Ts_old = T[-1, i]
        h_rad = radiative_h(Ts_old, Tviz, epsilon, sigma)
        h_eq = h_top + h_rad
        Tinf_eq = (h_top * Tinf_top + h_rad * Tviz) / h_eq
        T[-1, i] = ((k / dy) * T[-2, i] + h_eq * Tinf_eq) / ((k / dy) + h_eq)

    # tratamento simples dos cantos
    T[0, 0] = T_left
    T[-1, 0] = T_left
    T[0, -1] = T[0, -2]
    T[-1, -1] = T[-1, -2]

    return T


def solve_case(Nx, Ny, save_prefix="avaliacao1"):
    """Resolve o problema para uma dada malha Nx x Ny."""
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)

    dt = compute_stable_dt(dx, dy, alpha, safety=0.45)
    n_steps = int(np.ceil(t_final / dt))
    dt = t_final / n_steps

    T = np.full((Ny, Nx), T_init, dtype=float)
    T = apply_boundary_conditions(T, dx, dy)

    ix_mid = np.argmin(np.abs(x - Lx / 2.0))
    iy_mid = np.argmin(np.abs(y - Ly / 2.0))
    ix_q = np.argmin(np.abs(x - Lx / 4.0))
    iy_q = np.argmin(np.abs(y - Ly / 4.0))

    time_hist = [0.0]
    T_mid_hist = [T[iy_mid, ix_mid]]
    T_q_hist = [T[iy_q, ix_q]]
    snapshots = {0.0: T.copy()}

    for n in range(1, n_steps + 1):
        t = n * dt
        T_new = T.copy()

        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                d2Tdx2 = (T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]) / dx**2
                d2Tdy2 = (T[j + 1, i] - 2.0 * T[j, i] + T[j - 1, i]) / dy**2
                T_new[j, i] = T[j, i] + alpha * dt * (d2Tdx2 + d2Tdy2)

        T_new = apply_boundary_conditions(T_new, dx, dy)
        T = T_new

        time_hist.append(t)
        T_mid_hist.append(T[iy_mid, ix_mid])
        T_q_hist.append(T[iy_q, ix_q])

        for tf in field_times:
            if tf not in snapshots and abs(t - tf) <= 0.5 * dt:
                snapshots[tf] = T.copy()

    snapshots[t_final] = T.copy()

    X, Y = np.meshgrid(x, y)
    for tf in field_times:
        if tf in snapshots:
            plt.figure(figsize=(7, 5))
            cp_plot = plt.contourf(X, Y, snapshots[tf], levels=30)
            plt.colorbar(cp_plot, label="Temperatura [K]")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title(f"Campo de temperatura - t = {tf:.0f} s - malha {Nx}x{Ny}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"campo_T_t{int(tf)}_{save_prefix}_{Nx}x{Ny}.png"), dpi=300)
            plt.close()

    return {
        "x": x,
        "y": y,
        "T": T,
        "snapshots": snapshots,
        "time": np.array(time_hist),
        "T_mid_hist": np.array(T_mid_hist),
        "T_q_hist": np.array(T_q_hist),
    }


def plot_vertical_profile(results, x_target, title_suffix, filename):
    plt.figure(figsize=(7, 5))
    for N in mesh_list:
        x = results[N]["x"]
        y = results[N]["y"]
        T = results[N]["T"]
        ix = np.argmin(np.abs(x - x_target))
        plt.plot(T[:, ix], y, label=f"{N}x{N}")
    plt.xlabel("Temperatura [K]")
    plt.ylabel("y [m]")
    plt.title(title_suffix)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_horizontal_profile(results, y_target, title_suffix, filename):
    plt.figure(figsize=(7, 5))
    for N in mesh_list:
        x = results[N]["x"]
        y = results[N]["y"]
        T = results[N]["T"]
        iy = np.argmin(np.abs(y - y_target))
        plt.plot(x, T[iy, :], label=f"{N}x{N}")
    plt.xlabel("x [m]")
    plt.ylabel("Temperatura [K]")
    plt.title(title_suffix)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def main():
    results = {}
    for N in mesh_list:
        print(f"Resolvendo para malha {N}x{N}...")
        results[N] = solve_case(N, N)

    plot_vertical_profile(results, Lx / 2.0, "Perfil em t=4 h, x=Lx/2", "perfil_x_Lx2_t4h.png")
    plot_vertical_profile(results, Lx / 4.0, "Perfil em t=4 h, x=Lx/4", "perfil_x_Lx4_t4h.png")
    plot_horizontal_profile(results, Ly / 2.0, "Perfil em t=4 h, y=Ly/2", "perfil_y_Ly2_t4h.png")
    plot_horizontal_profile(results, 3.0 * Ly / 4.0, "Perfil em t=4 h, y=3Ly/4", "perfil_y_3Ly4_t4h.png")

    plt.figure(figsize=(7, 5))
    for N in mesh_list:
        plt.plot(results[N]["time"], results[N]["T_mid_hist"], label=f"{N}x{N}")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Temperatura [K]")
    plt.title("Evolução temporal em x=Lx/2, y=Ly/2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evolucao_temporal_centro.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    for N in mesh_list:
        plt.plot(results[N]["time"], results[N]["T_q_hist"], label=f"{N}x{N}")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Temperatura [K]")
    plt.title("Evolução temporal em x=Lx/4, y=Ly/4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evolucao_temporal_quarto.png"), dpi=300)
    plt.close()

    print("Simulação concluída.")
    print(f"Gráficos salvos em: {output_dir}")


if __name__ == "__main__":
    main()
