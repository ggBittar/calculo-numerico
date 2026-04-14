import os
import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Tenta usar GPU via CuPy/CUDA. Se não estiver disponível,
# faz fallback automático para NumPy (CPU, processamento em série).
# ============================================================
try:
    import cupy as cp
    _gpu_ok = cp.cuda.is_available()
except Exception:
    cp = None
    _gpu_ok = False

# ============================================================
# Constantes físicas e dados do problema
# ============================================================
SIGMA = 5.670374419e-8  # constante de Stefan-Boltzmann [W/(m^2.K^4)]

Lx = 1.0               # [m]
Ly = 1.0               # [m]
espessura = 5.0e-3     # [m]

# Propriedades típicas do alumínio puro (ajustáveis pelo usuário)
k = 237.0              # [W/(m.K)]
rho = 2700.0           # [kg/m^3]
cp_heat = 897.0        # [J/(kg.K)]
alpha = k / (rho * cp_heat)

T_left = 400.0         # [K] temperatura imposta em x = 0
T0 = 300.0             # [K] condição inicial

# Borda y = 0  (convecção)
h_bottom = 100.0       # [W/(m^2.K)]
Tinf_bottom = 350.0    # [K]

# Borda y = Ly (convecção + radiação)
h_top = 200.0          # [W/(m^2.K)]
Tinf_top = 280.0       # [K]
Tsur = 300.0           # [K] temperatura da vizinhança para radiação
epsilon = 0.8

TEMPO_FINAL = 4.0 * 3600.0  # 4 h [s]
TEMPOS_CAMPOS = [0.0, 1800.0, 3600.0, 7200.0, 10800.0, 14400.0]
MALHAS = [10, 20, 40, 80]

PASTA_FIG = "./figuras_avaliacao"
os.makedirs(PASTA_FIG, exist_ok=True)


def get_xp(use_gpu_preference=True):
    """Retorna o módulo numérico (cupy ou numpy) e uma string descrevendo o backend."""
    if use_gpu_preference and _gpu_ok:
        return cp, "gpu-cuda"
    return np, "cpu-serial"


def to_numpy(arr):
    """Converte array cupy/numpy para numpy para plotar/salvar."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def estabilidade_dt(nx, ny):
    """Limite estável para Euler explícito em 2D, com fator de segurança."""
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt_lim = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
    return 0.85 * dt_lim


def escolher_dt(nx, ny):
    """
    Escolhe dt estável e que divida 1800 s exatamente.
    Isso ajuda a capturar os instantes pedidos sem erro de arredondamento.
    """
    dt_estavel = estabilidade_dt(nx, ny)
    n_por_1800 = max(1, int(math.ceil(1800.0 / dt_estavel)))
    return 1800.0 / n_por_1800


def simulate(nx, ny, use_gpu_preference=True):
    """
    Resolve o problema 2D transiente usando:
      - diferenças finitas de 2ª ordem no espaço;
      - Euler explícito no tempo.

    Condições de contorno:
      x = 0      : temperatura imposta
      x = Lx     : adiabático
      y = 0      : convecção
      y = Ly     : convecção + radiação

    O código usa ghost nodes eliminados analiticamente para manter a discretização
    espacial de 2ª ordem nas bordas.
    """
    xp, backend = get_xp(use_gpu_preference)

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = escolher_dt(nx, ny)
    nsteps = int(round(TEMPO_FINAL / dt))

    x = xp.linspace(0.0, Lx, nx)
    y = xp.linspace(0.0, Ly, ny)

    # Campo de temperatura inicial
    T = xp.full((ny, nx), T0, dtype=xp.float64)
    T[:, 0] = T_left

    # Índices mais próximos dos pontos de interesse
    ix_half = int(np.argmin(np.abs(np.linspace(0.0, Lx, nx) - Lx / 2.0)))
    ix_quarter = int(np.argmin(np.abs(np.linspace(0.0, Lx, nx) - Lx / 4.0)))
    iy_half = int(np.argmin(np.abs(np.linspace(0.0, Ly, ny) - Ly / 2.0)))
    iy_3quarter = int(np.argmin(np.abs(np.linspace(0.0, Ly, ny) - 3.0 * Ly / 4.0)))
    iy_quarter = int(np.argmin(np.abs(np.linspace(0.0, Ly, ny) - Ly / 4.0)))

    # Armazenamento dos resultados pedidos
    snapshots = {0.0: to_numpy(T.copy())}
    time_hist = [0.0]
    center_hist = [float(to_numpy(T[iy_half, ix_half]))]
    quarter_hist = [float(to_numpy(T[iy_quarter, ix_quarter]))]

    # Coeficientes auxiliares do esquema explícito
    coef_x = alpha * dt / dx**2
    coef_y = alpha * dt / dy**2

    # Loop temporal principal
    for step in range(1, nsteps + 1):
        Told = T.copy()

        # ==========================
        # 1) Pontos internos
        # ==========================
        T[1:-1, 1:-1] = (
            Told[1:-1, 1:-1]
            + coef_x * (Told[1:-1, 2:] - 2.0 * Told[1:-1, 1:-1] + Told[1:-1, :-2])
            + coef_y * (Told[2:, 1:-1] - 2.0 * Told[1:-1, 1:-1] + Told[:-2, 1:-1])
        )

        # ==========================
        # 2) Borda esquerda (Dirichlet): x = 0
        # ==========================
        T[:, 0] = T_left

        # ==========================
        # 3) Borda direita (adiabática): dT/dx = 0
        # Ghost node: T_g = T_{N-2}
        # d2T/dx2 na borda = 2*(T_{N-2} - T_{N-1})/dx^2
        # ==========================
        j = slice(1, ny - 1)
        i = nx - 1
        T[j, i] = (
            Told[j, i]
            + alpha * dt * (
                2.0 * (Told[j, i - 1] - Told[j, i]) / dx**2
                + (Told[2:, i] - 2.0 * Told[1:-1, i] + Told[:-2, i]) / dy**2
            )
        )

        # ==========================
        # 4) Borda inferior (convecção): y = 0
        # k dT/dy = h (T - Tinf)
        # Ghost node: T_-1 = T_1 - 2 dy (h/k) (T_0 - Tinf)
        # ==========================
        j = 0
        i = slice(1, nx - 1)
        T_bottom = Told[j, i]
        d2y_bottom = 2.0 * (Told[j + 1, i] - T_bottom) / dy**2 - 2.0 * h_bottom * (T_bottom - Tinf_bottom) / (k * dy)
        d2x_bottom = (Told[j, 2:] - 2.0 * T_bottom + Told[j, :-2]) / dx**2
        T[j, i] = T_bottom + alpha * dt * (d2x_bottom + d2y_bottom)

        # ==========================
        # 5) Borda superior (convecção + radiação): y = Ly
        # -k dT/dy = h (T - Tinf) + eps sigma (T^4 - Tsur^4)
        # Ghost node obtido da condição de contorno.
        # ==========================
        j = ny - 1
        i = slice(1, nx - 1)
        T_top_now = Told[j, i]
        q_conv_rad = h_top * (T_top_now - Tinf_top) + epsilon * SIGMA * (T_top_now**4 - Tsur**4)
        d2y_top = 2.0 * (Told[j - 1, i] - T_top_now) / dy**2 - 2.0 * q_conv_rad / (k * dy)
        d2x_top = (Told[j, 2:] - 2.0 * T_top_now + Told[j, :-2]) / dx**2
        T[j, i] = T_top_now + alpha * dt * (d2x_top + d2y_top)

        # ==========================
        # 6) Cantos
        # ==========================
        # Canto inferior esquerdo: T imposta
        T[0, 0] = T_left

        # Canto superior esquerdo: T imposta
        T[-1, 0] = T_left

        # Canto inferior direito: adiabático em x e convecção em y
        T_br = Told[0, -1]
        d2x_br = 2.0 * (Told[0, -2] - T_br) / dx**2
        d2y_br = 2.0 * (Told[1, -1] - T_br) / dy**2 - 2.0 * h_bottom * (T_br - Tinf_bottom) / (k * dy)
        T[0, -1] = T_br + alpha * dt * (d2x_br + d2y_br)

        # Canto superior direito: adiabático em x e convecção+radiação em y
        T_tr = Told[-1, -1]
        q_top_corner = h_top * (T_tr - Tinf_top) + epsilon * SIGMA * (T_tr**4 - Tsur**4)
        d2x_tr = 2.0 * (Told[-1, -2] - T_tr) / dx**2
        d2y_tr = 2.0 * (Told[-2, -1] - T_tr) / dy**2 - 2.0 * q_top_corner / (k * dy)
        T[-1, -1] = T_tr + alpha * dt * (d2x_tr + d2y_tr)

        # Reimpõe a condição de Dirichlet por segurança numérica
        T[:, 0] = T_left

        # Registro temporal
        t = step * dt
        time_hist.append(t)
        center_hist.append(float(to_numpy(T[iy_half, ix_half])))
        quarter_hist.append(float(to_numpy(T[iy_quarter, ix_quarter])))

        # Salva snapshots exatamente nos tempos de interesse
        t_round = round(t, 10)
        for target in TEMPOS_CAMPOS[1:]:
            if abs(t_round - target) <= 1e-9:
                snapshots[target] = to_numpy(T.copy())

    return {
        "backend": backend,
        "nx": nx,
        "ny": ny,
        "dt": dt,
        "nsteps": nsteps,
        "x": to_numpy(x),
        "y": to_numpy(y),
        "T_final": to_numpy(T),
        "snapshots": snapshots,
        "time": np.array(time_hist),
        "center_hist": np.array(center_hist),
        "quarter_hist": np.array(quarter_hist),
        "profile_x_half": to_numpy(T[:, ix_half]),
        "profile_x_quarter": to_numpy(T[:, ix_quarter]),
        "profile_y_half": to_numpy(T[iy_half, :]),
        "profile_y_3quarter": to_numpy(T[iy_3quarter, :]),
        "ix_half": ix_half,
        "ix_quarter": ix_quarter,
        "iy_half": iy_half,
        "iy_3quarter": iy_3quarter,
        "iy_quarter": iy_quarter,
    }


def salvar_campos(result_ref):
    """Salva os 6 campos pedidos para a malha Nx=Ny=10."""
    x = result_ref["x"]
    y = result_ref["y"]
    X, Y = np.meshgrid(x, y)
    for t in TEMPOS_CAMPOS:
        T = result_ref["snapshots"][t]
        plt.figure(figsize=(7.5, 5.5))
        cs = plt.contourf(X, Y, T, levels=25)
        plt.colorbar(cs, label="Temperatura [K]")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"Campo de temperatura para t = {t:.0f} s (Nx=Ny=10)")
        plt.tight_layout()
        out = os.path.join(PASTA_FIG, f"campo_temperatura_t_{int(t):05d}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()


def salvar_perfis(results):
    """Salva os 4 perfis espaciais em t = 4 h, comparando as quatro malhas."""
    # 1) x = Lx/2, variando y
    plt.figure(figsize=(7.5, 5.5))
    for n, res in results.items():
        plt.plot(res["profile_x_half"], res["y"], label=f"N={n}")
    plt.xlabel("Temperatura [K]")
    plt.ylabel("y [m]")
    plt.title("Perfil em t = 4 h para x = Lx/2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIG, "perfil_x_meio.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2) x = Lx/4, variando y
    plt.figure(figsize=(7.5, 5.5))
    for n, res in results.items():
        plt.plot(res["profile_x_quarter"], res["y"], label=f"N={n}")
    plt.xlabel("Temperatura [K]")
    plt.ylabel("y [m]")
    plt.title("Perfil em t = 4 h para x = Lx/4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIG, "perfil_x_quarto.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3) y = Ly/2, variando x
    plt.figure(figsize=(7.5, 5.5))
    for n, res in results.items():
        plt.plot(res["x"], res["profile_y_half"], label=f"N={n}")
    plt.xlabel("x [m]")
    plt.ylabel("Temperatura [K]")
    plt.title("Perfil em t = 4 h para y = Ly/2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIG, "perfil_y_meio.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4) y = 3Ly/4, variando x
    plt.figure(figsize=(7.5, 5.5))
    for n, res in results.items():
        plt.plot(res["x"], res["profile_y_3quarter"], label=f"N={n}")
    plt.xlabel("x [m]")
    plt.ylabel("Temperatura [K]")
    plt.title("Perfil em t = 4 h para y = 3Ly/4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIG, "perfil_y_tres_quartos.png"), dpi=200, bbox_inches="tight")
    plt.close()


def salvar_historicos(results):
    """Salva os 2 históricos temporais pedidos para as quatro malhas."""
    # 1) x=Lx/2, y=Ly/2
    plt.figure(figsize=(7.5, 5.5))
    for n, res in results.items():
        plt.plot(res["time"] / 3600.0, res["center_hist"], label=f"N={n}")
    plt.xlabel("Tempo [h]")
    plt.ylabel("Temperatura [K]")
    plt.title("Evolução temporal em x=Lx/2, y=Ly/2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIG, "historico_centro.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2) x=Lx/4, y=Ly/4
    plt.figure(figsize=(7.5, 5.5))
    for n, res in results.items():
        plt.plot(res["time"] / 3600.0, res["quarter_hist"], label=f"N={n}")
    plt.xlabel("Tempo [h]")
    plt.ylabel("Temperatura [K]")
    plt.title("Evolução temporal em x=Lx/4, y=Ly/4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIG, "historico_quarto.png"), dpi=200, bbox_inches="tight")
    plt.close()


def salvar_resumo_txt(results):
    lines = []
    for n, res in results.items():
        lines.append(f"Malha N={n}x{n}")
        lines.append(f"  backend = {res['backend']}")
        lines.append(f"  dt = {res['dt']:.12f} s")
        lines.append(f"  nsteps = {res['nsteps']}")
        lines.append(f"  T_centro_final = {res['center_hist'][-1]:.6f} K")
        lines.append(f"  T_quarto_final = {res['quarter_hist'][-1]:.6f} K")
        lines.append("")
    with open(os.path.join(PASTA_FIG, "resumo_simulacao.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(use_gpu_preference=True):
    results = {}
    for n in MALHAS:
        print(f"Simulando malha {n}x{n}...")
        results[n] = simulate(n, n, use_gpu_preference=use_gpu_preference)
        print(f"  backend: {results[n]['backend']}")
        print(f"  dt: {results[n]['dt']:.6e} s | passos: {results[n]['nsteps']}")

    # Os campos são pedidos explicitamente para Nx=10 e Ny=10
    salvar_campos(results[10])
    salvar_perfis(results)
    salvar_historicos(results)
    salvar_resumo_txt(results)

    print(f"\nArquivos salvos em: {PASTA_FIG}")


if __name__ == "__main__":
    main(use_gpu_preference=True)
