import numpy as np
import matplotlib.pyplot as plt

"""
Questao 1
Conducao de calor 2D, transiente, k constante, usando volume fantasma
para impor as condicoes de contorno.

Equacao governante:
    rho*cp*dT/dt = k*(d2T/dx2 + d2T/dy2)

Condicoes de contorno:
    x = 0   : temperatura imposta (Dirichlet)
    x = Lx  : adiabatico
    y = 0   : conveccao
    y = Ly  : conveccao + radiacao
    t = 0   : T = 300 K

Observacao numerica:
- O campo T eh armazenado apenas nos pontos fisicos: shape (Nx, Ny)
- As celulas fantasma sao montadas a cada passo em uma malha expandida
  Tg com shape (Nx+2, Ny+2)
- O avanco temporal eh feito por Euler explicito
"""

# =============================
# Parametros fisicos
# =============================
rho = 7800.0               # kg/m^3
cp = 500.0                 # J/(kg.K)
k = 45.0                   # W/(m.K)
alpha = k / (rho * cp)     # difusividade termica

h_bottom = 20.0            # W/(m^2.K)
Tinf_bottom = 290.0        # K

h_top = 15.0               # W/(m^2.K)
Tinf_top = 295.0           # K

epsilon = 0.8
sigma = 5.670374419e-8     # W/(m^2.K^4)
Tsur = 285.0               # K

T_left = 500.0             # K (temperatura imposta em x=0)
T_init = 300.0             # K

# =============================
# Parametros numericos
# =============================
Lx = 1.0
Ly = 1.0
Nx = 41
Ny = 41

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Criterio simples para Euler explicito em 2D
# dt <= 1 / (2*alpha*(1/dx^2 + 1/dy^2))
dt_stable = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
dt = 0.30 * dt_stable

t_final = 2000.0
n_steps = int(np.ceil(t_final / dt))

save_every = max(1, n_steps // 6)


# =============================
# Funcoes auxiliares
# =============================
def radiation_flux(Ts, T_amb, emissivity=epsilon, stefan=sigma):
    """Fluxo radiativo positivo saindo da superficie."""
    return emissivity * stefan * (Ts**4 - T_amb**4)


def build_ghost_field(T):
    """
    Monta campo expandido Tg com uma camada de volumes fantasmas.

    Indexacao:
        T  : fisico     -> T[i, j], i=0..Nx-1, j=0..Ny-1
        Tg : expandido  -> Tg[1:-1, 1:-1] = T

    Formulas de volume fantasma usadas:

    1) x = 0 (Dirichlet, T = T_left)
       T_w = 2*T_left - T_P

    2) x = Lx (adiabatico, dT/dx = 0)
       T_e = T_P

    3) y = 0 (conveccao)
       -k (T_P - T_s) / (2*dy) = h (T_P - Tinf)
       => T_s = T_P + 2*dy*(h/k)*(T_P - Tinf)

    4) y = Ly (conveccao + radiacao)
       -k (T_n - T_P) / (2*dy) = h (T_P - Tinf) + eps*sigma*(T_P^4 - Tsur^4)
       => T_n = T_P - (2*dy/k)*[ h(T_P-Tinf) + eps*sigma(T_P^4-Tsur^4) ]
    """
    Tg = np.zeros((Nx + 2, Ny + 2), dtype=float)
    Tg[1:-1, 1:-1] = T

    # x = 0 -> Dirichlet
    Tg[0, 1:-1] = 2.0 * T_left - T[0, :]

    # x = Lx -> adiabatico
    Tg[-1, 1:-1] = T[-1, :]

    # y = 0 -> conveccao
    P_bottom = T[:, 0]
    Tg[1:-1, 0] = P_bottom + 2.0 * dy * (h_bottom / k) * (P_bottom - Tinf_bottom)

    # y = Ly -> conveccao + radiacao
    P_top = T[:, -1]
    q_top = h_top * (P_top - Tinf_top) + radiation_flux(P_top, Tsur)
    Tg[1:-1, -1] = P_top - (2.0 * dy / k) * q_top

    # cantos: media simples para manter consistencia numerica
    Tg[0, 0] = 0.5 * (Tg[1, 0] + Tg[0, 1])
    Tg[0, -1] = 0.5 * (Tg[1, -1] + Tg[0, -2])
    Tg[-1, 0] = 0.5 * (Tg[-2, 0] + Tg[-1, 1])
    Tg[-1, -1] = 0.5 * (Tg[-2, -1] + Tg[-1, -2])

    return Tg


def explicit_step(T):
    """Um passo de Euler explicito usando o campo com volumes fantasmas."""
    Tg = build_ghost_field(T)
    Tnew = T.copy()

    for i in range(Nx):
        for j in range(Ny):
            ii = i + 1
            jj = j + 1
            d2Tdx2 = (Tg[ii + 1, jj] - 2.0 * Tg[ii, jj] + Tg[ii - 1, jj]) / dx**2
            d2Tdy2 = (Tg[ii, jj + 1] - 2.0 * Tg[ii, jj] + Tg[ii, jj - 1]) / dy**2
            Tnew[i, j] = T[i, j] + alpha * dt * (d2Tdx2 + d2Tdy2)

    return Tnew


def simulate():
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    T = np.full((Nx, Ny), T_init, dtype=float)

    snapshots = [(0.0, T.copy())]
    time = 0.0

    for n in range(1, n_steps + 1):
        T = explicit_step(T)
        time += dt

        if n % save_every == 0 or n == n_steps:
            snapshots.append((time, T.copy()))

    return x, y, X, Y, T, snapshots


# =============================
# Execucao principal
# =============================
if __name__ == "__main__":
    print("Rodando Questao 1 com volume fantasma...")
    print(f"Nx = {Nx}, Ny = {Ny}")
    print(f"dx = {dx:.6e}, dy = {dy:.6e}")
    print(f"alpha = {alpha:.6e}")
    print(f"dt_estavel ~= {dt_stable:.6e}")
    print(f"dt usado    = {dt:.6e}")
    print(f"passos      = {n_steps}")

    x, y, X, Y, T, snapshots = simulate()

    print(f"Temperatura minima final: {T.min():.4f} K")
    print(f"Temperatura maxima final: {T.max():.4f} K")
    print(f"Temperatura media  final: {T.mean():.4f} K")

    # Mapa final
    plt.figure(figsize=(7, 5))
    cf = plt.contourf(X, Y, T, levels=25)
    plt.colorbar(cf, label="Temperatura [K]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Questao 1 - campo de temperatura final")
    plt.tight_layout()
    plt.show()

    # Perfil na linha central em y
    jmid = Ny // 2
    plt.figure(figsize=(7, 4))
    plt.plot(x, T[:, jmid], marker="o", ms=3)
    plt.xlabel("x [m]")
    plt.ylabel("T [K]")
    plt.title("Questao 1 - perfil em y = Ly/2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evolucao de snapshots
    plt.figure(figsize=(8, 5))
    for time, Ts in snapshots:
        plt.plot(x, Ts[:, jmid], label=f"t={time:.1f} s")
    plt.xlabel("x [m]")
    plt.ylabel("T [K]")
    plt.title("Questao 1 - evolucao temporal em y = Ly/2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
