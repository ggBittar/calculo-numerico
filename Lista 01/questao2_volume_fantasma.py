import numpy as np
import matplotlib.pyplot as plt

"""
Questao 2
Conducao de calor 2D, transiente, com condutividade dependente da
temperatura: k(T) = a*T + b, usando volume fantasma para impor
as condicoes de contorno.

Equacao governante:
    rho*cp*dT/dt = div( k(T) grad(T) )

Discretizacao espacial adotada (forma conservativa):
    div( k grad(T) ) ~=
      [ qx(i+1/2,j) - qx(i-1/2,j) ]/dx + [ qy(i,j+1/2) - qy(i,j-1/2) ]/dy

com fluxos difusivos por face:
    qx = k_face * dT/dx
    qy = k_face * dT/dy

As condicoes de contorno sao impostas por celulas fantasmas.
No topo, a radiacao torna o problema nao linear; o valor da celula
fantasma superior eh atualizado diretamente a partir do fluxo total
convectivo+radiativo calculado no contorno.
"""

# =============================
# Parametros fisicos
# =============================
rho = 7800.0               # kg/m^3
cp = 500.0                 # J/(kg.K)

a = 0.03                   # W/(m.K^2)
b = 5.0                    # W/(m.K)

h_bottom = 20.0            # W/(m^2.K)
Tinf_bottom = 290.0        # K

h_top = 15.0               # W/(m^2.K)
Tinf_top = 295.0           # K

epsilon = 0.8
sigma = 5.670374419e-8
Tsur = 285.0               # K

T_left = 500.0             # K
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

# Para k variavel, usa-se uma estimativa conservadora com k max esperado
k_ref = a * T_left + b
alpha_ref = k_ref / (rho * cp)
dt_stable = 1.0 / (2.0 * alpha_ref * (1.0 / dx**2 + 1.0 / dy**2))
dt = 0.20 * dt_stable

t_final = 2000.0
n_steps = int(np.ceil(t_final / dt))

save_every = max(1, n_steps // 6)


# =============================
# Funcoes auxiliares
# =============================
def k_func(T):
    return a * T + b


def radiation_flux(Ts, T_amb, emissivity=epsilon, stefan=sigma):
    return emissivity * stefan * (Ts**4 - T_amb**4)


def build_ghost_field(T):
    """
    Monta malha expandida com volumes fantasmas.

    1) x = 0 (Dirichlet, T = T_left)
       T_w = 2*T_left - T_P

    2) x = Lx (adiabatico)
       T_e = T_P

    3) y = 0 (conveccao)
       -k_P (T_P - T_s)/(2*dy) = h (T_P - Tinf)
       => T_s = T_P + 2*dy*(h/k_P)*(T_P - Tinf)

    4) y = Ly (conveccao + radiacao)
       -k_P (T_n - T_P)/(2*dy) = h (T_P - Tinf) + eps*sigma(T_P^4 - Tsur^4)
       => T_n = T_P - (2*dy/k_P)*[ h(T_P-Tinf) + eps*sigma(T_P^4-Tsur^4) ]
    """
    Tg = np.zeros((Nx + 2, Ny + 2), dtype=float)
    Tg[1:-1, 1:-1] = T

    # x = 0 -> Dirichlet
    Tg[0, 1:-1] = 2.0 * T_left - T[0, :]

    # x = Lx -> adiabatico
    Tg[-1, 1:-1] = T[-1, :]

    # y = 0 -> conveccao com k(T)
    P_bottom = T[:, 0]
    k_bottom = k_func(P_bottom)
    Tg[1:-1, 0] = P_bottom + 2.0 * dy * (h_bottom / k_bottom) * (P_bottom - Tinf_bottom)

    # y = Ly -> conveccao + radiacao com k(T)
    P_top = T[:, -1]
    k_top = k_func(P_top)
    q_top = h_top * (P_top - Tinf_top) + radiation_flux(P_top, Tsur)
    Tg[1:-1, -1] = P_top - (2.0 * dy / k_top) * q_top

    # cantos
    Tg[0, 0] = 0.5 * (Tg[1, 0] + Tg[0, 1])
    Tg[0, -1] = 0.5 * (Tg[1, -1] + Tg[0, -2])
    Tg[-1, 0] = 0.5 * (Tg[-2, 0] + Tg[-1, 1])
    Tg[-1, -1] = 0.5 * (Tg[-2, -1] + Tg[-1, -2])

    return Tg


def explicit_step(T):
    """
    Um passo explicito em forma conservativa para k(T).
    Fluxos por face usam condutividade media aritmetica entre centros.
    """
    Tg = build_ghost_field(T)
    Tnew = T.copy()

    for i in range(Nx):
        for j in range(Ny):
            ii = i + 1
            jj = j + 1

            Tc = Tg[ii, jj]
            Te = Tg[ii + 1, jj]
            Tw = Tg[ii - 1, jj]
            Tn = Tg[ii, jj + 1]
            Ts = Tg[ii, jj - 1]

            ke = 0.5 * (k_func(Tc) + k_func(Te))
            kw = 0.5 * (k_func(Tc) + k_func(Tw))
            kn = 0.5 * (k_func(Tc) + k_func(Tn))
            ks = 0.5 * (k_func(Tc) + k_func(Ts))

            qx_e = ke * (Te - Tc) / dx
            qx_w = kw * (Tc - Tw) / dx
            qy_n = kn * (Tn - Tc) / dy
            qy_s = ks * (Tc - Ts) / dy

            div_k_grad_T = (qx_e - qx_w) / dx + (qy_n - qy_s) / dy

            Tnew[i, j] = T[i, j] + (dt / (rho * cp)) * div_k_grad_T

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
    print("Rodando Questao 2 com volume fantasma e k(T)=aT+b...")
    print(f"Nx = {Nx}, Ny = {Ny}")
    print(f"dx = {dx:.6e}, dy = {dy:.6e}")
    print(f"k_ref = {k_ref:.6e}")
    print(f"alpha_ref = {alpha_ref:.6e}")
    print(f"dt_estavel ~= {dt_stable:.6e}")
    print(f"dt usado    = {dt:.6e}")
    print(f"passos      = {n_steps}")

    x, y, X, Y, T, snapshots = simulate()

    print(f"Temperatura minima final: {T.min():.4f} K")
    print(f"Temperatura maxima final: {T.max():.4f} K")
    print(f"Temperatura media  final: {T.mean():.4f} K")
    print(f"k minimo final: {k_func(T).min():.4f} W/(m.K)")
    print(f"k maximo final: {k_func(T).max():.4f} W/(m.K)")

    # Mapa final
    plt.figure(figsize=(7, 5))
    cf = plt.contourf(X, Y, T, levels=25)
    plt.colorbar(cf, label="Temperatura [K]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Questao 2 - campo de temperatura final")
    plt.tight_layout()
    plt.show()

    # Perfil na linha central em y
    jmid = Ny // 2
    plt.figure(figsize=(7, 4))
    plt.plot(x, T[:, jmid], marker="o", ms=3)
    plt.xlabel("x [m]")
    plt.ylabel("T [K]")
    plt.title("Questao 2 - perfil em y = Ly/2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evolucao de snapshots
    plt.figure(figsize=(8, 5))
    for time, Ts in snapshots:
        plt.plot(x, Ts[:, jmid], label=f"t={time:.1f} s")
    plt.xlabel("x [m]")
    plt.ylabel("T [K]")
    plt.title("Questao 2 - evolucao temporal em y = Ly/2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
