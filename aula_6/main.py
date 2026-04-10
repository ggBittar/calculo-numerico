import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parâmetros do problema
# -----------------------------
L = 1.0                 # comprimento da barra
alpha = 0.01            # difusividade térmica
Nx = 41                 # número total de nós (incluindo contornos)
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)

t0 = 0.0
tf = 5.0
dt = 0.02
n_steps = int((tf - t0) / dt)

# -----------------------------
# Condição inicial
# T(x,0) = sin(pi x / L)
# Contornos: T(0,t)=0 e T(L,t)=0
# -----------------------------
T0_full = np.sin(np.pi * x / L)
T0_full[0] = 0.0
T0_full[-1] = 0.0

# Vetor apenas com pontos internos
T0 = T0_full[1:-1].copy()

# -----------------------------
# Função que monta dT/dt
# usando diferenças centrais
# -----------------------------
def heat_rhs(t, T_internal):
    dTdt = np.zeros_like(T_internal)

    # montar vetor completo com contornos
    T_full = np.zeros(len(T_internal) + 2)
    T_full[1:-1] = T_internal
    T_full[0] = 0.0      # contorno esquerdo
    T_full[-1] = 0.0     # contorno direito

    # discretização espacial
    for i in range(1, len(T_full) - 1):
        dTdt[i - 1] = alpha * (T_full[i + 1] - 2*T_full[i] + T_full[i - 1]) / dx**2

    return dTdt

# -----------------------------
# RK2 de Heun
# -----------------------------
def rk2_heun_system(f, t0, y0, h, n):
    t_vals = [t0]
    y_vals = [y0.copy()]

    t = t0
    y = y0.copy()

    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h, y + h*k1)

        y = y + (h/2)*(k1 + k2)
        t = t + h

        t_vals.append(t)
        y_vals.append(y.copy())

    return np.array(t_vals), np.array(y_vals)

# -----------------------------
# RK4
# -----------------------------
def rk4_system(f, t0, y0, h, n):
    t_vals = [t0]
    y_vals = [y0.copy()]

    t = t0
    y = y0.copy()

    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h/2, y + (h/2)*k1)
        k3 = f(t + h/2, y + (h/2)*k2)
        k4 = f(t + h, y + h*k3)

        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h

        t_vals.append(t)
        y_vals.append(y.copy())

    return np.array(t_vals), np.array(y_vals)

# -----------------------------
# Solução exata
# -----------------------------
def exact_solution(x, t):
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L)**2 * t)

# -----------------------------
# Resolver
# -----------------------------
t_rk2, Y_rk2 = rk2_heun_system(heat_rhs, t0, T0, dt, n_steps)
t_rk4, Y_rk4 = rk4_system(heat_rhs, t0, T0, dt, n_steps)

# reconstruir solução completa com contornos
def reconstruct_full(Y_internal):
    Y_full = np.zeros((Y_internal.shape[0], Y_internal.shape[1] + 2))
    Y_full[:, 1:-1] = Y_internal
    return Y_full

T_rk2_full = reconstruct_full(Y_rk2)
T_rk4_full = reconstruct_full(Y_rk4)

# solução exata no tempo final
T_exact_tf = exact_solution(x, t_rk4[-1])

# erros no tempo final
err_rk2 = np.max(np.abs(T_rk2_full[-1] - T_exact_tf))
err_rk4 = np.max(np.abs(T_rk4_full[-1] - T_exact_tf))

print(f"Erro máximo RK2 no tempo final: {err_rk2:.6e}")
print(f"Erro máximo RK4 no tempo final: {err_rk4:.6e}")

# -----------------------------
# Gráfico no tempo final
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(x, T_rk2_full[-1], 's-', label='RK2 Heun')
plt.plot(x, T_rk4_full[-1], 'o-', label='RK4')
plt.plot(x, T_exact_tf, '--', label='Exata')
plt.xlabel('x')
plt.ylabel('T(x,t_f)')
plt.title('Equação do calor 1D no tempo final')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Evolução temporal no ponto central
# -----------------------------
i_center = Nx // 2
T_center_rk2 = T_rk2_full[:, i_center]
T_center_rk4 = T_rk4_full[:, i_center]
T_center_exact = exact_solution(x[i_center], t_rk4)

plt.figure(figsize=(10, 6))
plt.plot(t_rk2, T_center_rk2, 's-', label='RK2 Heun')
plt.plot(t_rk4, T_center_rk4, 'o-', label='RK4')
plt.plot(t_rk4, T_center_exact, '--', label='Exata')
plt.xlabel('t')
plt.ylabel('T(L/2,t)')
plt.title('Temperatura no centro da barra')
plt.legend()
plt.grid(True)
plt.show()