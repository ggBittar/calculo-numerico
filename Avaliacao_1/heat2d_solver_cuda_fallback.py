import os
import math
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# SOLUÇÃO NUMÉRICA DO PROBLEMA DE CONDUÇÃO 2D TRANSIENTE
# Formulação implícita no tempo + volumes finitos em malha cartesiana
#
# Recurso opcional de aceleração:
# - Se houver CUDA disponível e o pacote numba estiver instalado,
#   o código usa um solver iterativo de Jacobi na GPU.
# - Caso contrário, usa fallback em série na CPU com solver direto esparso.
#
# O código salva automaticamente figuras PNG na pasta "figuras".
# ================================================================

# -----------------------------
# DADOS DO PROBLEMA
# -----------------------------
Lx = 0.02          # comprimento [m]
Ly = 0.01          # altura/espessura [m]
k = 14.9           # condutividade térmica [W/(m.K)]
alpha = 3.95e-6    # difusividade térmica [m²/s]
h = 20.0           # coeficiente de convecção [W/(m².K)]
T_inf = 30.0       # temperatura ambiente [°C]
T0 = 30.0          # temperatura inicial [°C]
q_flux = 5.0e4     # fluxo de calor nas regiões aquecidas [W/m²]
t_final = 60.0     # tempo final [s]
dt = 0.2           # passo de tempo [s]

# Densidade calorífica volumétrica obtida a partir de alpha = k/(rho*cp)
rho_cp = k / alpha
z = 1.0            # profundidade unitária [m]

# -----------------------------
# MALHA DE VOLUMES FINITOS
# -----------------------------
Nx = 41
Ny = 21

dx = Lx / Nx
dy = Ly / Ny

# Coordenadas dos centros dos volumes
x = np.linspace(dx/2, Lx - dx/2, Nx)
y = np.linspace(dy/2, Ly - dy/2, Ny)

nt = int(round(t_final / dt))

# Pasta de saída
OUTDIR = "figuras"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# REGIÕES DA FACE SUPERIOR COM FLUXO IMPOSTO
# -----------------------------
def top_boundary_flux(xc: float):
    """Retorna o fluxo de calor na face superior.
    Nas faixas aquecidas retorna q_flux; fora delas retorna None,
    indicando que a condição é convectiva.
    """
    if (0.003 <= xc <= 0.008) or (0.012 <= xc <= 0.017):
        return q_flux
    return None


# -----------------------------
# INDEXAÇÃO 2D -> 1D
# -----------------------------
def idx(i: int, j: int) -> int:
    return j * Nx + i


# -----------------------------
# UTILITÁRIO PARA ENCONTRAR O ÍNDICE MAIS PRÓXIMO
# -----------------------------
def nearest_index(arr, value):
    return int(np.argmin(np.abs(arr - value)))


# ================================================================
# MONTAGEM DO SISTEMA LINEAR A*T^{n+1} = b
# ================================================================
def assemble_system(T_old):
    """Monta a matriz do sistema implícito em volumes finitos.

    Equação de cada volume:
        a_P T_P = a_E T_E + a_W T_W + a_N T_N + a_S T_S + b

    As condições de contorno são incorporadas por linearização,
    gerando contribuições Su e Sp.
    """
    from scipy.sparse import lil_matrix, csc_matrix

    N = Nx * Ny
    A = lil_matrix((N, N))
    b = np.zeros(N, dtype=np.float64)

    # Áreas das faces e volume do controle
    Ae = Aw = dy * z
    An = As = dx * z
    V = dx * dy * z

    # Coeficiente temporal do termo implícito
    aP0 = rho_cp * V / dt

    for j in range(Ny):
        for i in range(Nx):
            p = idx(i, j)

            aE = aW = aN = aS = 0.0
            Su = 0.0
            Sp = 0.0

            # -------------------------
            # FACE LESTE
            # -------------------------
            if i < Nx - 1:
                aE = k * Ae / dx
            else:
                # Convecção na borda direita
                R_cond = (dx / 2.0) / (k * Ae)
                R_conv = 1.0 / (h * Ae)
                U = 1.0 / (R_cond + R_conv)
                Sp -= U
                Su += U * T_inf

            # -------------------------
            # FACE OESTE
            # -------------------------
            if i > 0:
                aW = k * Aw / dx
            else:
                # Convecção na borda esquerda
                R_cond = (dx / 2.0) / (k * Aw)
                R_conv = 1.0 / (h * Aw)
                U = 1.0 / (R_cond + R_conv)
                Sp -= U
                Su += U * T_inf

            # -------------------------
            # FACE NORTE (j = 0 no desenho)
            # -------------------------
            if j > 0:
                aN = k * An / dy
            else:
                qtop = top_boundary_flux(x[i])
                if qtop is not None:
                    # Fluxo imposto na face superior
                    Su += qtop * An
                else:
                    # Convecção na parte restante da face superior
                    R_cond = (dy / 2.0) / (k * An)
                    R_conv = 1.0 / (h * An)
                    U = 1.0 / (R_cond + R_conv)
                    Sp -= U
                    Su += U * T_inf

            # -------------------------
            # FACE SUL
            # -------------------------
            if j < Ny - 1:
                aS = k * As / dy
            else:
                # Convecção na borda inferior
                R_cond = (dy / 2.0) / (k * As)
                R_conv = 1.0 / (h * As)
                U = 1.0 / (R_cond + R_conv)
                Sp -= U
                Su += U * T_inf

            # Coeficiente central
            aP = aE + aW + aN + aS + aP0 - Sp
            A[p, p] = aP

            if i < Nx - 1:
                A[p, idx(i + 1, j)] = -aE
            if i > 0:
                A[p, idx(i - 1, j)] = -aW
            if j > 0:
                A[p, idx(i, j - 1)] = -aN
            if j < Ny - 1:
                A[p, idx(i, j + 1)] = -aS

            b[p] = aP0 * T_old[j, i] + Su

    return csc_matrix(A), b


# ================================================================
# SOLVER GPU OPCIONAL (JACOBI)
# ================================================================
def try_enable_cuda():
    """Tenta habilitar o uso de CUDA via numba.
    Retorna um dicionário com o estado do backend.
    """
    try:
        from numba import cuda
        ok = cuda.is_available()
        return {"enabled": bool(ok), "cuda": cuda if ok else None, "reason": None if ok else "CUDA não disponível"}
    except Exception as exc:
        return {"enabled": False, "cuda": None, "reason": f"numba/CUDA indisponível: {exc}"}


def jacobi_cpu(A_csr, b, x0=None, max_iter=20000, tol=1e-8):
    """Jacobi em CPU. Útil como referência e fallback iterativo."""
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    D = A_csr.diagonal()
    R = A_csr.copy()
    R.setdiag(0.0)
    R.eliminate_zeros()

    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        err = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if err < tol:
            break
    return x


def jacobi_gpu(A_csr, b, cuda_mod, x0=None, max_iter=20000, tol=1e-8):
    """Resolve Ax=b com Jacobi na GPU usando CSR.

    Observação: a matriz é montada na CPU e enviada para a GPU.
    Para este problema, a GPU é um recurso opcional, não obrigatório.
    """
    from numba import cuda, float64

    n = A_csr.shape[0]
    indptr = A_csr.indptr.astype(np.int32)
    indices = A_csr.indices.astype(np.int32)
    data = A_csr.data.astype(np.float64)

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = x0.astype(np.float64).copy()
    x_new = np.zeros_like(x)

    d_indptr = cuda.to_device(indptr)
    d_indices = cuda.to_device(indices)
    d_data = cuda.to_device(data)
    d_b = cuda.to_device(b.astype(np.float64))
    d_x = cuda.to_device(x)
    d_xnew = cuda.to_device(x_new)

    threads = 128
    blocks = (n + threads - 1) // threads

    @cuda.jit
    def jacobi_step(indptr_, indices_, data_, b_, x_, xnew_):
        row = cuda.grid(1)
        if row < b_.shape[0]:
            start = indptr_[row]
            end = indptr_[row + 1]
            diag = 0.0
            sigma = 0.0
            for kk in range(start, end):
                col = indices_[kk]
                val = data_[kk]
                if col == row:
                    diag = val
                else:
                    sigma += val * x_[col]
            xnew_[row] = (b_[row] - sigma) / diag

    for _ in range(max_iter):
        jacobi_step[blocks, threads](d_indptr, d_indices, d_data, d_b, d_x, d_xnew)
        x_old = d_x.copy_to_host()
        x_new_host = d_xnew.copy_to_host()
        err = np.linalg.norm(x_new_host - x_old, ord=np.inf)
        d_x, d_xnew = d_xnew, d_x
        if err < tol:
            break

    return d_x.copy_to_host()


# ================================================================
# ESCOLHA DO SOLVER
# ================================================================
def solve_linear_system(A, b, prefer_gpu=True):
    """Resolve o sistema linear usando GPU se disponível.
    Fallback principal: solver direto esparso na CPU.
    """
    if prefer_gpu:
        cuda_state = try_enable_cuda()
        if cuda_state["enabled"]:
            try:
                A_csr = A.tocsr()
                x = jacobi_gpu(A_csr, b, cuda_state["cuda"])
                return x, "GPU (CUDA/Jacobi)"
            except Exception as exc:
                print(f"[aviso] Falha no solver GPU: {exc}")

    try:
        from scipy.sparse.linalg import spsolve
        x = spsolve(A, b)
        return x, "CPU (spsolve esparso)"
    except Exception as exc:
        print(f"[aviso] Falha no spsolve: {exc}")
        A_csr = A.tocsr()
        x = jacobi_cpu(A_csr, b)
        return x, "CPU (Jacobi em série)"


# ================================================================
# PÓS-PROCESSAMENTO E PLOTS
# ================================================================
def save_contour(Tfield, time_value, filename):
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(8, 4))
    cont = plt.contourf(X, Y, Tfield, levels=30)
    plt.colorbar(cont, label='Temperatura [°C]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Distribuição de temperatura em t = {time_value:.1f} s')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


def save_time_history(times, T1, T2, T3, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(times, T1, label='T(x=0.01, y≈0.00, t)')
    plt.plot(times, T2, label='T(x=0.01, y≈0.005, t)')
    plt.plot(times, T3, label='T(x=0.01, y≈0.01, t)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Temperatura [°C]')
    plt.title('Evolução temporal da temperatura nos pontos monitorados')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


def save_surface_plot(Tfield, time_value, filename):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Tfield, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.7, label='Temperatura [°C]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('T [°C]')
    ax.set_title(f'Superfície de temperatura em t = {time_value:.1f} s')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


# ================================================================
# EXECUÇÃO PRINCIPAL
# ================================================================
def main():
    # Campo inicial
    T = np.full((Ny, Nx), T0, dtype=np.float64)

    # Pontos pedidos no enunciado
    ix = nearest_index(x, 0.01)
    jy_top = nearest_index(y, 0.0)
    jy_mid = nearest_index(y, 0.005)
    jy_bot = nearest_index(y, 0.01)

    times = [0.0]
    T_top_hist = [T[jy_top, ix]]
    T_mid_hist = [T[jy_mid, ix]]
    T_bot_hist = [T[jy_bot, ix]]

    # Instantes para salvar campos 2D
    snapshot_times = [1, 5, 10, 30, 60]
    snapshots = {0.0: T.copy()}

    solver_used = None

    for n in range(1, nt + 1):
        t = n * dt
        A, b = assemble_system(T)
        T_vec, solver_used = solve_linear_system(A, b, prefer_gpu=True)
        T = T_vec.reshape((Ny, Nx))

        times.append(t)
        T_top_hist.append(T[jy_top, ix])
        T_mid_hist.append(T[jy_mid, ix])
        T_bot_hist.append(T[jy_bot, ix])

        for ts in snapshot_times:
            if abs(t - ts) < dt / 2.0:
                snapshots[ts] = T.copy()

    print(f"Solver utilizado: {solver_used}")
    print(f"T(x=0.01, y≈0.00, t=60 s)  = {T[jy_top, ix]:.6f} °C")
    print(f"T(x=0.01, y≈0.005, t=60 s) = {T[jy_mid, ix]:.6f} °C")
    print(f"T(x=0.01, y≈0.01, t=60 s)  = {T[jy_bot, ix]:.6f} °C")

    # Salva gráficos pertinentes
    save_time_history(times, T_top_hist, T_mid_hist, T_bot_hist, 'historico_temperatura.png')
    save_contour(T, t_final, 'campo_temperatura_final.png')
    save_surface_plot(T, t_final, 'superficie_temperatura_final.png')

    for ts in [0.0, 1, 5, 10, 30, 60]:
        if ts in snapshots:
            fname = f'campo_temperatura_t_{str(ts).replace(".", "_")}.png'
            save_contour(snapshots[ts], ts, fname)


if __name__ == '__main__':
    main()
