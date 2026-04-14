import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parâmetros
# -------------------------
Nx, Ny = 41, 31
Lx, Ly = 1.0, 0.5

dx = Lx / (Nx - 2)
dy = Ly / (Ny - 2)

a = 0.01
b = 5.0
dt = 1e-5

h1 = 25.0
h2 = 15.0
Tinf1 = 293.0
Tinf2 = 290.0
Tsur = 285.0

eps = 0.8
sigma = 5.67e-8

# -------------------------
# Função k(T)
# -------------------------
def k_func(T):
    return a*T + b

# -------------------------
# Inicialização
# -------------------------
T = np.ones((Nx, Ny)) * 300.0

# -------------------------
# Loop temporal
# -------------------------
for n in range(500):

    Tn = T.copy()

    # -------- interior --------
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            kE = k_func(Tn[i+1,j])
            kW = k_func(Tn[i-1,j])
            kN = k_func(Tn[i,j+1])
            kS = k_func(Tn[i,j-1])

            flux_x = (kE*(Tn[i+1,j]-Tn[i,j]) - kW*(Tn[i,j]-Tn[i-1,j])) / dx**2
            flux_y = (kN*(Tn[i,j+1]-Tn[i,j]) - kS*(Tn[i,j]-Tn[i,j-1])) / dy**2

            T[i,j] = Tn[i,j] + dt*(flux_x + flux_y)

    # -------- Dirichlet --------
    T[0,:] = 500.0

    # -------- Adiabático --------
    T[-1,:] = T[-2,:]

    # -------- Convecção --------
    for i in range(1, Nx-1):
        Tb = T[i,0]

        for _ in range(10):
            k_val = k_func(Tb)
            f = -k_val*(T[i,1]-Tb)/dy - h1*(Tb - Tinf1)
            df = (a*(Tb - T[i,1]) + k_val)/dy - h1
            Tb -= f/df

        T[i,0] = Tb

    # -------- Convecção + radiação --------
    for i in range(1, Nx-1):
        Tb = T[i,-1]

        for _ in range(15):
            k_val = k_func(Tb)
            f = -k_val*(Tb - T[i,-2])/dy - h2*(Tb - Tinf2) - eps*sigma*(Tb**4 - Tsur**4)
            df = -(a*(Tb - T[i,-2]) + k_val)/dy - h2 - 4*eps*sigma*Tb**3
            Tb -= f/df

        T[i,-1] = Tb

# -------------------------
# Plot
# -------------------------
plt.imshow(T.T, origin='lower', aspect='auto')
plt.colorbar()
plt.title("Questão 2 - k(T) - Diferenças Finitas")
plt.show()