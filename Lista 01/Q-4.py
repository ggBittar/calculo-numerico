import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parâmetros
# -------------------------
Nx, Ny = 41, 31
Lx, Ly = 1.0, 0.5

dx = Lx / (Nx - 2)
dy = Ly / (Ny - 2)

alpha = 1e-5
k = 10.0
dt = 1
e-5
h1 = 25.0
h2 = 15.0
Tinf1 = 293.0
Tinf2 = 290.0
Tsur = 285.0

eps = 0.8
sigma = 5.67e-8

# -------------------------
# Malha com ghost cells
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
            d2x = (Tn[i+1,j] - 2*Tn[i,j] + Tn[i-1,j]) / dx**2
            d2y = (Tn[i,j+1] - 2*Tn[i,j] + Tn[i,j-1]) / dy**2
            T[i,j] = Tn[i,j] + alpha*dt*(d2x + d2y)

    # -------- (a) Dirichlet --------
    T[0,:] = 500.0

    # -------- (b) Adiabático --------
    T[-1,:] = T[-2,:]

    # -------- (c) Convecção --------
    for i in range(1, Nx-1):
        T[i,0] = T[i,1] + dy*(h1/k)*(Tinf1 - T[i,0])

    # -------- (d) Convecção + radiação --------
    for i in range(1, Nx-1):
        Tb = T[i,-1]

        for _ in range(10):
            f = -k*(Tb - T[i,-2])/dy - h2*(Tb - Tinf2) - eps*sigma*(Tb**4 - Tsur**4)
            df = -k/dy - h2 - 4*eps*sigma*Tb**3
            Tb -= f/df

        T[i,-1] = Tb

# -------------------------
# Plot
# -------------------------
plt.imshow(T.T, origin='lower', aspect='auto')
plt.colorbar()
plt.title("Questão 1 - Diferenças Finitas")
plt.show()