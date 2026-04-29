## imports

import numpy as np
import matplotlib.pyplot as plt

## Parametros

nx = 5
Lx = 1.0
dx = Lx/(nx-1)

x = np.linspace(0, Lx, nx)
a = 1.0
C = 0.5

dt = C * dx**2 / a

tf = 1.0

t = np.arange(0, tf+dt, dt)

