import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.cos(x)

def df(x):
    return -np.sin(x)

def d2(f , dx, n):
    return (f[2:, n] - 2*f[1:-1, n] + f[:-2, n]) / dx**2

def finite_difference_central(x, dx):
    resp = (x[2:] - x[:-2]) / (2*dx)
    # resp[0] = (x[1]- x[0]) / dx
    # resp[-1] = (x[-1] - x[-2]) / dx
    return resp

def finite_difference_forward(x, dx):
    resp = (x[1:] - x[:-1]) / dx
    # resp[-1] = (x[-1] - x[-2]) / dx
    return resp

def finite_difference_backward(x, dx):
    resp = (x[1:] - x[:-1]) / dx
    # resp[0] = (x[1]- x[0]) / dx
    return resp