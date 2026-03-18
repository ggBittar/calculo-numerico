# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp


ctypedef cnp.float64_t DTYPE_t
cnp.import_array()


cdef inline double _apply_boundary_value(
    int condition_type,
    double value,
    double robin_alpha,
    double robin_beta,
    double robin_gamma,
    double neighbor,
    double delta,
) except *:
    cdef double denom
    if condition_type == 0:  # dirichlet
        return value
    if condition_type == 1:  # neumann
        return neighbor + value * delta
    if condition_type == 2:  # robin
        denom = robin_alpha + (robin_beta / delta)
        if -1e-15 < denom < 1e-15:
            raise ValueError("Condicao de Robin invalida: a + b/delta nao pode ser zero.")
        return (robin_gamma + (robin_beta / delta) * neighbor) / denom
    raise ValueError(f"Tipo de contorno nao suportado: {condition_type}")


cpdef cnp.ndarray[DTYPE_t, ndim=2] explicit_euler_heat_1d(
    cnp.ndarray[DTYPE_t, ndim=1] initial,
    cnp.ndarray[DTYPE_t, ndim=1] x,
    cnp.ndarray[DTYPE_t, ndim=1] t,
    double alpha,
    int left_type,
    double left_value,
    double left_robin_alpha,
    double left_robin_beta,
    double left_robin_gamma,
    int right_type,
    double right_value,
    double right_robin_alpha,
    double right_robin_beta,
    double right_robin_gamma,
):
    cdef Py_ssize_t n, i
    cdef Py_ssize_t nt = t.shape[0]
    cdef Py_ssize_t nx = x.shape[0]
    cdef double dx = x[1] - x[0]
    cdef double dt = t[1] - t[0]
    cdef double factor = alpha * dt / (dx * dx)
    cdef cnp.ndarray[DTYPE_t, ndim=2] solution = np.zeros((nt, nx), dtype=np.float64)

    for i in range(nx):
        solution[0, i] = initial[i]

    solution[0, 0] = _apply_boundary_value(
        left_type, left_value, left_robin_alpha, left_robin_beta, left_robin_gamma, solution[0, 1], dx
    )
    solution[0, nx - 1] = _apply_boundary_value(
        right_type, right_value, right_robin_alpha, right_robin_beta, right_robin_gamma, solution[0, nx - 2], dx
    )

    for n in range(nt - 1):
        for i in range(1, nx - 1):
            solution[n + 1, i] = solution[n, i] + factor * (
                solution[n, i + 1] - 2.0 * solution[n, i] + solution[n, i - 1]
            )

        solution[n + 1, 0] = _apply_boundary_value(
            left_type, left_value, left_robin_alpha, left_robin_beta, left_robin_gamma, solution[n + 1, 1], dx
        )
        solution[n + 1, nx - 1] = _apply_boundary_value(
            right_type,
            right_value,
            right_robin_alpha,
            right_robin_beta,
            right_robin_gamma,
            solution[n + 1, nx - 2],
            dx,
        )

    return solution


cpdef cnp.ndarray[DTYPE_t, ndim=3] explicit_euler_heat_2d(
    cnp.ndarray[DTYPE_t, ndim=2] initial,
    cnp.ndarray[DTYPE_t, ndim=1] x,
    cnp.ndarray[DTYPE_t, ndim=1] y,
    cnp.ndarray[DTYPE_t, ndim=1] t,
    double alpha,
    int x_min_type,
    double x_min_value,
    double x_min_robin_alpha,
    double x_min_robin_beta,
    double x_min_robin_gamma,
    int x_max_type,
    double x_max_value,
    double x_max_robin_alpha,
    double x_max_robin_beta,
    double x_max_robin_gamma,
    int y_min_type,
    double y_min_value,
    double y_min_robin_alpha,
    double y_min_robin_beta,
    double y_min_robin_gamma,
    int y_max_type,
    double y_max_value,
    double y_max_robin_alpha,
    double y_max_robin_beta,
    double y_max_robin_gamma,
):
    cdef Py_ssize_t n, i, j
    cdef Py_ssize_t nt = t.shape[0]
    cdef Py_ssize_t nx = x.shape[0]
    cdef Py_ssize_t ny = y.shape[0]
    cdef double dx = x[1] - x[0]
    cdef double dy = y[1] - y[0]
    cdef double dt = t[1] - t[0]
    cdef double fx = alpha * dt / (dx * dx)
    cdef double fy = alpha * dt / (dy * dy)
    cdef cnp.ndarray[DTYPE_t, ndim=3] solution = np.zeros((nt, nx, ny), dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            solution[0, i, j] = initial[i, j]

    for j in range(ny):
        solution[0, 0, j] = _apply_boundary_value(
            x_min_type,
            x_min_value,
            x_min_robin_alpha,
            x_min_robin_beta,
            x_min_robin_gamma,
            solution[0, 1, j],
            dx,
        )
        solution[0, nx - 1, j] = _apply_boundary_value(
            x_max_type,
            x_max_value,
            x_max_robin_alpha,
            x_max_robin_beta,
            x_max_robin_gamma,
            solution[0, nx - 2, j],
            dx,
        )

    for i in range(nx):
        solution[0, i, 0] = _apply_boundary_value(
            y_min_type,
            y_min_value,
            y_min_robin_alpha,
            y_min_robin_beta,
            y_min_robin_gamma,
            solution[0, i, 1],
            dy,
        )
        solution[0, i, ny - 1] = _apply_boundary_value(
            y_max_type,
            y_max_value,
            y_max_robin_alpha,
            y_max_robin_beta,
            y_max_robin_gamma,
            solution[0, i, ny - 2],
            dy,
        )

    for n in range(nt - 1):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                solution[n + 1, i, j] = solution[n, i, j] + fx * (
                    solution[n, i + 1, j] - 2.0 * solution[n, i, j] + solution[n, i - 1, j]
                ) + fy * (
                    solution[n, i, j + 1] - 2.0 * solution[n, i, j] + solution[n, i, j - 1]
                )

        for j in range(ny):
            solution[n + 1, 0, j] = _apply_boundary_value(
                x_min_type,
                x_min_value,
                x_min_robin_alpha,
                x_min_robin_beta,
                x_min_robin_gamma,
                solution[n + 1, 1, j],
                dx,
            )
            solution[n + 1, nx - 1, j] = _apply_boundary_value(
                x_max_type,
                x_max_value,
                x_max_robin_alpha,
                x_max_robin_beta,
                x_max_robin_gamma,
                solution[n + 1, nx - 2, j],
                dx,
            )

        for i in range(nx):
            solution[n + 1, i, 0] = _apply_boundary_value(
                y_min_type,
                y_min_value,
                y_min_robin_alpha,
                y_min_robin_beta,
                y_min_robin_gamma,
                solution[n + 1, i, 1],
                dy,
            )
            solution[n + 1, i, ny - 1] = _apply_boundary_value(
                y_max_type,
                y_max_value,
                y_max_robin_alpha,
                y_max_robin_beta,
                y_max_robin_gamma,
                solution[n + 1, i, ny - 2],
                dy,
            )

    return solution
