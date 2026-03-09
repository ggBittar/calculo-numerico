import numpy as np

METHOD_NAME = "Diferenças Finitas"


def estimate(points, values, axis, spacings):
    h = spacings[axis]
    result = np.empty_like(values, dtype=float)
    center = [slice(None)] * values.ndim
    prev_ = [slice(None)] * values.ndim
    next_ = [slice(None)] * values.ndim
    center[axis], prev_[axis], next_[axis] = slice(1, -1), slice(0, -2), slice(2, None)
    result[tuple(center)] = (values[tuple(next_)] - values[tuple(prev_)]) / (2.0 * h)
    left = [slice(None)] * values.ndim
    left_n = [slice(None)] * values.ndim
    left[axis], left_n[axis] = 0, 1
    result[tuple(left)] = (values[tuple(left_n)] - values[tuple(left)]) / h
    right = [slice(None)] * values.ndim
    right_p = [slice(None)] * values.ndim
    right[axis], right_p[axis] = -1, -2
    result[tuple(right)] = (values[tuple(right)] - values[tuple(right_p)]) / h
    return result
