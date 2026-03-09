import numpy as np

METHOD_NAME = "Diferença Regressiva"


def estimate(points, values, axis, spacings):
    h = spacings[axis]
    result = np.empty_like(values, dtype=float)

    cur = [slice(None)] * values.ndim
    prev_ = [slice(None)] * values.ndim
    cur[axis], prev_[axis] = slice(1, None), slice(0, -1)
    result[tuple(cur)] = (values[tuple(cur)] - values[tuple(prev_)]) / h

    first = [slice(None)] * values.ndim
    next_ = [slice(None)] * values.ndim
    first[axis], next_[axis] = 0, 1
    result[tuple(first)] = (values[tuple(next_)] - values[tuple(first)]) / h
    return result
