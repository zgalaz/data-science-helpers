import numpy as np


def check_observation_count(func):
    def func_wrapper(x, y, *args, **kwargs):
        if not x.shape[0] == y.shape[0]:
            raise ValueError("Input arrays must have the same number of rows (observations)")
        return func(x, y, *args, **kwargs)
    return func_wrapper


def check_numpy_array(func):
    def func_wrapper(x, y, *args, **kwargs):
        if not all((isinstance(x, np.ndarray), isinstance(y, np.ndarray))):
            raise TypeError("Input arrays must be of type np.ndarray")
        return func(x, y, *args, **kwargs)
    return func_wrapper
