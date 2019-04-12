import numpy as np
import pandas as pd


# ---------------
# Type validators
# ---------------

def validate_df_dataframe(func):
    def func_wrapper(df, *args, **kwargs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input dataframe must be of type pd.DataFrame")
        return func(df, *args, **kwargs)
    return func_wrapper


def validate_x_y_numpy_array(func):
    def func_wrapper(x, y, *args, **kwargs):
        if not all((isinstance(x, np.ndarray), isinstance(y, np.ndarray))):
            raise TypeError("Input arrays must be of type np.ndarray")
        return func(x, y, *args, **kwargs)
    return func_wrapper


def validate_x_y_str(func):
    def func_wrapper(x, y, *args, **kwargs):
        if not all((isinstance(x, str), isinstance(y, str))):
            raise TypeError("Input arrays must be of type str")
        return func(x, y, *args, **kwargs)
    return func_wrapper


def validate_args_numpy_array(func):
    def func_wrapper(*args, **kwargs):
        if not all([True if isinstance(arg, np.ndarray) else False for arg in args]):
            raise TypeError("Input positional arguments must be of type np.ndarray")
        return func(*args, **kwargs)
    return func_wrapper


# ---------------------
# Dimensions validators
# ---------------------

def validate_args_one_dimensional(func):
    def func_wrapper(*args, **kwargs):
        if not all([True if arg.ndim == 1 else False for arg in args]):
            raise ValueError("Input arrays must be one dimensional")
        return func(*args, **kwargs)
    return func_wrapper


def validate_x_y_observation_count(func):
    def func_wrapper(x, y, *args, **kwargs):
        if not x.shape[0] == y.shape[0]:
            raise ValueError("Input arrays must have the same number of rows (observations)")
        return func(x, y, *args, **kwargs)
    return func_wrapper


def validate_args_two_dimensional(func):
    def func_wrapper(*args, **kwargs):
        if not all([True if arg.ndim == 2 else False for arg in args]):
            raise ValueError("Input arrays must be one dimensional")
        return func(*args, **kwargs)
    return func_wrapper


def validate_args_rank_one_or_one_dimensional(func):
    def func_wrapper(*args, **kwargs):
        if not all([True if arg.ndim == 1 or any(a == 1 for a in arg.shape) else False for arg in args]):
            raise ValueError("Input arrays must be one dimensional")
        return func(*args, **kwargs)
    return func_wrapper
