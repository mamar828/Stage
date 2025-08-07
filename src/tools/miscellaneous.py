import os
import warnings
from contextlib import redirect_stdout


def silence_function(func):
    """
    Decorates verbose functions to silence their terminal output.
    """
    def inner_func(*args, **kwargs):
        with open(os.devnull, "w") as outer_space, redirect_stdout(outer_space), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return inner_func
