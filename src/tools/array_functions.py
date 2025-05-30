import numpy as np


def list_to_array(lst: list, padding_value: float=np.nan) -> np.ndarray:
    """
    Converts a 2D inhomogeneous list to a numpy array using padding to make the array rectangular.

    Parameters
    ----------
    lst : list
        The list to convert.
    padding_value : float, default=np.nan
        The value to use for padding the shorter lists.

    Returns
    -------
    np.ndarray
        The converted numpy array.
    """
    max_length = max(len(sub_lst) for sub_lst in lst)
    if isinstance(lst[0], (list, tuple)):
        array = np.array(
            [np.pad(np.array(sub_lst).astype(float), (0, max_length - len(sub_lst)), constant_values=padding_value) 
            for sub_lst in lst]
        )
    elif isinstance(lst[0], np.ndarray):
        array = np.array(
            [np.pad(sub_lst.astype(float), (0, max_length - len(sub_lst)), constant_values=padding_value) 
            for sub_lst in lst]
        )
    else:
        raise ValueError("Unsupported list type. Only lists of lists, tuples, or numpy arrays are supported.")
    return array
