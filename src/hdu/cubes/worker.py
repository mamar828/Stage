from typing import Any


def _worker_split(obj, i: int) -> Any:
    """
    Splits a worker task to the object by calling the method obj._worker with the specified index. This function exists
    to enable multiprocessing as the children tasks need to be able to access directly the function.
    
    Parameters
    ----------
    obj : object
        Object that has a obj._worker method.
    i : int
        Index that will be passed as an argument to the called methods.
        
    Returns
    -------
    Any
        The results are given by the obj._worker method and do not need to be of a specific type.
    """
    return obj._worker(i)
