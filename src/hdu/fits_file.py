import os
import warnings
from typing import Self
from astropy.io import fits
from copy import deepcopy
from contextlib import redirect_stdout
from colorist import BrightColor as C


class FitsFile:
    """
    Encapsulates the methods specific to .fits files.
    """

    def copy(self) -> Self:
        return deepcopy(self)

    @staticmethod
    def save(filename: str, hdu_list: fits.HDUList, overwrite: bool = False):
        """
        Saves to a file.

        Parameters
        ----------
        filename : str
            Indicates the path and name of the created file. If the file already exists, a warning will appear and the
            file can be overwritten.
        hdu_list : fits.HDUList
            List of HDU objects to save.
        overwrite : bool, default=False
            Specifies if the file should automatically be erased.
        """
        try:
            hdu_list.writeto(filename, overwrite=overwrite, output_verify="warn")
        except OSError as exception:
            # Only catch exceptions related to an already existing file
            if isinstance(exception.args[0], str) and exception.args[0][-31:] == " the argument \"overwrite=True\".":
                while True:
                    decision = input(f"{C.RED}{filename} already exists, do you wish to overwrite it? [y/n]{C.OFF}")
                    if decision.lower() == "y":
                        hdu_list.writeto(filename, overwrite=True, output_verify="warn")
                        print(f"{C.GREEN}File overwritten.{C.OFF}")
                        break
                    elif decision.lower() == "n":
                        break
            else:
                raise exception

    @staticmethod
    def silence_function(func):
        """
        Decorates verbose functions to silence their terminal output.
        """
        def inner_func(*args, **kwargs):
            with open(os.devnull, "w") as outer_space, redirect_stdout(outer_space), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return func(*args, **kwargs)

        return inner_func
