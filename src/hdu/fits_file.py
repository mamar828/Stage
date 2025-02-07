from astropy.io import fits
from copy import deepcopy
import os
from contextlib import redirect_stdout
from asyncio import run as asyncio_run
from telegram_send import send as telegram_send
from time import time
from eztcolors import Colors as C


class FitsFile:
    """
    Encapsulates the methods specific to .fits files.
    """

    def copy(self):
        return deepcopy(self)

    @staticmethod
    def save(filename: str, hdu_list: fits.HDUList, overwrite: bool=False):
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
                    decision = input(f"{C.RED}{filename} already exists, do you wish to overwrite it? [y/n]{C.END}")
                    if decision.lower() == "y":
                        hdu_list.writeto(filename, overwrite=True, output_verify="warn")
                        print(f"{C.LIGHT_GREEN}File overwritten.{C.END}")
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
        def inner_func(self, *args, **kwargs):
            with open(os.devnull, "w") as outer_space, redirect_stdout(outer_space):
                return func(self, *args, **kwargs)
            
        return inner_func

    def _notification_send(message: str):
        """
        Sends a notification message via Telegram. This function is called by the notify function.

        Parameters
        ----------
        message : str
            The message to be sent.
        """
        try:
            asyncio_run(telegram_send(messages=[message]))
        except:
            print("No telegram bot configuration was available.")

    @staticmethod
    def notify(func):
        """
        Decorates a function to notify when it has finished running.
        """
        def inner_func(self, *args, **kwargs):
            start_time = time()
            result = func(self, *args, **kwargs)
            FitsFile._notification_send(f"{func.__name__} has finished running in {round(time()-start_time)}s.")
            return result
        
        return inner_func
