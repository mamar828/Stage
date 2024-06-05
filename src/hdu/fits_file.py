from astropy.io import fits
from eztcolors import Colors as C


class FitsFile:
    """
    Encapsulates the methods specific to .fits files.
    """

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
            hdu_list.writeto(filename, overwrite=overwrite)
        except OSError:
            while True:
                decision = input(f"{C.RED}{filename} already exists, do you wish to overwrite it? [y/n]{C.END}")
                if decision.lower() == "y":
                    hdu_list.writeto(filename, overwrite=True)
                    print(f"{C.LIGHT_GREEN}File overwritten.{C.END}")
                    break
                elif decision.lower() == "n":
                    break
