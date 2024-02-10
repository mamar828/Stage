import numpy as np
from astropy.io import fits

from coords import *



class Header():
    def __init__(self, header: fits.Header):
        self.header = header
    
    def to_galactic(self):
        RA_key = None
        DEC_key = None
        for key, value in self.header.items():
            # Disregard bool values
            if isinstance(value, str):
                if "RA" in value:
                    RA_key = int(key[-1])
                elif "DEC" in value:
                    DEC_key = int(key[-1])
        
        self.header[f"CTYPE{RA_key}"] = "GLON-SFL"
        self.header[f"CTYPE{DEC_key}"] = "GLAT-SFL"

        for element in ["CRVAL"]:
            ra = RA.from_sexagesimal(self.header[f"{element}{RA_key}"])
            dec = DEC.from_sexagesimal(self.header[f"{element}{DEC_key}"])
            
            self.header[f"{element}{RA_key}"] = l.from_equatorial(ra, dec).sexagesimal
            self.header[f"{element}{DEC_key}"] = b.from_equatorial(ra, dec).sexagesimal
        
        return self.header
