import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.tesseract import Tesseract
from src.hdu.maps.grouped_maps import GroupedMaps


cube = CubeCO.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")[500:800,:,:]
print(500 + cube.header.get_coordinate(-4000, axis=0))

tesseract = Tesseract.load("data/Loop4_co/N1/fit_tesseract.fits")
tesseract_splitted = tesseract.split(25, 0)
# filtered_tesseracts = 