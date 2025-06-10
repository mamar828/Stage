import graphinglib as gl

from src.hdu.cubes.cube_co import CubeCO
from src.hdu.tesseract import Tesseract
from src.coordinates.ds9_coords import DS9Coords


p = CubeCO.load("data/Loop4/p/12co/Loop4p_wcs.fits")[500:850,:,:].bin((1,2,2))
total_object_p = Tesseract.load(f"data/Loop4/p/12co/object_filtered_2.fits")

# fig = gl.Figure(size=(10, 7))
# spec = p[:,*DS9Coords(8,8)]
# spec.setattrs({
#     "PEAK_PROMINENCE" : 0.4,
#     "PEAK_MINIMUM_DISTANCE" : 6,
#     "PEAK_WIDTH" : 2.5,
#     "INITIAL_GUESSES_BINNING" : 2,
#     "MAX_RESIDUE_SIGMAS" : 5
# })

# spec.fit()
# spec.fit()
# fig.add_elements(spec.plot, spec.total_functions_plot)
# fig.add_elements(*total_object_p.get_spectrum_plot(p, DS9Coords(8, 8)))
# fig.show()



for y in range(7, 22):
    for x in range(7, 17):
        fig = gl.Figure(size=(10, 7))
        try:
            fig.add_elements(*total_object_p.get_spectrum_plot(p, DS9Coords(x, y)))
        except KeyError:
            continue
        print(x, y)
        fig.show()

"""
Notes :
8, 8,
12, 8,
15, 9,
12, 10,
9, 11,
13, 14,
13, 15,
8, 16,
13, 16,
13, 17,
13, 18,
14, 19,
8, 21


9, 7

9, 9        -
10, 9       -
11, 9       -
12, 11      -
12, 12      -
9, 13       -

13, 13
9, 17
13, 17
15, 19
13, 21
"""

