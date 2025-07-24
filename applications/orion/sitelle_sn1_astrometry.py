import numpy as np
from astropy.io import fits
import pyregion
import src.graphinglib as gl

from src.hdu.cubes.cube import Cube
from src.hdu.maps.map import Map
import src.tools.astrometry as astrometry


# Get the deep frame
# ------------------
# c = Cube.load(f"data/orion/astrometry/sitelle_sn1/Orion1_SN1.merged.cm1.1.0.new.fits")
# c.header["CTYPE3"] = "WAVE-"
# c.get_deep_frame().save(f"data/orion/astrometry/sitelle_sn1/deep_frame.fits")

# Detect stars and fit
# --------------------
m = Map.load(f"data/orion/astrometry/sitelle_sn1/deep_frame.fits")

star_positions = astrometry.detect_stars_from_regions(
    m,
    pyregion.open("data/orion/astrometry/sitelle_sn1/stars.reg"),
)
astrometry.get_detection_figure(m.data, star_positions).show()
# fig = gl.SmartFigure(1, 2, size=(13, 6), elements=[
#     [m.data.plot, *[gl.Point(x, y, marker_style="+", label=str(i+1))
#                     for i, (x, y) in enumerate(zip(star_positions["xcentroid"], star_positions["ycentroid"]))]],
#     [m.data.plot, *[gl.Point(x, y, marker_style="+", label=str(i+1))
#                     for i, (x, y) in enumerate(zip(star_positions["x_fit"], star_positions["y_fit"]))]]
# ])
# fig[0][0].color_map_range = 50, 500
# fig[1][0].color_map_range = 50, 500
# fig.show()

lines = open(f"data/orion/astrometry/sitelle_sn1/list_star_SN1_field.txt", "r").readlines()
wcs_coords = [line.split(",")[0] for line in lines]
new_wcs = astrometry.get_wcs_transformation(
    pixel_coords=np.column_stack((star_positions["x_fit"], star_positions["y_fit"])),
    wcs_coords=wcs_coords
)
# m.header = astrometry.update_header_wcs(m.header, new_wcs)
# m.save(f"data/orion/astrometry/sitelle_sn1/deep_frame_wcs.fits")

# c.header = astrometry.update_header_wcs(c.header, new_wcs)
# c.header["CTYPE3"] = "WAVE-SIP"
# c.save(f"data/orion/astrometry/sitelle_sn1/Orion1_SN1.merged.cm1.1.0.new_wcs.fits")
