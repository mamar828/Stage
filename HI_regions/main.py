from astropy.io import fits

from shear_detection import *
from coords import *


def analyze_LOOP4():
    HI = LOOP4_cube(fits.open("HI_regions/data_cubes/LOOP4/LOOP4_FINAL_GLS.fits")).swap_axes({"x":"v", "y":"l", "z":"b"})

    shear_points = HI.extract_shear(
        y_bounds=[300,350], 
        z_bounds=[b("32:29:29.077"),b("32:31:15.078")], 
        tolerance=0.5,
        max_regroup_separation=5, 
        pixel_width=3, 
        max_accepted_shear=5
    )

    HI.watch_shear(shear_points, fullscreen=True)


# analyze_LOOP4()


def analyze_spider():
    HI_raw = Spider_cube(fits.open("HI_regions/data_cubes/spider/spider_vlb.fits"),
                         axes_info={"x":"v","y":"l","z":"b"})
    HI = HI_raw.invert_axis("x")

    shear_points = HI.extract_shear(
        y_bounds=[400,1000],
        z_bounds=[400,600],
        rejection=1.1,
        max_regroup_separation=10,
        accepted_width=3,
        max_accepted_shear=15
    )

    HI.watch_shear(shear_points, fullscreen=True)


analyze_spider()
