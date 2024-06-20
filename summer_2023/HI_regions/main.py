from astropy.io import fits

from shear_detection import *
from coords import *


def analyze_LOOP4():
    HI = LOOP4_cube(fits.open("summer_2023/HI_regions/data_cubes/LOOP4/LOOP4_bin2.fits")).swap_axes({"x":"v", "y":"b", "z":"l"})
    # HI.save_as_fits_file("t.fits")

    shear_points = HI.extract_shear(
        y_bounds=[100,400], 
        z_bounds=[300, 350], 
        # z_bounds=[b("32:29:53.077"),b("32:31:15.078")], 
        tolerance=0.5,
        max_regroup_separation=5, 
        pixel_width=3,
        accepted_width=1,
        max_accepted_shear=5
    )

    HI.watch_shear(shear_points, fullscreen=True)


analyze_LOOP4()


def analyze_spider():
    HI_raw = Spider_cube(fits.open("summer_2023/HI_regions/data_cubes/spider/spider_vlb.fits"),
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


# analyze_spider()
