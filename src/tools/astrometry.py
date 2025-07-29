import numpy as np
import src.graphinglib as gl
import astropy.units as u
import pyregion

from typing import Iterable
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.modeling import models, fitting
from astropy.table import Table
from astropy.coordinates import SkyCoord, FK5, ICRS
from astropy.wcs.utils import fit_wcs_from_points
from astropy.wcs import WCS
from scipy.signal import peak_widths

from src.headers.header import Header
from src.hdu.maps.map import Map


def fit_star_position(image: np.ndarray, star_positions: Table) -> Table:
    """
    Fits the star positions using a two-dimensional Gaussian model and a two-dimensional plane.

    Parameters
    ----------
    star_positions : Table
        Table of detected stars with their properties such as position, flux, and background.
    image : np.ndarray
        The image in which the stars were detected.

    Returns
    -------
    Table
        Table of fitted star positions with their properties.
    """
    fitted_x = []
    fitted_y = []
    fitted_fwhm = []
    for i, star in enumerate(star_positions, start=1):
        x = star["xcentroid"]
        y = star["ycentroid"]
        box_apothem = 10        # This is the half-width of the box around the star to fit to constrain the model
        x_min = int(max(x - box_apothem, 0))
        x_max = int(min(x + box_apothem + 1, image.shape[1]))
        y_min = int(max(y - box_apothem, 0))
        y_max = int(min(y + box_apothem + 1, image.shape[0]))
        box = image[y_min:y_max, x_min:x_max]
        y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]

        background = np.nanmedian(box)
        amplitude_guess = float(box.max() - background)

        # Estimate the FWHM of the star in the box
        x_width = peak_widths(box[box_apothem,:], [box_apothem])[0]
        y_width = peak_widths(box[:,box_apothem], [box_apothem])[0]

        model = (
            models.Gaussian2D(
                amplitude=amplitude_guess,
                x_mean=x,
                y_mean=y,
                x_stddev=x_width / 2.3548,
                y_stddev=y_width / 2.3548,
                theta=0,
                bounds={"x_mean": (x_min, x_max), "y_mean": (y_min, y_max), "amplitude": (0, 3*amplitude_guess)})
            + models.Planar2D(background, 0, 0)
        )
        fit_p = fitting.LevMarLSQFitter()
        params = fit_p(model, x_grid, y_grid, box, maxiter=100000)
        fitted_x.append(params[0].x_mean.value)
        fitted_y.append(params[0].y_mean.value)
        fitted_fwhm.append(np.mean([params[0].x_fwhm, params[0].y_fwhm]))

        # Show the fitted model and the profile along x
        # ---------------------------------------------
        # print(x, y, fitted_fwhm[-1])
        # gl.SmartFigure(elements=[
        #     gl.Curve(x_space := np.arange(box_apothem*2 + 1) + x_min,
        #              box[y_val := min(round(params[0].y_mean.value - y_min), box_apothem*2)],
        #              label="Profile along x"),
        #     gl.Curve(
        #         x_space,
        #         model.evaluate(
        #             x_grid, y_grid,
        #             params[0].amplitude.value, params[0].x_mean.value, params[0].y_mean.value,
        #             params[0].x_stddev.value, params[0].y_stddev.value, params[0].theta.value,
        #             params[1].slope_x.value, params[1].slope_y.value, params[1].intercept.value,
        #         )[y_val], label="Fitted profile along x"),
        # ]).show()

    star_positions["x_fit"] = fitted_x
    star_positions["y_fit"] = fitted_y
    star_positions["mean_fwhm"] = fitted_fwhm
    return star_positions

def detect_stars_from_dao_star_finder(image: np.ndarray, threshold_factor: float, fwhm_pixels: float) -> Table:
    """
    Detects stars in an image using the DAOStarFinder method. Code from sebvicens/SIGNALS_codes.

    Parameters
    ----------
    image : np.ndarray
        The image in which to detect stars. Deep frames are typically used for this purpose.
    threshold_factor : float
        Factor to calculate the detection threshold (threshold = factor * std).
    fwhm_pixels : float
        Full Width at Half Maximum (FWHM) of the stars in pixels.

    Returns
    -------
    Table
        Table of detected stars with their properties such as position, flux, and background.
    """
    # Calculate image statistics using sigma clipping
    std = float(sigma_clipped_stats(image, sigma=3.0)[2])
    threshold = threshold_factor * std
    star_finder = DAOStarFinder(threshold=threshold, fwhm=fwhm_pixels)
    star_positions = star_finder(image)
    return fit_star_position(image, star_positions)

def detect_stars_from_boxes(
    image: np.ndarray,
    boxes: Iterable[tuple[slice, slice]],
) -> Table:
    """
    Detects stars in an image using the maximum value in each specified box and fits a Gaussian model to the box.

    Parameters
    ----------
    image : np.ndarray
        The image in which to detect stars.
    boxes : Iterable[tuple[slice, slice]]
        Iterable of boxes defined by slices for the y and x axes. Each box is a tuple of two slices (y_slice, x_slice)
        in which a search for a single star is performed. The maximum value in each box is used as the star's position
        guess for the `fit_star_position` function.

    Returns
    -------
    Table
        Table of detected stars with the `xcentroid`, `ycentroid`, `x_fit`, `y_fit`, and `mean_fwhm` columns. The
        returned stars are given in the same order as the boxes.
    """
    star_positions = Table(names=["xcentroid", "ycentroid"], dtype=[float, float])
    for y_slice, x_slice in boxes:
        box = image[y_slice, x_slice]
        max_y, max_x = np.unravel_index(np.argmax(box), box.shape)
        star_positions.add_row([max_x + x_slice.start, max_y + y_slice.start])

    star_positions = fit_star_position(image, star_positions)
    return star_positions

def detect_stars_from_regions(
    image: Map,
    regions: pyregion.core.ShapeList
) -> Table:
    """
    Detects stars in an image using a region file and fits a Gaussian model to the detected peak in each region.

    Parameters
    ----------
    image : Map
        The Map containing the image in which to detect stars.
    regions : pyregion.core.ShapeList
        List of regions defined in the DS9 format.

    Returns
    -------
    Table
        Table of detected stars with the `xcentroid`, `ycentroid`, `x_fit`, `y_fit`, and `mean_fwhm` columns. The
        returned stars are given in the same order as the regions in the region file.
    """
    star_positions = Table(names=["xcentroid", "ycentroid"], dtype=[float, float])

    for region in regions:
        masked_map = image.get_masked_region(region)
        max_y, max_x = np.unravel_index(np.nanargmax(masked_map.data), masked_map.shape)
        star_positions.add_row([max_x, max_y])

    star_positions = fit_star_position(image.data, star_positions)
    return star_positions

def get_wcs_transformation(
    pixel_coords: np.ndarray,
    wcs_coords: Iterable[str],
    unit: tuple[u.Unit, u.Unit] = (u.hourangle, u.deg),
) -> WCS:
    """
    Gives WCS transformation based on pixel coordinates and their corresponding WCS coordinates.

    Parameters
    ----------
    pixel_coords : np.ndarray
        Array of pixel coordinates in the form [[x1, y1], [x2, y2], ...]. This should follow numpy's convention
        (0-indexed).
    wcs_coords : list[str]
        List of WCS coordinates as a one-dimensional list. This should be in the format that can be parsed by the
        `SkyCoord` class. For example, it can be a list of strings like ["12h30m00s +30d00m00s", ...].
    unit : tuple[u.Unit, u.Unit], default=(u.hourangle, u.deg)
        The units of the WCS coordinates. Default is (hourangle, deg) for right ascension and declination.

    Returns
    -------
    WCS
        The WCS object containing the transformation information.
    """
    fits_pixel_coords = pixel_coords + 1 # FITS convention (1-indexed)
    sky_coord = SkyCoord(wcs_coords, frame=ICRS, unit=unit)
    wcs = fit_wcs_from_points(
        xy=(fits_pixel_coords[:,0], fits_pixel_coords[:,1]),
        world_coords=sky_coord,
    )
    return wcs

def update_header_wcs(
    original_header: Header,
    wcs: WCS,
) -> Header:
    """
    Gives a new header with the WCS information from the given WCS object. All other non-WCS keywords are copied.

    Parameters
    ----------
    original_header : Header
        The original header from which to copy non-WCS keywords.
    wcs : WCS
        The WCS object containing the transformation information. This can be the object returned by the
        `get_wcs_transformation` function.

    Returns
    -------
    Header
        A copy of the original header with the WCS information updated.
    """
    wcs_header = wcs.to_header()
    new_header = original_header.copy()
    wcs_related_keywords = [
        "CRPIX1", "CRPIX2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2", "CUNIT1", "CUNIT2",
        "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "LONPOLE", "LATPOLE", "MJDREF", "RADESYS", "EQUINOX"
    ]
    for kw in wcs_related_keywords:
        if "PC" in kw:
            new_header.remove(f"CD{kw[-3:]}", ignore_missing=True)
        new_header[kw] = wcs_header[kw]
    return new_header

def get_detection_figure(
    image: np.ndarray,
    star_positions: Table,
) -> gl.SmartFigure:
    """
    Creates a figure showing the detected stars on the image.

    Parameters
    ----------
    image : np.ndarray
        The image in which the stars were detected.
    star_positions : Table
        Table of detected stars with their properties such as position, flux, and background.

    Returns
    -------
    gl.SmartFigure
        A figure showing the image with the detected stars marked.
    """
    max_real_val = np.nanmean(image) + 0.4 * np.nanstd(image)
    max_range = np.nanmax(image[image < max_real_val])

    hm = gl.Heatmap(image, origin_position="lower", color_map_range=(0, max_range), color_map="viridis")
    detections = [gl.Point(x, y, marker_style="+", label=str(i + 1), color="black")
                  for i, (x, y) in enumerate(zip(star_positions["x_fit"], star_positions["y_fit"]))]
    fig = gl.SmartFigure(elements=[hm, *detections]).set_rc_params({"legend.handletextpad" : -1})
    fig.set_custom_legend([gl.LegendMarker("Detections", marker_style="+", edge_color="black", marker_size=10)])
    return fig
