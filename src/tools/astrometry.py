from astropy.table import Table
import numpy as np
import src.graphinglib as gl

from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.modeling import models, fitting


def detect_stars(image: np.ndarray, threshold_factor: float, fwhm_pixels: float) -> Table:
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
    stars : Table
        Table of detected stars with their properties such as position, flux, and background.
    """
    # Calculate image statistics using sigma clipping
    std = float(sigma_clipped_stats(image, sigma=3.0)[2])
    threshold = threshold_factor * std
    star_finder = DAOStarFinder(threshold=threshold, fwhm=fwhm_pixels)
    star_positions = star_finder(image)
    return fit_star_position(image, star_positions)

def fit_star_position(image: np.ndarray, star_positions: Table) -> Table:
    """
    Fits the star positions using a two-dimensional Gaussian model.

    Parameters
    ----------
    star_positions : Table
        Table of detected stars with their properties such as position, flux, and background.
    image : np.ndarray
        The image in which the stars were detected.

    Returns
    -------
    fitted_positions : Table
        Table of fitted star positions with their properties.
    """
    fitted_x = []
    fitted_y = []
    fitted_fwhm = []
    for star in star_positions:
        x = star["xcentroid"]
        y = star["ycentroid"]
        box_apothem = 10
        x_min = int(max(x - box_apothem, 0))
        x_max = int(min(x + box_apothem + 1, image.shape[1]))
        y_min = int(max(y - box_apothem, 0))
        y_max = int(min(y + box_apothem + 1, image.shape[0]))
        box = image[y_min:y_max, x_min:x_max]
        y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
        model = (
            models.Gaussian2D(amplitude=box.max(), x_mean=x, y_mean=y, x_stddev=5, y_stddev=5, theta=0,
                              bounds={"x_mean": (x_min, x_max), "y_mean": (y_min, y_max)})
            + models.Planar2D(0, 0, 0)
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
        #     gl.Curve(x_space := min(np.arange(box_apothem*2 + 1) + x_min),
        #              box[y_val := min(round(params[0].y_mean.value - y_min), box_apothem*2)], label="Profile along x"),
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
