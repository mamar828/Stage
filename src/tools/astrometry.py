from astropy.table import Table
import numpy as np

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
    for star in star_positions:
        x = star['xcentroid']
        y = star['ycentroid']
        box_apothem = 3
        x_min = int(max(x - box_apothem, 0))
        x_max = int(min(x + box_apothem + 1, image.shape[1]))
        y_min = int(max(y - box_apothem, 0))
        y_max = int(min(y + box_apothem + 1, image.shape[0]))
        box = image[y_min:y_max, x_min:x_max]
        y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
        model = models.Gaussian2D(amplitude=box.max(), x_mean=x, y_mean=y, x_stddev=5, y_stddev=5,
                                  bounds={"x_mean": (x_min, x_max), "y_mean": (y_min, y_max)})
        fit_p = fitting.LevMarLSQFitter()
        try:
            params = fit_p(model, x_grid, y_grid, box, maxiter=100000)
            fitted_x.append(params.x_mean.value)
            fitted_y.append(params.y_mean.value)
        except Exception:
            fitted_x.append(x)
            fitted_y.append(y)

    star_positions['x_fit'] = fitted_x
    star_positions['y_fit'] = fitted_y
    return star_positions
