import numpy as np
import src.graphinglib as gl

from src.hdu.maps.map import Map
from src.hdu.cubes.cube import Cube
from src.tools.deconvolution import deconvolve_cube, get_deconvolution_error


# nii_1_cube = Cube.load("data/orion/linewidths/deconvolution_binned/nii_1_deconvolved.fits")
# sii_1_cube = Cube.load("data/orion/linewidths/deconvolution_binned/sii_1_line_1_deconvolved.fits")

# nii_1_fwhm_channels = Map.load("data/orion/linewidths/maps/nii_1_stddev.fits") * 2 * np.sqrt(2 * np.log(2))
# sii_1_fwhm_channels = Map.load("data/orion/linewidths/maps/sii_1_line_1_stddev.fits") * 2 * np.sqrt(2 * np.log(2))

# nii_1_fwhm_AA = nii_1_fwhm_channels * nii_1_cube.header["XIL"] / 47
# sii_1_fwhm_AA = sii_1_fwhm_channels * sii_1_cube.header["XIL"] / 47

# temperature = 4.73 * 10**4 * (nii_1_fwhm_AA.get_reprojection_on(sii_1_fwhm_AA.header)**2 - sii_1_fwhm_AA**2)

# temperature_plot = temperature.data.plot
# temperature_plot.color_map_range = (0, 1000)
# gl.SmartFigure(elements=[temperature_plot]).show()


# TEST WITH A SINGLE DECONVOLVED SPECTRUM
# ---------------------------------------
nii_global_spec = np.loadtxt("data/orion/linewidths/global_spectrum_tests/nii_2_global_spectrum.dat")
nii_cdelt_AA = 3.4076504661393465 / 48
ha_global_spec = np.loadtxt("data/orion/linewidths/global_spectrum_tests/ha_2_global_spectrum.dat")
ha_cdelt_AA = 3.3863680690151563 / 48

calib_global_spec = np.loadtxt("data/orion/linewidths/global_spectrum_tests/global_calibration.dat")
# random_array = np.random.normal(
#     np.mean(ha_global_spec[:10, 1], axis=0),
#     np.std(ha_global_spec[:10, 1], axis=0) / 10,
#     24,
# )
# ha_global_spec[24:, 1] = random_array
# ha_global_spec[:24, 1] = random_array

calib_centroids = 7.5
n_iterations = 100

deconvolved = []
figs = []
for line_name, global_spec in zip(["nii_2", "ha_2"], [nii_global_spec, ha_global_spec]):
    deconvolved_data, offsetted_data, offsetted_lsf = deconvolve_cube(
        global_spec[:, 1][:, None, None],
        calib_global_spec[:, 1][:, None, None],
        calib_centroids,
        n_iterations,
    )
    deconvolution_error, reconvolved = get_deconvolution_error(offsetted_data, offsetted_lsf, deconvolved_data)
    print(f"Deconvolution error for {line_name}: {np.mean(deconvolution_error)}")

    x_vals = np.arange(48) + 1
    figs.append(gl.SmartFigure(
        title=f"Deconvolution for {line_name}",
        elements=[
            gl.Curve(x_vals, offsetted_data[:,0,0], label="Offsetted Data"),
            gl.Curve(x_vals, offsetted_lsf[:,0,0], label="Offsetted LSF"),
            gl.Curve(x_vals, deconvolved_data[:,0,0], label="Deconvolved Data"),
            gl.Curve(x_vals, reconvolved[:,0,0], label="Reconvolved Data"),
        ],
    ))

    deconvolved.append(deconvolved_data)

x_vals = np.arange(48) + 1
nii_deconv_curve = gl.Curve(x_vals, deconvolved[0][:, 0, 0], label="NII Deconvolved Spectrum")
ha_deconv_curve = gl.Curve(x_vals, deconvolved[1][:, 0, 0], label="ha Deconvolved Spectrum")

nii_fit = gl.FitFromGaussian(nii_deconv_curve, guesses=[1, 24, 1])
ha_fit = gl.FitFromGaussian(ha_deconv_curve, guesses=[1, 24, 1])

nii_std = nii_fit.parameters[2]
ha_std = ha_fit.parameters[2]


figs.append(gl.SmartFigure(
    num_rows=2,
    elements=[[nii_deconv_curve, nii_fit], [ha_deconv_curve, ha_fit]],
    subtitles=[
        rf"$\sigma={nii_std:.2f}$ channels, FWHM$={nii_std*2*np.sqrt(2*np.log(2)) * nii_cdelt_AA:.4f}\AA$",
        rf"$\sigma={ha_std:.2f}$ channels, FWHM$={ha_std*2*np.sqrt(2*np.log(2)) * ha_cdelt_AA:.4f}\AA$",
    ],
))

global_fig = gl.SmartFigure(
    2,
    2,
    elements=figs[:2],
    height_ratios=(2, 3)
)
global_fig[1, :] = figs[2]
global_fig.show()
