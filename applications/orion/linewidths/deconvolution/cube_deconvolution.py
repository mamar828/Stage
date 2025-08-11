from src.coordinates.ds9_coords import DS9Coords
from src.tools.deconvolution import *
from src.hdu.cubes.cube import Cube
from src.hdu.maps.map import Map


# CUBE PREPPING TESTS
# -------------------
coords = DS9Coords(91, 37)
data_cube = Cube.load("data/orion/data_cubes/sii_2.fits").bin((1,4,4))[:,*coords].data[:,None,None]
calib_cube = Cube.load("data/orion/calibration/calibration_binned.fits").bin((1,4,4))[:,*coords].data[:,None,None]
calib_centroids = Map.load("data/orion/calibration/calibration_centroids.fits").bin((4,4))[*coords]

# FOR SII ONLY
random_array = np.random.normal(
    np.mean(data_cube[38:], axis=0),
    np.std(data_cube[38:], axis=0),
    (24, *data_cube.shape[1:]),
)

data_cube[24:, :, :] = random_array

n_iterations = 100

deconvolved_data, offsetted_data, offsetted_lsf = deconvolve_cube(
    data_cube,
    calib_cube,
    calib_centroids,
    n_iterations,
)

deconvolution_error, reconvolved_data = get_deconvolution_error(offsetted_data, offsetted_lsf, deconvolved_data)
print(np.nanmean(deconvolution_error))

x_vals = np.arange(data_cube.shape[0]) + 1
gl.SmartFigure(
    elements=[
        gl.Curve(x_vals, offsetted_data[:,0,0], label="Offsetted Data"),
        gl.Curve(x_vals, offsetted_lsf[:,0,0], label="Offsetted LSF"),
        gl.Curve(x_vals, deconvolved_data[:,0,0], label="Deconvolved Data"),
        gl.Curve(x_vals, reconvolved_data[:,0,0], label="Reconvolved Data"),
    ]
).show()


# # Map(deconvolution_error).save("data/orion/linewidths/deconvolution/sii_1_deconvolution_error.fits")
# # Cube(deconvolved_data).save("data/orion/linewidths/deconvolution/sii_1_deconvolved.fits")


# DECONVOLUTION PROCESS
# ---------------------
calib_cube = Cube.load("data/orion/calibration/calibration_binned.fits")
calib_centroids = Map.load("data/orion/calibration/calibration_centroids.fits")
n_iterations = 100

def deconvolve_process(data_cube: Cube, line_name: str) -> None:
    if line_name.split("_")[1] == "1":
        calib_cube = Cube.load("data/orion/calibration/calibration_binned.fits").bin((1, 3, 3))
        calib_centroids = Map.load("data/orion/calibration/calibration_centroids.fits").bin((3, 3))
    else:
        calib_cube = Cube.load("data/orion/calibration/calibration_binned.fits").bin((1, 4, 4))
        calib_centroids = Map.load("data/orion/calibration/calibration_centroids.fits").bin((4, 4))

    deconvolved_data, offsetted_data, offsetted_lsf = deconvolve_cube(
        data_cube,
        calib_cube,
        calib_centroids,
        n_iterations,
    )

    # x_vals = np.arange(data_cube.shape[0]) + 1
    # gl.SmartFigure(
    #     elements=[
    #         gl.Curve(x_vals, offsetted_data[:,0,0], label="Offsetted Data"),
    #         gl.Curve(x_vals, offsetted_lsf[:,0,0], label="Offsetted LSF"),
    #         gl.Curve(x_vals, deconvolved_data[:,0,0], label="Deconvolved Data"),
    #     ],
    # ).show()

    deconv_cube = Cube(deconvolved_data, data_cube.header)
    deconvolution_error, _ = get_deconvolution_error(offsetted_data, offsetted_lsf, deconvolved_data)
    deconv_error_map = Map(deconvolution_error, header=data_cube.header.spectral)

    deconv_cube.save(f"data/orion/linewidths/deconvolution_binned/{line_name}_deconvolved.fits")
    deconv_error_map.save(f"data/orion/linewidths/deconvolution_binned/{line_name}_deconvolution_error.fits")


# # All data cubes except SII have a single line, so they are treated the same way.
# lines = ["nii_1", "nii_2", "ha_1", "ha_2", "oiii_1", "oiii_2"]
# for line in lines:
#     # deconvolve_process(Cube.load(f"data/orion/data_cubes/{line}.fits"), line)
#     bin_ = "3x3" if line.endswith("1") else "4x4"
#     deconvolve_process(Cube.load(f"data/orion/data_cubes/binned/{line}_binned_{bin_}.fits"), line)

# # SII has two lines, so it needs special treatment to avoid artifacts from the second line
# for line in ["sii_1", "sii_2"]:
#     # data_cube = Cube.load(f"data/orion/data_cubes/{line}.fits")
#     bin_ = "3x3" if line.endswith("1") else "4x4"
#     data_cube = Cube.load(f"data/orion/data_cubes/binned/{line}_binned_{bin_}.fits")
#     for i, (random_slice, data_slice) in enumerate(
#         [(slice(38, None), slice(24, None)), (slice(None, 10), slice(None, 24))],
#         start=1,
#     ):
#         # random_slice gives the slices sampled for filling randomly the data_slice channels of the data cube
#         current_data_cube = data_cube.copy()
#         random_fill_array = np.random.normal(
#             np.mean(current_data_cube.data[random_slice, :, :], axis=0),
#             np.std(current_data_cube.data[random_slice, :, :], axis=0),
#             (24, *current_data_cube.shape[1:]),
#         )
#         current_data_cube.data[data_slice, :, :] = random_fill_array

#         deconvolve_process(data_cube, f"{line}_line_{i}")
