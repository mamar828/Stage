import numpy as np
import os
import src.graphinglib as gl
import astropy.units as u
from astropy.wcs.utils import fit_wcs_from_points
from astropy.coordinates import SkyCoord, FK5
from astropy.table import Table

from src.hdu.cubes.cube import Cube
from src.hdu.maps.map import Map
from src.tools.astrometry import *


# Create the deep frames for the Orion data cubes
# -----------------------------------------------
# for file in os.listdir("data/orion/data_cubes"):
#     if "_" in file and file.endswith(".fits"):
#         Cube.load(f"data/orion/data_cubes/{file}").get_deep_frame().save(
#             f"data/orion/deep_frames/{file.split(".")[0]}_df.fits"
#         )

# Detect stars in the deep frame
# ------------------------------
# deep_frame = Map.load("data/orion/deep_frames/ha_2_df.fits")
# deep_frame = Cube.load("data/orion/data_cubes/ha_2.fits")[16,:,:]
# old_threshold_factor = 1.95
# old_fwhm_pixels = 8
# threshold_factor = 2
# fwhm_pixels = 10

# # old_detections = detect_stars(deep_frame.data, old_threshold_factor, old_fwhm_pixels)
# star_detections = detect_stars(deep_frame.data, threshold_factor, fwhm_pixels)
# fig = gl.SmartFigureWCS(
#     projection=deep_frame.header.wcs,
#     elements=[
#         (df := deep_frame.data.plot),
#         # gl.Scatter(
#         #     old_detections["xcentroid"],
#         #     old_detections["ycentroid"],
#         #     marker_style="+",
#         #     face_color="red",
#         #     label="Old Detections",
#         # ),
#         # gl.Scatter(
#         #     star_detections["x_fit"],
#         #     star_detections["y_fit"],
#         #     marker_style="*",
#         #     face_color="red",
#         #     label="Detected Stars (fit)",
#         # ),
#         gl.Scatter(
#             star_detections["xcentroid"],
#             star_detections["ycentroid"],
#             marker_style="x",
#             face_color="black",
#             label="Detected Stars",
#         ),
#     ],
#     # title=f"Old detections: {len(old_detections)}, New detections: {len(star_detections)}",
#     # size=(13, 12),
# )
# df.color_map_range = 3, 18
# # fig.save("figures/orion/astrometry/star_detections.pdf")
# fig.show()#fullscreen=True)


parameters = {
    "sii_1_df.fits": {"threshold_factor": 1.3, "fwhm_pixels": 6.5,
                      "pixel_i": [0, 1, 2, 3, 5, 4, 8, 6, 9, 7, 10, 11, 14, 15, 13, 12, 17], "wcs_i": np.arange(17)},
    "sii_2_df.fits": {"threshold_factor": 2, "fwhm_pixels": 6,
                      "pixel_i": [2, 3, 4, 6, 5, 1, 0], "wcs_i": [0, 1, 2, 3, 4, 5, 6]},
    "ha_1_df.fits": {"threshold_factor": 1.35, "fwhm_pixels": 10,
                     "pixel_i": [0, 1, 3, 4, 5, 6, 7], "wcs_i": [0, 1, 3, 5, 7, 8, 11]},
    "ha_2_df.fits": {"threshold_factor": 2, "fwhm_pixels": 10,
                     "pixel_i": [9, 14, 16, 7], "wcs_i": [0, 1, 2, 5]},
    "nii_1_df.fits": {"threshold_factor": 0.69, "fwhm_pixels": 10.5,
                      "pixel_i": [0, 1, 2, 3, 4, 5, 8], "wcs_i": [0, 1, 3, 5, 7, 8, 13]},
    "nii_2_df.fits": {"threshold_factor": 1.25, "fwhm_pixels": 10,
                      "pixel_i": [11, 14, 15, 16, 1, 0], "wcs_i": [0, 1, 2, 4, 5, 6]},
    "oiii_1_df.fits": {"threshold_factor": 1.9, "fwhm_pixels": 8,
                       "pixel_i": [0, 1, 3, 4, 5, 6, 8, 7], "wcs_i": [0, 1, 3, 5, 7, 8, 13, 14]},
}


# Link the star positions to their coordinates
# --------------------------------------------
# FILENAME = "ha_2_df.fits"

# deep_frame = Map.load(f"data/orion/astrometry/deep_frames_old/{FILENAME}")
# star_detections = detect_stars(
#     deep_frame.data, # Cube.load("data/orion/astrometry/data_cubes_old/ha_2.fits")[16,:,:].data, # only for the ha_2 field
#     parameters[FILENAME]["threshold_factor"],
#     parameters[FILENAME]["fwhm_pixels"],
# )

# # Create a slice for ordering the detections
# pixel_i = parameters[FILENAME]["pixel_i"]
# wcs_i = parameters[FILENAME]["wcs_i"]
# pixel_coords = np.column_stack((star_detections["x_fit"], star_detections["y_fit"]))[pixel_i]
# centroid_coords = np.column_stack((star_detections["xcentroid"], star_detections["ycentroid"]))[pixel_i]
# fig = gl.SmartFigure(
#     elements=[
#         deep_frame.data.plot,
#         *[gl.Point(x, y, label=str(i), marker_style="x", color="red") for i, (x, y) in enumerate(pixel_coords)],
#         *[gl.Point(x, y, label=str(i)) for i, (x, y) in enumerate(centroid_coords)],
#     ],
#     aspect_ratio="equal"
# )
# fig[0][0].color_map_range = 200, 900
# fig.show()

# lines = open(f"data/orion/astrometry/reference/list_star_ohp_field_{FILENAME.split('_')[1]}.txt", "r").readlines()
# wcs_coords = [line.split(",")[0] for line in lines]
# wcs_coords = [c for i, c in enumerate(wcs_coords) if (wcs_i == slice(None, None)) or (i in wcs_i)]

# pixel_coords += 1 # FITS convention (1-indexed)
# sky_coord = SkyCoord(wcs_coords, frame=FK5, unit=(u.hourangle, u.deg))
# new_header = fit_wcs_from_points(
#     xy=(pixel_coords[:,0], pixel_coords[:,1]),
#     world_coords=sky_coord,
# ).to_header()
# deep_frame_aligned = deep_frame.copy()
# deep_frame_aligned.header = new_header
# deep_frame_aligned.save(f"data/orion/astrometry/deep_frames_aligned_old/{FILENAME}", overwrite=False)


# sitelle = Map.load("data/orion/astrometry/deep_frames_aligned_old/Orion-A_SN3.merged.cm1.1.0.deep_frame.fits")
# nii = Map.load("data/orion/astrometry/deep_frames_aligned_old/ha_2_df.fits")
# fig = gl.SmartFigure(
#     elements=[
#         nii.data.plot,
#         sitelle.get_reprojection_on(nii.header).data.plot
#     ]
# )
# fig[0][0].color_map = "grey"
# fig[0][0].color_map_range = 400, 1200
# fig[0][0].show_color_bar = False

# fig[0][1].color_map = "autumn"
# fig[0][1].color_map_range = 3e5, 2e6
# fig[0][1].show_color_bar = False
# fig[0][1].alpha_value = 0.2
# fig.show()


# Apply the new WCS to the data cubes (deprecated)
# -----------------------------------
# for file in os.listdir("data/orion/astrometry/deep_frames_aligned_old"):
#     if file.endswith(".fits") and not "SN3" in file:
#         deep_frame = Map.load(f"data/orion/astrometry/deep_frames_aligned_old/{file}")
#         cube = Cube.load(f"data/orion/astrometry/data_cubes_old/{file.split("_df")[0]}.fits")
#         for kw in ["CRPIX1", "CRPIX2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CRVAL1", "CRVAL2"]:
#             if "PC" in kw:
#                 cube.header.remove(f"CD{kw[-3:]}", ignore_missing=True)
#             cube.header[kw] = deep_frame.header[kw]
#         cube.save(f"data/orion/astrometry/data_cubes_aligned_old/{file.split("_df")[0]}.fits", overwrite=True)



# ================================================
# NEW DATA (OIII field 2 + all non-filtered field)
# ================================================
# Cube.load("data/orion/astrometry/data_cubes_new/oiii_2.fits") \
#     .get_deep_frame() \
#     .save("data/orion/astrometry/oiii_2_df_new.fits")

# deep_frame = Map.load("data/orion/astrometry/oiii_2_df_new.fits")
# threshold_factor = 1.2
# fwhm_pixels = 8
parameters["oiii_2_df.fits"] = {"threshold_factor": 1.2, "fwhm_pixels": 8,
                                "pixel_i": [33, 35, 1], "wcs_i": [2, 4, 5, 6, 1, 0]}

# star_detections = detect_stars(
#     deep_frame.data,
#     threshold_factor,
#     fwhm_pixels,
# )
# other_stars = fit_star_position(deep_frame.data, Table(data=[[88, 249, 450], [328, 416, 84]],
#                                                       names=["xcentroid", "ycentroid"]))
# manual_stars = np.column_stack((other_stars["x_fit"].value, other_stars["y_fit"].value))

# # Create a slice for ordering the detections
# pixel_i = parameters["oiii_2_df.fits"]["pixel_i"]
# wcs_i = parameters["oiii_2_df.fits"]["wcs_i"]
# pixel_coords = np.column_stack((star_detections["x_fit"], star_detections["y_fit"]))[pixel_i]
# pixel_coords = np.vstack((manual_stars[:2], pixel_coords, manual_stars[2:]))
# fig = gl.SmartFigure(
#     elements=[
#         deep_frame.data.plot,
#         *[gl.Point(x, y, label=str(i), marker_style="x", color="red") for i, (x, y) in enumerate(pixel_coords)],
#     ],
#     aspect_ratio="equal"
# )
# fig[0][0].color_map_range = 50, 200
# fig.show()

# lines = open(f"data/orion/astrometry/reference/list_star_ohp_field_2.txt", "r").readlines()
# wcs_coords = [line.split(",")[0] for line in lines]
# wcs_coords = [c for i, c in enumerate(wcs_coords) if (wcs_i == slice(None, None)) or (i in wcs_i)]

# pixel_coords += 1 # FITS convention (1-indexed)
# sky_coord = SkyCoord(wcs_coords, frame=FK5, unit=(u.hourangle, u.deg))
# new_header = fit_wcs_from_points(
#     xy=(pixel_coords[:,0], pixel_coords[:,1]),
#     world_coords=sky_coord,
# ).to_header()
# deep_frame_aligned = deep_frame.copy()
# deep_frame_aligned.header = new_header
# deep_frame_aligned.save(f"data/orion/astrometry/oiii_2_df_aligned.fits", overwrite=False)

# Copy all other data cubes with their aligned WCS
# ------------------------------------------------
# for file in os.listdir("data/orion/astrometry/data_cubes_new"):
#     if file.endswith(".fits"):
#         if file == "oiii_2.fits":
#             deep_frame = Map.load(f"data/orion/astrometry/oiii_2_df_aligned.fits")
#         else:
#             deep_frame = Map.load(f"data/orion/astrometry/deep_frames_aligned_old/{file[:-5]}_df.fits")

#         cube = Cube.load(f"data/orion/astrometry/data_cubes_new/{file}")
#         for kw in ["CRPIX1", "CRPIX2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CRVAL1", "CRVAL2"]:
#             if "PC" in kw:
#                 cube.header.remove(f"CD{kw[-3:]}", ignore_missing=True)
#             cube.header[kw] = deep_frame.header[kw]
#         cube.header["CDELT3"] = cube.header["CHAN_KMS"]
#         cube.save(f"data/orion/data_cubes/{file}", overwrite=True)

# Calculate the seeing for each field
# -----------------------------------
# seeing_stars = {
#     "sii_1_df.fits": [0, 1, 5, 6, 7, 8],
#     "sii_2_df.fits": [1, 2, 5],
#     "ha_1_df.fits": [0, 1, 3],
#     "ha_2_df.fits": [2],
#     "nii_1_df.fits": [0, 1, 3, 4],
#     "nii_2_df.fits": [1, 2, 4],
#     "oiii_1_df.fits": [0, 1, 3, 4],
#     "oiii_2_df.fits": [0, 1, 2],
# }
# f = open("data/orion/astrometry/seeing.csv", "w")
# f.write("field,seeing,individual FWHMs\n")
# for file in os.listdir("data/orion/astrometry/deep_frames_aligned_old") + ["oiii_2_df.fits"]:
#     if file.endswith(".fits"):
#         if file == "oiii_2_df.fits":
#             image = Map.load(f"data/orion/astrometry/{file[:-5]}_aligned.fits")
#         elif file.startswith("ha_2"):
#             # Use the ha_2 field for the ha_2_df.fits deep frame
#             image = Cube.load("data/orion/astrometry/data_cubes_old/ha_2.fits")[16, :, :]
#         else:
#             image = Map.load(f"data/orion/astrometry/deep_frames_aligned_old/{file}")

#         star_detections = detect_stars(
#             image.data,
#             parameters[file]["threshold_factor"],
#             parameters[file]["fwhm_pixels"],
#         )
#         fwhms = star_detections["mean_fwhm"][parameters[file]["pixel_i"]][seeing_stars[file]]
#         f.write(f"{file.split('_df')[0]:>6},{np.mean(fwhms):>5.2f},{" ".join([f"{f:.2f}" for f in fwhms])}\n")
# f.close()

# Binning according to the seeing
# -------------------------------
# for file in os.listdir("data/orion/data_cubes"):
#     if file.endswith(".fits"):
#         if "1" in file:
#             binning = 3
#         else:
#             binning = 4
#         Cube.load(f"data/orion/data_cubes/{file}").bin((1, binning, binning)).save(
#             f"data/orion/data_cubes/binned/{file.split('.fits')[0]}_binned_{binning}x{binning}.fits",
#             overwrite=True
#         )
