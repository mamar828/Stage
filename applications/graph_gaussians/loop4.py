import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import shutil

from src.hdu.cubes.cube import Cube


array = np.loadtxt("data/graph_gaussians/LOOP4_cube_gauss_run_0.dat")
column_density_array = np.log10(1.822e18 * array[:,2] * array[:,4] / 1e19)

fig, ax = plt.subplots(1, figsize=(10,7))
plt.rcParams["font.size"] = 14
plt.rcParams["font.serif"] = "Computer Modern"

# Data
header = Cube.load("summer_2023/HI_regions/data_cubes/LOOP4/LOOP4_FINAL_GLS.fits").header
convert_func = np.vectorize(header.get_value)

sc = ax.scatter(
    x=convert_func(array[:,3]) / 1000,      # Convert: channel -> m/s -> km/s
    y=array[:,4]*np.abs(header["CDELT3"] / 1000),
    s=0.05,
    c=column_density_array,
    cmap=mpl.cm.inferno,
    marker=",",
    edgecolor="none",
    vmin=-1,
    vmax=1.25
)

# Lines
plt.plot([0, 20], [5.4, 5.4], linestyle="--", color="blue", linewidth=1.5)
plt.plot([0, 20], [2.3, 2.3], linestyle="--", color="red", linewidth=1.5)


cbar = fig.colorbar(sc, cax=ax.inset_axes([1.01, 0, .03, 1]))

cbar.set_label("$\log_{10}(N_{HI}\ [10^{19}\mathrm{\ cm}^{-2}])$")
ax.set_xlabel("$\mu$ [km s$^{-1}$]", fontsize=plt.rcParams["font.size"])
ax.set_ylabel("$\sigma$ [km s$^{-1}$]", fontsize=plt.rcParams["font.size"])

ax.set_xlim(4, 11)
ax.set_ylim(0, 9)

plt.tick_params(axis="both", direction="in", labelsize=plt.rcParams["font.size"])
# shutil.copyfile("figures/graph_gaussians/loop4.png", "figures/graph_gaussians/loop4_old.png")
plt.savefig("figures/graph_gaussians/loop4_2.png", bbox_inches="tight", dpi=600)
# plt.show()
