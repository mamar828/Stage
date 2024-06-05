import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import shutil


arr = np.loadtxt("data/graph_gaussians/LOOP4_cube_gauss_run_0.dat")
arr = arr.reshape((arr.shape[0] // 3, 3, 5))
# arr is now of shape : (number_of pixels, number_of_gaussians, number_of_parameters)

fig, ax = plt.subplots(1, figsize=(10,7))
plt.rcParams["font.size"] = 14
plt.rcParams["font.sans-serif"] = "Times New Roman"

cmap = mpl.cm.inferno
norm = mpl.colors.Normalize(vmin=1, vmax=4)
cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax.inset_axes([1.01, 0, .03, 1]))

cbar.set_label("Gaussian function amplitude")
ax.set_xlabel("$\mu$ [km s$^{-1}$]", fontsize=plt.rcParams["font.size"])
ax.set_ylabel("$\sigma$ [km s$^{-1}$]", fontsize=plt.rcParams["font.size"])

for i in range(3):
    ax.scatter(
        x=arr[:,i,3],
        y=arr[:,i,4],
        s=0.05,
        c=arr[:,i,2],
        cmap=cmap,
        marker=",",
        edgecolor="none"
    )

ax.set_xlim(115, 125)
ax.set_ylim(0, 12)

plt.tick_params(axis="both", direction="in", labelsize=plt.rcParams["font.size"])
shutil.copyfile("figures/graph_gaussians/loop4.png", "figures/graph_gaussians/loop4_old.png")
plt.savefig("figures/graph_gaussians/loop4.png", bbox_inches="tight", dpi=600)
# plt.show()
