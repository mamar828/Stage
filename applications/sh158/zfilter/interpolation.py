import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pyperclip


coefficients = np.array([
    [      0.0,-0.001256,-0.002512,-0.003558,-0.003977,-0.003977,-0.003767,-0.003977,-0.003977,-0.003558,-0.002512,-0.001256,      0.0],
    [-0.001256, -0.00293,-0.003139,-0.003767,-0.003558,-0.002512,-0.002093,-0.002512,-0.003558,-0.003767,-0.003139, -0.00293,-0.001256],
    [-0.002512,-0.003139,-0.003349,-0.002512,-0.001047, 0.003349, 0.005023, 0.003349,-0.001047,-0.002512,-0.003349,-0.003139,-0.002512],
    [-0.003558,-0.003767,-0.002512, 0.001256, 0.010883, 0.018418, 0.022189, 0.018418, 0.010883, 0.001256,-0.002512,-0.003767,-0.003558],
    [-0.003977,-0.003558,-0.001047, 0.010883,  0.02365, 0.032231, 0.036836, 0.032231,  0.02365, 0.010883,-0.001047,-0.003558,-0.003977],
    [-0.003977,-0.002512, 0.003349, 0.018418, 0.032231, 0.043533, 0.049812, 0.043533, 0.032231, 0.018418, 0.003349,-0.002512,-0.003977],
    [-0.003767,-0.002093, 0.005023, 0.022189, 0.036836, 0.049812, 0.054835, 0.049812, 0.036836, 0.022189, 0.005023,-0.002093,-0.003767],
    [-0.003977,-0.002512, 0.003349, 0.018418, 0.032231, 0.043533, 0.049812, 0.043533, 0.032231, 0.018418, 0.003349,-0.002512,-0.003977],
    [-0.003977,-0.003558,-0.001047, 0.010883,  0.02365, 0.032231, 0.036836, 0.032231,  0.02365, 0.010883,-0.001047,-0.003558,-0.003977],
    [-0.003558,-0.003767,-0.002512, 0.001256, 0.010883, 0.018418, 0.022189, 0.018418, 0.010883, 0.001256,-0.002512,-0.003767,-0.003558],
    [-0.002512,-0.003139,-0.003349,-0.002512,-0.001047, 0.003349, 0.005023, 0.003349,-0.001047,-0.002512,-0.003349,-0.003139,-0.002512],
    [-0.001256, -0.00293,-0.003139,-0.003767,-0.003558,-0.002512,-0.002093,-0.002512,-0.003558,-0.003767,-0.003139, -0.00293,-0.001256],
    [      0.0,-0.001256,-0.002512,-0.003558,-0.003977,-0.003977,-0.003767,-0.003977,-0.003977,-0.003558,-0.002512,-0.001256,      0.0],
])

# pyperclip.copy(str(list(zip(range(-6,7), np.diag(coefficients).tolist()))))

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
linspace = np.linspace(-6,6,13)
X, Y = np.meshgrid(linspace, linspace)

# Plot the surface.
# surf_1 = ax.plot_surface(X, Y, coefficients, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(0, 0.06)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf_1, shrink=0.5, aspect=5)

# plt.show()

filter_13x13 = np.stack((X, Y, coefficients), axis=0)

huge_coefficients = np.array([
    [-530, -535, -547, -560, -565, -561, -546, -526, -500, -468, -423, -360, -277, -181, -90, -24, 0],
    [-988, -999, -1025, -1053, -1067, -1059, -1028, -984, -933, -874, -795, -685, -540, -372, -210, -92, -24],
    [-766, -781, -817, -857, -881, -878, -850, -807, -759, -713, -661, -594, -505, -397, -290, -210, -90],
    [-395, -414, -463, -524, -574, -600, -597, -575, -545, -519, -498, -480, -458, -429, -397, -372, -181],
    [124,  100,  34,  -59, -159, -248, -312, -347, -359, -359, -363, -380, -412, -458, -505, -540, -277],
    [771,  742, 656, 519, 344, 154, -22, -154, -234, -269, -285, -314, -380, -480, -594, -685, -360],
    [1514, 1478, 1367, 1174, 904, 592, 279, 20,  -148, -227, -255, -285, -363, -498, -661, -795, -423],
    [2310, 2264, 2107, 1864, 1500, 1061, 611, 229, -37, -173, -227, -299, -359, -545, -713, -874, -468],
    [3119, 3060, 2875, 2556, 2107, 1569, 1016, 531, 177, -37, -148, -234, -359, -545, -759, -933, -500],
    [3916, 3840, 3610, 3232, 2718, 2122, 1510, 961, 531, 229, 20,  -154, -347, -575, -807, -984, -526],
    [4687, 4595, 4324, 3894, 3341, 2719, 2090, 1510, 1016, 611, 279, -22, -312, -597, -850, -1028, -546],
    [5428, 5322, 5016, 4546, 3973, 3347, 2719, 2122, 1569, 1061, 592, 154, -22, -600, -878, -1059, -561],
    [6126, 6010, 5682, 5191, 4602, 3973, 3341, 2718, 2107, 1500, 904, 344, -159, -574, -881, -1067, -565],
    [6752, 6630, 6294, 5791, 5191, 4546, 3894, 3326, 2556, 1864, 1174, 519, -59, -524, -857, -1053, -560],
    [7259, 7138, 6807, 6294, 5682, 5016, 4324, 3718, 2875, 2107, 1367, 656, 34,  -463, -817, -1025, -547],
    [7593, 7475, 7138, 6807, 6126, 5322, 4595, 3840, 3060, 2264, 1478, 742, 100, -414, -781, -999, -535],
    [7709, 7593, 7259, 6752, 6126, 5428, 4687, 3916, 3119, 2310, 1514, 771, 124, -395, -766, -988, -530],
])

tabs = []
for i, vals in enumerate(huge_coefficients):
    tabs += [list(reversed(vals)) + vals[1:].tolist()]
for tab in reversed(tabs[:-1]):
    tabs += [tab]
tabs = np.array(tabs)*1e-6

# Save the complete 33x33 filter
# np.savetxt("src/tools/zurflueh_filter/cpp_lib/coefficients_33x33.csv", tabs, fmt="%.6f",delimiter=",")

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plot_surf = lambda arr: ax.plot_surface(arr[0], arr[1], arr[2], cmap=cm.coolwarm, linewidth=0, antialiased=False)
linspace = np.linspace(-16,16,33)
X, Y = np.meshgrid(linspace, linspace)
filter_33x33 = np.stack((X, Y, tabs), axis=0)

# Plot the surface.
# surf_1 = plot_surf(filter_13x13)
surf_2 = plot_surf(filter_33x33)

# Customize the z axis.
ax.set_zlim(0, 0.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.03f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf_1, shrink=0.5, aspect=5)
# fig.colorbar(surf_2, shrink=0.5, aspect=5)

# plt.show()

interp = RegularGridInterpolator((linspace, linspace), filter_33x33[2])
new_linspace = np.linspace(-6,6,13)
xx, yy = np.meshgrid(new_linspace, new_linspace)
new_13x13 = interp(np.array(np.stack((xx, yy), axis=-1)) * 16 / 6)

plot_surf(np.stack((xx, yy, new_13x13/new_13x13.sum()), axis=0))

# print((new_13x13/new_13x13.sum()).sum())

# plt.show()

# print(tabs.sum())

from src.tools.zurflueh_filter.zfilter import create_zfilter
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.03f}')
def create_filter(width: int):
    linspace = np.arange(width)-width//2
    xx, yy = np.meshgrid(linspace, linspace)
    zfiltered = create_zfilter(pixel_width=width)
    plot_surf(
        np.stack((xx, yy, zfiltered), axis=0)
    )
    return zfiltered

max_ = create_filter(33).max()
ax.set_zlim(0, max_)

plt.show()
