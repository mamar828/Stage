import numpy as np
import graphinglib as gl

from src.tools.zurflueh_filter.zfilter import zfilter

rand = np.random.normal(0, 0.1, (50,50))
# test_array_linear = np.tile(np.arange(50), (50,1)) + rand
test_array_linear = np.tile(np.linspace(3,7,50), (50,1)) + rand
test_array_diagonal = np.tile(np.arange(50), (50,1)) + np.tile(np.arange(50), (50,1)).transpose()/4 + rand
g = np.vectorize(lambda x, y: 50*np.exp(-((x-25)**2+(y-25)**2)/(2*10**2)))
test_array_gaussian = g(*np.meshgrid(np.arange(50), np.arange(50))) + rand
p = np.vectorize(lambda x, y: 0.1*x+0.03*y + 0.1*x**2+0.005*y**2)
test_array_polynomial2 = p(*np.meshgrid(np.arange(50), np.arange(50))) + rand
# test_array_diagonal[20:25,20:25] = np.NAN
gradient = zfilter(test_array_linear, 11)

titles = ["Random Array", "Added Gradient", "Test Array", "Filtered Array", "Filtered Gradient", "Random - Filtered"]
figs = [gl.Figure(title=title) for title in titles]

figs[0].add_elements(gl.Heatmap(rand, show_color_bar=True, vmin=-1.5, vmax=1.5))
figs[1].add_elements(gl.Heatmap(test_array_linear - rand, show_color_bar=True))
figs[2].add_elements(gl.Heatmap(test_array_linear, show_color_bar=True))
figs[3].add_elements(gl.Heatmap(test_array_linear - gradient, show_color_bar=True, vmin=-1.5, vmax=1.5))
figs[4].add_elements(gl.Heatmap(gradient, show_color_bar=True))
figs[5].add_elements(gl.Heatmap(rand - (test_array_linear-gradient), show_color_bar=True, vmin=-1.5, vmax=1.5))

fig = gl.MultiFigure.from_grid(figs, (2,3), size=(14,9))
fig.show()
# fig.save("figures/zfilter/zurflueh_test_diagonal_3.png")

# arr = np.array([
#     [0,1,2,3,4],
#     [5,6,7,8,9],
#     [10,11,12,13,14],
#     [15,16,17,18,19],
#     [20,21,22,23,24]
# ])

# z = zfilter(arr, 5)
# print(np.nanmean(z))
