import numpy as np
import graphinglib as gl

from src.tools.zurflueh_filter.zfilter import zfilter

rand = (np.random.random((50,50))-0.5)*3
test_array_linear = np.tile(np.arange(50), (50,1)) + rand
test_array_diagonal = np.tile(np.arange(50), (50,1)) + np.tile(np.arange(50), (50,1)).transpose()/4 + rand
g = np.vectorize(lambda x, y: 50*np.exp(-((x-25)**2+(y-25)**2)/(2*10**2)))
test_array_gaussian = g(*np.meshgrid(np.arange(50), np.arange(50))) + rand
p = np.vectorize(lambda x, y: 0.1*x+0.03*y + 0.1*x**2+0.005*y**2)
test_array_polynomial2 = p(*np.meshgrid(np.arange(50), np.arange(50))) + rand
test_array_diagonal[20:25,20:25] = np.NAN
gradient = zfilter(test_array_diagonal)

titles = ["Random Array", "Added Gradient", "Test Array", "Filtered Array", "Filtered Gradient", "Random - Filtered"]
figs = [gl.Figure(title=title) for title in titles]

figs[0].add_elements(gl.Heatmap(rand, show_color_bar=True, vmin=-1.5, vmax=1.5))
figs[1].add_elements(gl.Heatmap(test_array_diagonal - rand, show_color_bar=True))
figs[2].add_elements(gl.Heatmap(test_array_diagonal, show_color_bar=True))
figs[3].add_elements(gl.Heatmap(test_array_diagonal - gradient, show_color_bar=True, vmin=-1.5, vmax=1.5))
figs[4].add_elements(gl.Heatmap(gradient, show_color_bar=True))
figs[5].add_elements(gl.Heatmap(rand - (test_array_diagonal-gradient), show_color_bar=True, vmin=-1.5, vmax=1.5))

fig = gl.MultiFigure.from_grid(figs, (2,3), size=(14,9))
fig.save("figures/zfilter/zurflueh_test_diagonal_2.pdf")
