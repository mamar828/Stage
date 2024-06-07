import numpy as np
import graphinglib as gl


# def func(x, y):
#     return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y**2) ** 2


# rng = np.random.default_rng()
# points = rng.random((1000, 2))
# values = func(points[:, 0], points[:, 1])

# fig = gl.Figure()
# hm = gl.Heatmap.from_points(
#     points,
#     values,
#     (0, 1),
#     (0, 1),
#     origin_position="lower"
# )


# # map = gl.Heatmap.from_function(
# #     lambda x, y: np.cos(x * 0.2) + np.sin(y * 0.3), (0, 49), (49, 0)
# # )

# # points = gl.Heatmap.from_function(
# #     lambda x, y: np.cos(x * 0.2) + np.sin(y * 0.3), (0, 49), (49, 0)
# # )

# # points = gl.Heatmap(

# # )

# fig.add_elements(hm)
# fig.show()



# def func(x, y):
#     return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y**2) ** 2


# rng = np.random.default_rng()
# points = rng.random((1000, 2))
# values = func(points[:, 0], points[:, 1])

# fig = gl.Figure()
# # hm = gl.Heatmap(
# #     points,
# #     values,
# #     (0, 1),
# #     (0, 1),
# #     origin_position="lower",
# # )



# hm = gl.Heatmap(
#     np.random.random((100,100))
# )


# fig.add_elements(hm)
# fig.show()




# def func(x, y):
#     return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y**2) ** 2


# rng = np.random.default_rng()
# points = rng.random((1000, 2))
# values = func(points[:, 0], points[:, 1])

# fig = gl.Figure()
# hm = gl.Heatmap(
#     np.stack([points, values], axis=0)
# )

# # hm = gl.Heatmap.from_points(
# #     points,
# #     values,
# #     (0, 1),
# #     (0, 1),
# #     grid_interpolation="cubic",
# #     number_of_points=(100, 100),
# #     origin_position="lower",
# # )
# fig.add_elements(hm)
# fig.show()

arr = np.random.random((25, 2))

sc = gl.Scatter(
    arr[:,0],
    arr[:,1],
)

fig = gl.Figure()
fig.add_elements(sc)
fig.show()
