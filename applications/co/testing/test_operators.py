import numpy as np

from src.hdu.arrays.array_2d import Array2D
from src.hdu.maps.map import Map
from src.hdu.maps.grouped_maps import GroupedMaps


mapi = Map(Array2D(np.array([[1,2,3],[4,5,6],[7,8,9]])), Array2D(np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])))

f = mapi
f /= mapi
print(f.data)
print(f.uncertainties)
