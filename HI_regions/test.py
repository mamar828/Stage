import numpy as np


v = np.zeros((3,4,5))
print(v.shape)
c = np.moveaxis(v, 0, 2)
print(c.shape)