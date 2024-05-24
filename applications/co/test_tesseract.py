import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from src.hdu.tesseract import Tesseract


t = Tesseract(
    data=ak.Array(
        [[[[1,0.1,2,0.2,3,0.3], [10,1,20,2,30,3]],[[100,10,200,20,300,30], [np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN]]],
         [[[1,0.1,2,0.2,3,0.3], [10,1,20,2,30,3]],[[np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN], [np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN]]]]
    )
)

fig, axs = plt.subplots(1)
t[1:3].to_grouped_maps().mean[0].data.plot(axs)
plt.show()
