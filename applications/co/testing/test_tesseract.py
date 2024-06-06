import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from src.hdu.tesseract import Tesseract


# t = Tesseract(
#     data=ak.Array(
#         [[[[1,0.1,2,0.2,3,0.3], [10,1,20,2,30,3]],[[100,10,200,20,300,30], [np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN]]],
#          [[[1,0.1,2,0.2,3,0.3], [10,1,20,2,30,3]],[[np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN], [np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN]]]]
#     )
# )

# fig, axs = plt.subplots(1)
# t[1:3].to_grouped_maps().mean[0].data.plot(axs)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.tesseract import Tesseract
from src.hdu.maps.grouped_maps import GroupedMaps


# if __name__ == "__main__":
    # cube = CubeCO.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")[500:800,:,:]
    # print(500 + cube.header.get_frame(-4000, axis=0))
    # cube.header["COMMENT"] = "Loop4N1_FinalJS was previously sliced at channel 500, all values of mean must then be " \
    #                        + "added to 500 to account for this shift."
    # chi2, fit_results = cube.fit()
    # # chi2.save("data/Loop4_co/N1/fit_chi2.fits")
    # fit_results.save("data/Loop4_co/N1/fit_tesseract_2.fits")

    # results = fit_results.to_grouped_maps()
    # results.save("data/Loop4_co/Loop4N1_fit.fits")

    # results = GroupedMaps.load("data/Loop4_co/Loop4N1_fit.fits")
    # object_ray = results[195:230]
    # fig, axs = plt.subplots(1)
    # (cube[0,:,:].data*0).plot(axs, alpha=0.5, show_cbar=False)
    # object_ray.mean[0].data.plot(axs)
    # plt.show()

    # t = Tesseract.load("data/Loop4_co/N1/fit_tesseract.fits")
    # ts = t.split(10, 1)

    # ts[0].concatenate(ts[1], axis=1).save("tess_2.fits")
    # for i, tess in enumerate(ts):
    #     tess.save(f"tess_{i}.fits")
    

    # input_array = Tesseract(np.array([
    #     [[1,      np.nan, 3,      np.NAN, 5],
    #      [np.nan, np.nan, 3,      4,      np.NAN],
    #      [1,      2,      np.nan, 4,      np.NAN]],
    #     [[1,      2,      np.NAN, 4,      5],
    #      [1,      np.nan, 3,      4,      np.NAN],
    #      [np.nan, np.nan, np.nan, 4,      np.NAN]],
    #     [[np.nan, 2,      np.NAN, 4,      np.nan],
    #      [1,      2,      np.nan, np.nan, 5],
    #      [np.nan, 2,      np.nan, np.nan, np.NAN]]
    # ]), None)

    # output_array = input_array.compress()
    # print(output_array.data)


tess = Tesseract.load("data/Loop4_co/N1/object.fits")
tess[1,11,12:14] = None
tess.save("test_crop.fits")
