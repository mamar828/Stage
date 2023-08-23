from astropy.io import fits
from fits_analyzer import *

from scipy import fft
from matplotlib.colors import LogNorm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

array = fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0].data
array[np.isnan(array)] = 0

import numpy as np
import matplotlib.pyplot as plt

# Create a sample 2D array
# size_x = 64
# size_y = 64
# x = np.arange(size_x)
# y = np.arange(size_y)
# xx, yy = np.meshgrid(x, y)
# array = np.sin(0.1 * xx) + np.cos(0.2 * yy)

# Compute the 2D FFT
fft_array = np.fft.fft2(array)

# Shift the zero frequency component to the center
fft_array_shifted = np.fft.fftshift(fft_array)

# Define a cutoff frequency (remove frequencies beyond this)
cutoff_frequency = 0.1

# Create a mask to keep low-frequency components
rows, cols = array.shape
center_row, center_col = rows // 2, cols // 2
mask = np.zeros_like(array, dtype=np.bool_)
mask[center_row - int(center_row * cutoff_frequency):center_row + int(center_row * cutoff_frequency),
     center_col - int(center_col * cutoff_frequency):center_col + int(center_col * cutoff_frequency)] = True

# Apply the mask to the shifted FFT
filtered_fft = fft_array_shifted * mask

# Shift the zero frequency component back to the corner
filtered_fft_shifted = np.fft.fftshift(filtered_fft)

# Compute the inverse FFT to obtain the filtered image
filtered_image = np.fft.ifft2(filtered_fft_shifted).real

# Plot the original image, the filtered image, and the mask
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(array, cmap='viridis', vmin=10, vmax=40)
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(filtered_image, cmap='viridis', vmin=10, vmax=40)
plt.title('Filtered Image')
plt.subplot(1, 3, 3)
plt.imshow(mask, cmap='viridis')
plt.title('Mask')
plt.show()














# import numpy as np
# import matplotlib.pyplot as plt



# # Compute the 2D Fourier transform
# fft_array = np.fft.fft2(array)

# # Shift the zero frequency component to the center
# fft_array_shifted = np.fft.fftshift(fft_array)

# # Compute the magnitude spectrum (absolute values)
# magnitude_spectrum = np.abs(fft_array_shifted)

# # Plot the original image and its magnitude spectrum
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(array, cmap='gray')
# plt.title('Original Image')
# plt.subplot(1, 2, 2)
# plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
# plt.title('Magnitude Spectrum (log scale)')
# plt.show()






# # afficher l'image
# plt.imshow(image, cmap='viridis', vmin=0, vmax=40)
# plt.show()


# def afficher_tf(tf):
#     """ Affiche le spectre d'une image

#     Args :
#     tf -- la transformée de Fourier d'une image
#     """
#     magnitude_spectrum = 20*np.log(np.abs(tf))
#     plt.imshow(magnitude_spectrum.astype(np.uint8))
#     plt.show()


# # calculer le spectre de l'image
# im_fft = fft.fft2(image)
# plt.imshow(im_fft.astype(np.uint8))
# plt.show()
# # décaler le spectre
# fft_shift = fft.fftshift(im_fft)

# # afficher du spectre
# afficher_tf(fft_shift)

# # filtrer le spectre
# lignes, colonnes = fft_shift.shape
# centre = (lignes//2, colonnes//2)

# fft_shift[centre[0]-40:centre[0]+40, centre[1]-40:centre[1]+40] = 0

# # afficher le spectre
# afficher_tf(fft_shift)

# # calculer l'image filtrée
# img_filtre = fft.ifft2(fft_shift)
# img_filtre = np.abs(img_filtre)

# # afficher l'image filtrée
# plt.imshow(img_filtre, cmap="viridis")
# plt.show()