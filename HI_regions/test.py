import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

# v = np.zeros((3,4,5))
# print(v.shape)
# c = np.moveaxis(v, 0, 2)
# print(c.shape)


# def rotate_image(image, angle):
#     rotated_image = ndimage.rotate(image, angle)
#     return rotated_image

# def rotate_back_image(image, angle):
#     # Calculate the inverse angle
#     inverse_angle = -angle

#     # Rotate the image back using the inverse angle
#     rotated_back_image = ndimage.rotate(image, inverse_angle, reshape=True)

#     return rotated_back_image

# # You can use the rotate_image function to rotate the image, and then use the rotate_back_image function to rotate it back using the inverse angle. Make sure to provide the same angle that was used for the initial rotation.

# # Here's an example usage:

# image = Image.open("HI_regions/squirrel.webp")
# rotation_angle = 30

# # Assuming you have an image array called 'image' and an angle called 'rotation_angle'
# rotated_image = rotate_image(image, rotation_angle)
# plt.imshow(rotated_image)
# plt.show()

# # Assuming 'rotated_image' is the rotated image obtained from above
# rotated_back_image = rotate_back_image(rotated_image, rotation_angle)
# plt.imshow(rotated_back_image)
# plt.show()


number = 5.7734

# Format the number with leading zeros and up to 3 decimal places
formatted_number = f"{number:06.3f}"

print(formatted_number)
