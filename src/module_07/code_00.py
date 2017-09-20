import os, numpy as np

# import the PIL image lib
from PIL import Image

# the full file path for an image
file_path = os.path.join('data','notMNIST','A','MDEtMDEtMDAudHRm.png')

# init image as a PIL image object
pil_image = Image.open(file_path)

# convert PIL image to numpy array
np_image = np.array(pil_image)

print('PIL image type: {}'       .format(type(pil_image)))
print('numpy image data type: {}'.format(type(np_image)))
print('numpy image data shape:{}'.format(np_image.shape))

'''
    OK, here is idiomatic python, numpy, PIL image data manipulation
'''