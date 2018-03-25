import os
import numpy as np
import skimage.io, skimage.transform

data_path = "./data/"
img_ext = ".jpg"
min_ext = ".min.jpg"
minned_size = 32

for filename in os.listdir(data_path):
    if filename.endswith(img_ext):
        image = skimage.io.imread(data_path+filename)
        image = skimage.transform.resize(image, (minned_size, minned_size))
        new_filename = data_path + filename[:-4] + min_ext
        skimage.io.imsave(new_filename, image)

