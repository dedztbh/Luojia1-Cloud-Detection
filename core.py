from PIL import Image
import numpy as np
import cv2
from scipy import ndimage

from global_const import data_dir, R, C


from multiprocessing import get_context

# Helper functions

def get_img(dirname):
    im = Image.open('{}{}/{}_gec.tif'.format(data_dir, dirname, dirname))
    np_img = np.asarray(im.getdata())
    # restore normal FP representation, but magnified 1e5 times or it's too dark
    np_img = np_img ** (3 / 2) * 1e-5
    w, h = im.size
    np_img.shape = (h, w)
    return cv2.resize(np_img, (R, C))

def pmap(f, things):
    with get_context('spawn').Pool() as p:
        return np.asarray(list(p.map(f, things)))
    # return np.asarray(list(map(f, things)))


# These are the steps to run on an image to generate cloud mask.

def unsharp(image):
    unsharp_kernel = np.asarray([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, unsharp_kernel)    

def lowpass(x):
    shape = x.shape
    x = np.where(x <= 70, x, 0)
    x.shape = shape
    return x

def run_avg(x):
    return cv2.blur(x, (5, 5))

def highpass(x):
    shape = x.shape
    x = np.where(14 <= x, x, 0)
    x.shape = shape
    return x

# From https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv/42812226
def remove_small_obj(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
        
    for i in range(0, nb_components):
        if sizes[i] < 200:
            img[output == i + 1] = 0

    return img

def run_regularize_shape(x):
    return ndimage.morphology.grey_dilation(x, size=(100, 100))

def run_gaussian(x):
    return cv2.GaussianBlur(x, (99, 99), 100)