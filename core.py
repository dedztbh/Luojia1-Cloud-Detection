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

def unsharp(image: np.ndarray):
    unsharp_kernel = np.asarray([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, unsharp_kernel)


def lowpass(x: np.ndarray):
    hi = 72.5 - x.mean() * 2.5

    shape = x.shape
    x = np.where(x <= hi, x, 0)
    x.shape = shape
    return x


def average_blur(x: np.ndarray):
    ksize = 13

    return cv2.blur(x, (ksize, ksize))


def highpass(x: np.ndarray):
    lo = 15 - x.mean() * 2.5

    shape = x.shape
    x = np.where(lo <= x, x, 0)
    x.shape = shape
    return x


# From https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv/42812226
def remove_small_obj(img: np.ndarray):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    sizes = stats[1:, -1]
    
    obj_threshold = np.mean(sizes)
    
    nb_components = nb_components - 1
    for i in range(0, nb_components):
        if sizes[i] < obj_threshold:
            img[output == i + 1] = 0
    return img


def grey_dilation(x: np.ndarray):
    gdsize = round(80 - x.mean() * 5)

    return ndimage.morphology.grey_dilation(x, (gdsize, gdsize))


def gaussian_blur(x: np.ndarray):
    gksize = 125
    gstd = 2000

    return cv2.GaussianBlur(x, (gksize, gksize), gstd)


cloud_mask_generate_procedure = [
    unsharp,
    lowpass,
    average_blur,
    highpass,
    remove_small_obj,
    grey_dilation,
    gaussian_blur
]
