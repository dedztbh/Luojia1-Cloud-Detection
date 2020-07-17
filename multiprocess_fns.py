from PIL import Image
import numpy as np
import cv2
from scipy import ndimage

from globals import data_dir, R, C, avg_ksize


from multiprocessing import get_context


def get_img(dirname):
    im = Image.open('{}{}/{}_gec.tif'.format(data_dir, dirname, dirname))
    np_img = np.asarray(im.getdata())
    # restore normal FP representation, but magnified 1e5 times or it's too dark
    np_img = np_img ** (3 / 2) * 1e-5
    w, h = im.size
    np_img.shape = (h, w)
    return cv2.resize(np_img, (R, C))


# From https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv/42812226
def remove_small_obj(img):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time
    # we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 200

    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img[output == i + 1] = 0

    return img


def unsharp(image):
    unsharp_kernel = np.genfromtxt('unsharp.csv', delimiter=',')
    return cv2.filter2D(image, -1, unsharp_kernel)


def run_avg(x):
    return cv2.blur(x, (avg_ksize, avg_ksize))

def run_gaussian(x):
    return cv2.GaussianBlur(x, (199, 199), 200)


def run_regularize_shape(x):
    return ndimage.morphology.grey_dilation(x, size=(200, 200))


def pmap(f, things):
    with get_context('spawn').Pool() as p:
        return list(p.map(f, things))
    # return list(map(f, things))
