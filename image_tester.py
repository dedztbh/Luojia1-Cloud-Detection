from PIL import Image
import numpy as np
import cv2
import os

from core import cloud_mask_generate_procedure_binary, pmap, get_img
from global_const import data_dir


def save_img(img, name):
    Image.fromarray(img).convert('RGB').save('{}.png'.format(name))


def do_one(name, np_img, w=2000, h=2000):
    if w != 2000 or h != 2000:
        img = cv2.resize(np_img, (2000, 2000))
    else:
        img = np_img

    for f in cloud_mask_generate_procedure_binary:
        img = f(img)

    if w != 2000 or h != 2000:
        img = cv2.resize(img, (w, h)).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    save_img(img * 70, 'output/{}'.format(name))

    out_img_rgb = np.dstack((np.uint8(np.round(np_img)),
                             safe_mult(img, 70),
                             np.zeros((h, w), dtype=np.uint8)))

    save_img(out_img_rgb, 'overlay/{}'.format(name))


def safe_mult(x, v):
    shape = x.shape
    x = np.where(x <= (255 // v), x, (255 // v))
    x.shape = shape
    return x * v


def wrapper(x):
    do_one(x[0], x[1])


if __name__ == '__main__':
    # name = 'ISS035-E-17200.JPG'
    # im = Image.open(name)
    # np_img = np.asarray(im.convert('L').getdata()).astype(np.float32)
    # w, h = im.size
    # np_img.shape = (h, w)
    # do_one(name, np_img)

    data_dirs = []
    for dirname in os.listdir(data_dir):
        if dirname.startswith('LuoJia1'):
            data_dirs.append(dirname)

    data_dirs.sort()

    print(data_dirs)

    images = pmap(get_img, data_dirs)

    pmap(wrapper, zip(data_dirs, images))

    # for (name, im) in zip(data_dirs, images):
    #     do_one(name, im)

    img = get_img('LuoJia1-01_LR201809083317_20180907212040_HDR_0033')
