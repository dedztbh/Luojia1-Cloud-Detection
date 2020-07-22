from PIL import Image
import numpy as np
from copy import deepcopy

from core import cloud_mask_generate_procedure


def save_img(img, name='out'):
    Image.fromarray(img).convert('RGB').save('{}.png'.format(name))


def main():
    im = Image.open('ISS030-E-187822.JPG')
    np_img = np.asarray(im.convert('L').getdata()).astype(np.float32)
    w, h = im.size
    np_img.shape = (h, w)
    # img = cv2.resize(np_img, (2000, 2000))
    img = deepcopy(np_img)

    for f in cloud_mask_generate_procedure:
        img = f(img)

    # img = cv2.resize(img, (w, h)).astype(np.uint8)
    img = img.astype(np.uint8)

    save_img(img)

    out_img_rgb = np.dstack((np.uint8(np.round(np_img)),
                             safe_mult(img, 2),
                             np.zeros((h, w), dtype=np.uint8)))

    save_img(out_img_rgb, name='overlay')


def safe_mult(x, v):
    shape = x.shape
    x = np.where(x <= (255 // v), x, (255 // v))
    x.shape = shape
    return x * v


if __name__ == '__main__':
    main()
