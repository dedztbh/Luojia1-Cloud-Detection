from PIL import Image
import numpy as np
import cv2

from core import cloud_mask_generate_procedure_binary


def save_img(img, name='out'):
    Image.fromarray(img).convert('RGB').save('{}.png'.format(name))


def main():
    im = Image.open('ISS035-E-17200.JPG')
    np_img = np.asarray(im.convert('L').getdata()).astype(np.float32)
    w, h = im.size
    np_img.shape = (h, w)
    img = cv2.resize(np_img, (2000, 2000))
    # img = deepcopy(np_img)

    for f in cloud_mask_generate_procedure_binary:
        img = f(img)

    img = cv2.resize(img, (w, h)).astype(np.uint8)
    img = img.astype(np.uint8)

    save_img(img * 70)

    out_img_rgb = np.dstack((np.uint8(np.round(np_img)),
                             safe_mult(img, 70),
                             np.zeros((h, w), dtype=np.uint8)))

    save_img(out_img_rgb, name='overlay')


def safe_mult(x, v):
    shape = x.shape
    x = np.where(x <= (255 // v), x, (255 // v))
    x.shape = shape
    return x * v


if __name__ == '__main__':
    main()
