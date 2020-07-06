import torch
from PIL import Image
import numpy as np
import os
import pickle
import torch.nn.functional as F

data_dir = './data/'
C = 2048
R = 2048

data_names = []
images = []

if os.path.exists('images.pkl'):
    with open('images.pkl', 'rb') as f:
        images = pickle.load(f)
else:
    for dirname in os.listdir(data_dir):
        if dirname.startswith('LuoJia1'):
            im = Image.open('{}{}/{}_gec.tif'.format(data_dir, dirname, dirname))
            np_img = np.asarray(im.getdata())
            np_img = np_img ** (3 / 2) * 1e-5
            w, h = im.size
            np_img.shape = (h, w)
            im = Image.fromarray(np_img)
            im = im.resize((R, C))
            np_img = np.asarray(im.getdata())
            w, h = im.size
            np_img.shape = (h, w)
            images.append(np_img)
    with open('images.pkl', 'wb') as f:
        pickle.dump(images, f)


def bandpass(x, lo, hi):
    shape = x.shape
    avg = (lo + hi) / 2
    diff = hi - avg
    x = np.where(abs(x - avg) <= diff, x, 0)
    x.shape = shape
    return x


def highpass(x, lo):
    shape = x.shape
    x = np.where(lo <= x, x, 0)
    x.shape = shape
    return x


def lowpass(x, hi):
    shape = x.shape
    x = np.where(x <= hi, x, 0)
    x.shape = shape
    return x


img = images[2]

Image.fromarray(img).show()

filter = np.genfromtxt('filter.csv', delimiter=',')
filter.shape = (1, 1, 3, 3)
filter = torch.from_numpy(filter)

img_t = torch.from_numpy(img)
img_t = img_t.unsqueeze(0).unsqueeze(0)
out_img = F.conv2d(img_t, filter, padding=1)  # unsharp mask
out_img = torch.from_numpy(lowpass(out_img.numpy(), 60))
out_img = F.avg_pool2d(out_img, 8)
out_img = highpass(out_img.squeeze().numpy(), 15) * 10
out_img = Image.fromarray(out_img)
out_img = out_img.resize((C, R))
out_img.show()
