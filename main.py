import torch
from PIL import Image
import numpy as np
import os
import pickle
import torch.nn.functional as F
import cv2

from filters import lowpass, highpass

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
            # restore normal FP representation, but magnified 1e5 times or it's too dark
            np_img = np_img ** (3 / 2) * 1e-5
            w, h = im.size
            np_img.shape = (h, w)
            images.append(cv2.resize(np_img, (R, C)))
    with open('images.pkl', 'wb') as f:
        pickle.dump(images, f)


def show_tensor(out_img):
    Image.fromarray(out_img.squeeze().numpy()).show()


img = images[1]

Image.fromarray(img).show()

kernel = np.genfromtxt('filter.csv', delimiter=',')
kernel.shape = (1, 1, 3, 3)
kernel = torch.from_numpy(kernel)
img_t = torch.from_numpy(img)
img_t = img_t.unsqueeze(0).unsqueeze(0)
out_img = F.conv2d(img_t, kernel, padding=1)  # unsharp mask
# show_tensor(out_img)  # streetlights are sharpened and cloud remains

out_img = torch.from_numpy(lowpass(out_img.numpy(), 70))
# show_tensor(out_img)  # now only cloud remains

out_img = F.avg_pool2d(out_img, 8)
# show_tensor(out_img * 10)  # filtered out streetlights

out_img = highpass(out_img.squeeze().numpy(), 10)  # remove noise

out_img_rgb = np.uint8(np.round(np.dstack((img, cv2.resize(out_img, (R, C)) * 10, np.zeros((R, C))))))
out_img_rgb = Image.fromarray(out_img_rgb)
out_img_rgb.show()
