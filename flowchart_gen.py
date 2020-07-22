from PIL import Image

from core import get_img, cloud_mask_generate_procedure
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy

flowchart_img_dir = './flowchart'


def save_img(img, name):
    Image.fromarray(img).convert('RGB').save('{}/{}.png'.format(flowchart_img_dir, name))


def show_imgs(imgs, name='flowchart'):
    rows = 2
    columns = len(imgs) // rows
    w = columns * 8
    h = rows * 8
    fig = plt.figure(figsize=(w, h))
    idx = 0
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text(imgs[idx][0])
        plt.imshow(Image.fromarray(imgs[idx][1] * 5).resize((1000, 1000)).convert('L'))
        idx += 1
    plt.savefig('{}.png'.format(name), bbox_inches='tight')


def main():
    img = get_img('LuoJia1-01_LR201809083317_20180907212040_HDR_0033')
    imgs = [('Original', img)]
    imgdiffs = [('Original', np.zeros((2000, 2000)))]
    for f in cloud_mask_generate_procedure:
        img2 = deepcopy(img)
        img = f(img)
        title = f.__name__.replace('_', ' ').title()
        imgs.append((title, img))
        imgdiffs.append((title, img2 - img))

    show_imgs(imgs)
    show_imgs(imgdiffs, name='diff')


if __name__ == '__main__':
    main()
