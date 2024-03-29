from PIL import Image

from core import get_img, cloud_mask_generate_procedure_binary
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy

flowchart_img_dir = './flowchart'


def save_img(img, name):
    Image.fromarray(img).convert('RGB').save('{}/{}.png'.format(flowchart_img_dir, name))


def show_imgs(imgs, name='flowchart'):
    rows = 2
    columns = len(imgs) // rows
    w = columns * 16
    h = rows * 16
    fig = plt.figure(figsize=(w, h))
    idx = 0
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text(imgs[idx][0])
        plt.imshow(Image.fromarray(imgs[idx][1]).convert('L'))
        idx += 1
    plt.savefig('{}.png'.format(name), bbox_inches='tight')


def main():
    img = get_img('LuoJia1-01_LR201809083317_20180907212040_HDR_0033')
    imgs = [('Original', img * 5)]
    imgdiffs = [('Original', np.zeros((2000, 2000)))]
    for f in cloud_mask_generate_procedure_binary:
        img2 = deepcopy(img)
        img = f(img)
        title = f.__name__.replace('_', ' ').title()
        imgs.append((title, img * 5))
        imgdiffs.append((title, (img2 - img) * 5))

    show_imgs(imgs)
    show_imgs(imgdiffs, name='diff')

    imgdiffs = [(name, -imgdiff) for (name, imgdiff) in imgdiffs]
    show_imgs(imgdiffs, name='diff2')


if __name__ == '__main__':
    main()
