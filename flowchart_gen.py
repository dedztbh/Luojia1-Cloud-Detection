from PIL import Image

from core import get_img, cloud_mask_generate_procedure
import matplotlib.pyplot as plt

flowchart_img_dir = './flowchart'


def save_img(img, name):
    Image.fromarray(img).convert('RGB').save('{}/{}.png'.format(flowchart_img_dir, name))


def show_imgs(imgs):
    rows = 2
    columns = len(imgs) // rows
    w = columns * 4
    h = rows * 4
    fig = plt.figure(figsize=(w, h))
    idx = 0
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text(imgs[idx][0])
        plt.imshow(Image.fromarray(imgs[idx][1] * 5).resize((500, 500)).convert('L'))
        idx += 1
    plt.savefig('flowchart.png', bbox_inches='tight')


def main():
    img = get_img('LuoJia1-01_LR201809083317_20180907212040_HDR_0033')
    imgs = [('Original', img)]
    for f in cloud_mask_generate_procedure:
        img = f(img)
        imgs.append((f.__name__.replace('_', ' ').title(), img))

    show_imgs(imgs)


if __name__ == '__main__':
    main()
