from PIL import Image
import os

path = "test/"
dirs = os.listdir(path)


def resize():
    for item in dirs:
        if os.path.isfile(path + item) and item.endswith(".png"):
            f, e = os.path.splitext(path + item)
            if os.path.isfile(f + ' .png'):
                os.remove(f + ' .png')
            im = Image.open(path + item)
            imResize = im.resize((150, 150), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)


resize()
