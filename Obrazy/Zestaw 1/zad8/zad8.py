import numpy as np
from PIL import Image

def floyd_steinberg_bin(image, threshold=39):

    gray_im = np.array(image.convert('L'))
    output_im = np.array(gray_im, dtype=np.float32)

    height, width = output_im.shape

    for y in range(height):
        for x in range(width):
            old_pixel = output_im[y, x]
            new_pixel = 0 if old_pixel < threshold else 255
            output_im[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < width:
                output_im[y, x + 1] += error * 7 / 16
            if y + 1 < height:
                if x > 0:
                    output_im[y + 1, x - 1] += error * 3 / 16
                output_im[y + 1, x] += error * 5 / 16
                if x + 1 < width:
                    output_im[y + 1, x + 1] += error / 16
    output_im = np.clip(output_im, 0, 255)
    return Image.fromarray(output_im.astype(np.uint8))


def floyd_steinberg_5wartosci(image):

    gray_im = np.array(image.convert('L'))
    output_im = np.array(gray_im, dtype=np.float32)

    height, width = output_im.shape

    thresholds = [20, 40, 60, 120, 255]
    wartosci = [0, 64, 128, 192, 255]

    for y in range(height):
        for x in range(width):
            old_pixel = output_im[y, x]
            
            if old_pixel < thresholds[0]:
                new_pixel = wartosci[0]
            elif old_pixel < thresholds[1]:
                new_pixel = wartosci[1]
            elif old_pixel < thresholds[2]:
                new_pixel = wartosci[2]
            elif old_pixel < thresholds[3]:
                new_pixel = wartosci[3]
            elif old_pixel < thresholds[4]:
                new_pixel = wartosci[4]
            
            output_im[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < width:
                output_im[y, x + 1] += error * 7 / 16
            if y + 1 < height:
                if x > 0:
                    output_im[y + 1, x - 1] += error * 3 / 16
                output_im[y + 1, x] += error * 5 / 16
                if x + 1 < width:
                    output_im[y + 1, x + 1] += error / 16
    output_im = np.clip(output_im, 0, 255)
    return Image.fromarray(output_im.astype(np.uint8))




obraz = Image.open("../stanczyk.png")

obraz_bin = floyd_steinberg_bin(obraz)
obraz_5wart = floyd_steinberg_5wartosci(obraz)

obraz_bin.save("stanczyk_bin.png")
obraz_5wart.save("stanczyk_5wart.png")