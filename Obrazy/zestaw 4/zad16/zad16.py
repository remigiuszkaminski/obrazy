import numpy as np
from scipy.ndimage import convolve
from skimage import io

obraz = io.imread('./jesien_filtered.png').astype(np.float32)

h = np.array([[1, 4, 6, 4, 1],
              [4, 16, 24, 16, 4],
              [6, 24, 36, 24, 6],
              [4, 16, 24, 16, 4],
              [1, 4, 6, 4, 1]]) / 256.0

def van_citteret_deconv(obraz, dwumian, ile):
    nowy = obraz.copy()
    for _ in range(ile):
        blur = convolve(nowy, dwumian)
        nowy = nowy + (obraz - blur)

    return nowy

obraz_deconv = van_citteret_deconv(obraz, h, 2)
obraz2_deconv = van_citteret_deconv(obraz, h, 5)
obraz3_deconv = van_citteret_deconv(obraz, h, 15)

normalized_obraz_deconv = (255 * (obraz_deconv - np.min(obraz_deconv)) / np.ptp(obraz_deconv)).astype(np.uint8)
normalized_obraz2_deconv = (255 * (obraz2_deconv - np.min(obraz2_deconv)) / np.ptp(obraz2_deconv)).astype(np.uint8)
normalized_obraz3_deconv = (255 * (obraz3_deconv - np.min(obraz3_deconv)) / np.ptp(obraz3_deconv)).astype(np.uint8)

io.imsave('jesien_deconv.png', normalized_obraz_deconv)
io.imsave('jesien2_deconv.png', normalized_obraz2_deconv)
io.imsave('jesien3_deconv.png', normalized_obraz3_deconv)


#roznice
roznica1 = np.abs(obraz2_deconv - obraz_deconv)
roznica2 = np.abs(obraz3_deconv - obraz2_deconv)

roznica1_uint8 = (255 * (roznica1 - np.min(roznica1)) / np.ptp(roznica1)).astype(np.uint8)
roznica2_uint8 = (255 * (roznica2 - np.min(roznica2)) / np.ptp(roznica2)).astype(np.uint8)

io.imsave('jesien_roznica1.png', roznica1_uint8)
io.imsave('jesien_roznica2.png', roznica2_uint8)

