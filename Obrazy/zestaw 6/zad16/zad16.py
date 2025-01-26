import numpy as np
from skimage import io, morphology
import cv2
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte
from skimage.measure import find_contours

obraz = cv2.imread("StoLat.png", cv2.IMREAD_GRAYSCALE)

otsu_thresh = threshold_otsu(obraz)
binarny = obraz > otsu_thresh

io.imsave("StoLat_otsu.png", img_as_ubyte(binarny))


dylatacja = np.ones((30, 1))
binarny_dylatacja = morphology.dilation(binarny, dylatacja)
#save
io.imsave("StoLat_dylatacja.png", img_as_ubyte(binarny_dylatacja))

erozja = np.ones((15,1))
binarny_erozja = morphology.erosion(binarny_dylatacja, erozja)
#save
io.imsave("StoLat_erozja.png", img_as_ubyte(binarny_erozja))

kontury = find_contours(binarny_erozja, 0.7)

kontury = kontury[:-1]
przefiltrowane = [x for x in kontury if np.max(x[:, 1]) - np.min(x[:, 1]) < np.max(x[:, 0]) - np.min(x[:, 0])]


konturowy = np.zeros_like(obraz)

for kontur in przefiltrowane:
    for x in kontur:
        konturowy[int(x[0]), int(x[1])] = 255

io.imsave("StoLat_kontury.png", img_as_ubyte(konturowy))

kontury_na_obrazie = obraz.copy()

for kontur in przefiltrowane:
    for x in kontur:
        kontury_na_obrazie[int(x[0]), int(x[1])] = 255

io.imsave("StoLat_kontury_na_obrazie.png", img_as_ubyte(kontury_na_obrazie))