import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import find_contours


obraz = cv2.imread("SzukanieJedynek.png", cv2.IMREAD_GRAYSCALE)

#obrot o 15
(h, w) = obraz.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 15, 1.0)
obraz_obrot = cv2.warpAffine(obraz, M, (w, h))

cv2.imwrite("SzukanieJedynek_obrot.png", obraz_obrot)

otsu_thresh = threshold_otsu(obraz)
binarny = obraz_obrot < otsu_thresh

cv2.imwrite("SzukanieJedynek_obrot_otsu.png", binarny.astype(np.uint8) * 255)

