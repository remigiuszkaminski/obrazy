import cv2
import numpy as np

obraz = cv2.imread('Sky_And_Water_I.png', cv2.IMREAD_GRAYSCALE)

matriks = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])

wys, szer = obraz.shape

tranformacja = matriks[:2, :]

transformacja_z__sasiadem = cv2.warpAffine(obraz, tranformacja, (szer + int(0.5 * wys), wys), flags=cv2.INTER_NEAREST)
transformacja_bilinear = cv2.warpAffine(obraz, tranformacja, (szer + int(0.5 * wys), wys), flags=cv2.INTER_LINEAR)

cv2.imwrite('Sky_And_Water_I_nearest.png', transformacja_z__sasiadem)
cv2.imwrite('Sky_And_Water_I_bilinear.png', transformacja_bilinear)

