import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from skimage.filters import gaussian, threshold_otsu
from skimage import img_as_ubyte

obraz = io.imread("pajak.png").astype(np.float32) / 255

#gaus
sigma = 2
gaus = gaussian(obraz, sigma=sigma, channel_axis=-1).astype(np.float32)
gaus_uint8 = (255 * gaus).astype(np.uint8)
io.imsave("pajak_gaus.png", gaus_uint8)

#gradienty soblem rgb
sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

sobelx_r = cv2.filter2D(gaus[:, :, 0], -1, sobelx)
sobelx_g = cv2.filter2D(gaus[:, :, 1], -1, sobelx)
sobelx_b = cv2.filter2D(gaus[:, :, 2], -1, sobelx)

sobely_r = cv2.filter2D(gaus[:, :, 0], -1, sobely)
sobely_g = cv2.filter2D(gaus[:, :, 1], -1, sobely)
sobely_b = cv2.filter2D(gaus[:, :, 2], -1, sobely)

gxmax = np.maximum(np.maximum(sobelx_r, sobelx_g), sobelx_b)
gymax = np.maximum(np.maximum(sobely_r, sobely_g), sobely_b)

# io.imsave("pajak_sobelx2.png", (255 * gxmax).astype(np.uint8))
plt.imsave("pajak_sobelx.png", gxmax, cmap='gray')
plt.imsave("pajak_sobely.png", gymax, cmap='gray')


gradient = np.sqrt(gxmax ** 2 + gymax ** 2)


plt.imsave("pajak_gradient.png", gradient, cmap='gray')

#przyporzadkowac kierunkom gradientu mozliwe kierunki krawedzi
kierunek = np.arctan2(gymax, gxmax)
kierunek = kierunek * 180 / np.pi
kierunek[kierunek < 0] += 180

kierunek_normalized = (kierunek - kierunek.min()) / (kierunek.max() - kierunek.min()) * 255
#zapisz
plt.imsave("pajak_kierunek.png", kierunek_normalized, cmap='gray')

def non_maximum_suppresion(gradient, kierunek):
    nms = np.zeros(gradient.shape)
    kierunek = kierunek % 180
    for i in range(1, gradient.shape[0] - 1):
        for j in range(1, gradient.shape[1] - 1):
            q = 255
            r = 255 
            if 0 <= kierunek[i, j] < 22.5 or 157.5 <= kierunek[i, j] <= 180:
                q = gradient[i, j + 1]
                r = gradient[i, j - 1]
            elif 22.5 <= kierunek[i, j] < 67.5:
                q = gradient[i + 1, j - 1]
                r = gradient[i - 1, j + 1]
            elif 67.5 <= kierunek[i, j] < 112.5:
                q = gradient[i + 1, j]
                r = gradient[i - 1, j]
            elif 112.5 <= kierunek[i, j] < 157.5:
                q = gradient[i - 1, j - 1]
                r = gradient[i + 1, j + 1]
            if gradient[i, j] >= q and gradient[i, j] >= r:
                nms[i, j] = gradient[i, j]
            else:
                nms[i, j] = 0
    return nms

nms = non_maximum_suppresion(gradient, kierunek)

plt.imsave("pajak_nms.png", nms, cmap='gray')

#otsu skalowanko do 255
scaled = img_as_ubyte(nms / nms.max())


T2 = threshold_otsu(scaled)
T1 = 0.5 * T2
print("Wyniki dla t1 i t2", T1, T2)

def hysteresis_thresholding(image, low_threshold, high_threshold):
    weak = 50  # Piksele słabe
    strong = 255  # Piksele silne
    result = np.zeros_like(image)
    strong_pixels = np.where(image >= high_threshold)
    weak_pixels = np.where((image >= low_threshold) & (image < high_threshold))
    result[strong_pixels] = strong
    result[weak_pixels] = weak

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if result[i, j] == weak:
                # Sprawdzanie, czy sąsiaduje z silnym pikselem
                if (strong in result[i - 1:i + 2, j - 1:j + 2]):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result

koncowy = hysteresis_thresholding(scaled, T1, T2)

plt.imsave("pajak_koncowy.png", koncowy, cmap='gray')