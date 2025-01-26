import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu
from skimage import io
import cv2

obraz = io.imread("zyczenia.png", as_gray=True).astype(np.float32)

sigma = 2
gaus = gaussian(obraz, sigma=sigma).astype(np.float32)
plt.imsave("zyczenia_gaus.png", gaus, cmap='gray')

hx = np.array([1, 0, -1], dtype=np.float32)
hy = np.array([[1], [0], [-1]], dtype=np.float32)

hxx = np.outer(hx, hx).astype(np.float32)
hyy = np.outer(hy, hy).astype(np.float32)
hxy = np.outer(hx, hy.flatten()).astype(np.float32)

gxx = cv2.filter2D(gaus, -1, hxx)
gyy = cv2.filter2D(gaus, -1, hyy)
gxy = cv2.filter2D(gaus, -1, hxy)

plt.imsave("zyczenia_hxx.png", gxx, cmap='gray')
plt.imsave("zyczenia_hyy.png", gyy, cmap='gray')
plt.imsave("zyczenia_hxy.png", gxy, cmap='gray')

def hesse(gxx, gyy, gxy):
    det = gxx * gyy - gxy ** 2
    h11h22 = gxx + gyy
    k1 = 0.5 * (h11h22 + np.sqrt(h11h22 ** 2 - 4 * det))
    k2 = 0.5 * (h11h22 - np.sqrt(h11h22 ** 2 - 4 * det))
    
    return k1, k2

k1, k2 = hesse(gxx, gyy, gxy)

plt.imsave("zyczenia_glowna.png", k1, cmap='gray')
direction = np.arctan2(gxy, gxx)
rows, cols = k1.shape
suppressed_k1 = np.copy(k1)
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        angle = direction[i, j] * 180.0 / np.pi
        angle = (angle + 180) % 180

        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
            q = k1[i, j + 1]
            r = k1[i, j - 1]
        elif 22.5 <= angle < 67.5:
            q = k1[i + 1, j - 1]
            r = k1[i - 1, j + 1]
        elif 67.5 <= angle < 112.5:
            q = k1[i + 1, j]
            r = k1[i - 1, j]
        elif 112.5 <= angle < 157.5:
            q = k1[i - 1, j - 1]
            r = k1[i + 1, j + 1]

        if (k1[i, j] >= q) and (k1[i, j] >= r):
            suppressed_k1[i, j] = k1[i, j]
        else:
            suppressed_k1[i, j] = 0

plt.imsave("zyczenia_nms.png", suppressed_k1, cmap='gray')

suppressed_k1_8bit = np.uint8(suppressed_k1 / suppressed_k1.max() * 255)

T2 = threshold_otsu(suppressed_k1_8bit)
T1 = 0.5 * T2
print(T1, T2)

def hysteresis_thresholding(img, low, high):
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)

    strong = 255
    weak = 120

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if res[i, j] == weak:
                if ((res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or (res[i + 1, j + 1] == strong)
                        or (res[i, j - 1] == strong) or (res[i, j + 1] == strong)
                        or (res[i - 1, j - 1] == strong) or (res[i - 1, j] == strong) or (res[i - 1, j + 1] == strong)):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    return res

final_image = hysteresis_thresholding(suppressed_k1_8bit, T1, T2)

plt.imsave("zyczenia_hyst.png", final_image, cmap='gray')