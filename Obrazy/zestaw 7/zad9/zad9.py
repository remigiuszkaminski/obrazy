import cv2
import numpy as np
from skimage.filters import threshold_otsu
from scipy.interpolate import griddata


image = cv2.imread('Kopernik.png', cv2.IMREAD_GRAYSCALE)
height, width = image.shape

region_size = 23

thresholds = []
for i in range(0, height, region_size):
    for j in range(0, width, region_size):
        subregion = image[i:i+region_size, j:j+region_size]
        local_thresh = threshold_otsu(subregion)
        thresholds.append((i + region_size // 2, j + region_size // 2, local_thresh))

points = np.array([(x, y) for x, y, _ in thresholds])
values = np.array([val for _, _, val in thresholds])
grid_x, grid_y = np.mgrid[0:height, 0:width]
threshold_map = griddata(points, values, (grid_x, grid_y), method='linear')

binary_image = image > threshold_map

cv2.imwrite('zmienny.png', binary_image.astype(np.uint8) * 255)

#global

global_seg = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('global_seg.png', global_seg[1])


