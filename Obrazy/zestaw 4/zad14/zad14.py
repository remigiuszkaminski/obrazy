import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

obraz = cv2.imread('Escher.png', cv2.IMREAD_GRAYSCALE)

h1 = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]])
h2 = np.array([[-1,  0,  1],
               [-2,  0,  2],
               [-1,  0,  1]]) 

def convolve(obraz, kernel):
    return convolve2d(obraz, kernel, mode='same', boundary='symm')

obraz_8bit = obraz.astype(np.uint8)
obraz_32bit = obraz.astype(np.float32)

g1_8bit = convolve(obraz_8bit, h1)
g2_8bit = convolve(obraz_8bit, h2)
g3_8bit = np.sqrt(g1_8bit**2 + g2_8bit**2)

g1_8bit = np.clip(g1_8bit, 0, 255).astype(np.uint8)
g2_8bit = np.clip(g2_8bit, 0, 255).astype(np.uint8)
g3_8bit = np.clip(g3_8bit, 0, 255).astype(np.uint8)

g1_32bit = convolve(obraz_32bit, h1)
g2_32bit = convolve(obraz_32bit, h2)
g3_32bit = np.sqrt(g1_32bit**2 + g2_32bit**2) 

g1_32bit_norm = (g1_32bit - g1_32bit.min()) / (g1_32bit.max() - g1_32bit.min())
g2_32bit_norm = (g2_32bit - g2_32bit.min()) / (g2_32bit.max() - g2_32bit.min())
g3_32bit_norm = (g3_32bit - g3_32bit.min()) / (g3_32bit.max() - g3_32bit.min())

plt.imsave('g1_32bit.png', g1_32bit_norm, cmap='gray')
plt.imsave('g2_32bit.png', g2_32bit_norm, cmap='gray')
plt.imsave('g3_32bit.png', g3_32bit_norm, cmap='gray')

#save images
plt.imsave('g1_8bit.png', g1_8bit, cmap='gray')
plt.imsave('g2_8bit.png', g2_8bit, cmap='gray')
plt.imsave('g3_8bit.png', g3_8bit, cmap='gray')
