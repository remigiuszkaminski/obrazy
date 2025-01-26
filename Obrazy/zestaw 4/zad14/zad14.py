# import numpy as np
# from scipy.ndimage import convolve
# from skimage import io, img_as_ubyte
# from skimage.color import rgb2gray

# obraz = io.imread('../Escher.png')

# if obraz.ndim == 3:
#     obraz = rgb2gray(obraz)

# obraz = img_as_ubyte(obraz)

# h1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# h2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# g1 = convolve(obraz, h1)
# g2 = convolve(obraz, h2)

# g1_clip = np.clip(g1, 0, 255).astype(np.uint8)
# g2_clip = np.clip(g2, 0, 255).astype(np.uint8)

# g3_float = np.sqrt(g1.astype(float) ** 2 + g2.astype(float) ** 2)
# g3_uint8 = np.clip(g3_float, 0, 255).astype(np.uint8)
# io.imsave('Escher_poziome.png', g1_clip)
# io.imsave('Escher_pionowe.png', g2_clip)
# io.imsave('Escher_krawedzie.png', g3_uint8)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import io, img_as_float, img_as_ubyte
from skimage.color import rgb2gray

# Wczytaj obraz
g = io.imread('../Escher.png')

# Jeśli obraz jest w RGB, konwertuj go na skale szarości
if g.ndim == 3:
    g = rgb2gray(g)

# Konwertuj obraz na 8-bitowy
g_8bit = img_as_ubyte(g)  # Przekształca do formatu 8-bitowego (0-255)

# Konwertuj obraz na 32-bitowy
g_32bit = img_as_float(g)  # Przekształca do formatu 32-bitowego (0.0 - 1.0)

# Definiowanie filtrów poziomych i pionowych (np. Sobel)
h1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Filtr poziomy
h2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Filtr pionowy

# Splot poziomy i pionowy dla obrazu 8-bitowego
g1_8bit = convolve(g_8bit, h1)
g2_8bit = convolve(g_8bit, h2)

# Splot poziomy i pionowy dla obrazu 32-bitowego
g1_32bit = convolve(g_32bit, h1)
g2_32bit = convolve(g_32bit, h2)

# (c) Obliczenie gradientu (sqrt(g1^2 + g2^2)) dla obu obrazów
g3_8bit = np.sqrt(g1_8bit**2 + g2_8bit)
g3_32bit = np.sqrt(g1_32bit**2 + g2_32bit)

# Normalizacja obrazów 32-bitowych do zakresu [0, 1] dla wyświetlania
g1_32bit_normalized = (g1_32bit - g1_32bit.min()) / (g1_32bit.max() - g1_32bit.min())
g2_32bit_normalized = (g2_32bit - g2_32bit.min()) / (g2_32bit.max() - g2_32bit.min())
g3_32bit_normalized = (g3_32bit - g3_32bit.min()) / (g3_32bit.max() - g3_32bit.min())

# Wyświetlanie wyników
fig, axs = plt.subplots(2, 4, figsize=(15, 10))

# Obraz 8-bitowy (Wejściowy)
axs[0, 0].imshow(g_8bit, cmap='gray')
axs[0, 0].set_title('Obraz 8-bitowy (Wejściowy)')
axs[0, 0].axis('off')

# Obraz po splotach poziomych 8-bitowych
axs[0, 1].imshow(g1_8bit, cmap='gray')
axs[0, 1].set_title('Splot poziomy (8-bitowy)')
axs[0, 1].axis('off')

# Obraz po splotach pionowych 8-bitowych
axs[0, 2].imshow(g2_8bit, cmap='gray')
axs[0, 2].set_title('Splot pionowy (8-bitowy)')
axs[0, 2].axis('off')

# Obraz gradientu 8-bitowego
axs[0, 3].imshow(g3_8bit, cmap='gray')
axs[0, 3].set_title('Gradient (8-bitowy)')
axs[0, 3].axis('off')

# Obraz 32-bitowy (Wejściowy)
axs[1, 0].imshow(g_32bit, cmap='gray')
axs[1, 0].set_title('Obraz 32-bitowy (Wejściowy)')
axs[1, 0].axis('off')

# Obraz po splotach poziomych 32-bitowych
axs[1, 1].imshow(g1_32bit_normalized, cmap='gray')
axs[1, 1].set_title('Splot poziomy (32-bitowy)')
axs[1, 1].axis('off')

# Obraz po splotach pionowych 32-bitowych
axs[1, 2].imshow(g2_32bit_normalized, cmap='gray')
axs[1, 2].set_title('Splot pionowy (32-bitowy)')
axs[1, 2].axis('off')

# Obraz gradientu 32-bitowego
axs[1, 3].imshow(g3_32bit_normalized, cmap='gray')
axs[1, 3].set_title('Gradient (32-bitowy)')
axs[1, 3].axis('off')

plt.tight_layout()
plt.show()
