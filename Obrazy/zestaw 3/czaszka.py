import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure

def oblicz_histogram(obraz):
    histogram = np.histogram(obraz.flatten(), bins=256, range=(0, 256))
    return histogram

def normalizuj_histogram(histogram, liczba_pikseli):
    histogram_normalizowany = histogram / liczba_pikseli
    return histogram_normalizowany


def oblicz_histogram_skumulowany(histogram_normalizowany):
    histogram_skumulowany = np.cumsum(histogram_normalizowany)
    return histogram_skumulowany

def hyperbolizuj_histogram(obraz, alpha):
    obraz_float = obraz.astype(np.float32)
    G = 256
    obraz_hyperbolizowany = (G - 1) * np.power(obraz_float / (G - 1), 1 / (alpha + 1))
    obraz_hyperbolizowany = np.clip(obraz_hyperbolizowany, 0, 255).astype(np.uint8)
    return obraz_hyperbolizowany

obraz = io.imread('czaszka.png', as_gray=True)
histogram = oblicz_histogram(obraz)
liczba_pikseli = obraz.size
histogram_normalizowany = normalizuj_histogram(histogram, liczba_pikseli)
histogram_skumulowany = oblicz_histogram_skumulowany(histogram_normalizowany)

np.savetxt('normalized_histogram.txt', histogram_normalizowany)
np.savetxt('cumulative_histogram.txt', histogram_skumulowany)

obraz_hyperbolizowany = hyperbolizuj_histogram(obraz, 1/3)

io.imsave('czaszka_hyperbolizacja.png', obraz_hyperbolizowany)




