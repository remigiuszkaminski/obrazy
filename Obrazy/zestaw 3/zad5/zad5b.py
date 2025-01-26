import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

a = -1.0 / 3.0

obraz = cv2.imread('czaszka.png', cv2.IMREAD_GRAYSCALE)

def hiperbolizacja(obraz, a):
    histogram, bins = np.histogram(obraz.flatten(), 256, [0, 256])
    histogram_znorm = histogram / np.sum(histogram)
    skumulowany_histogram = np.cumsum(histogram_znorm)
    G = 256
    cmin = skumulowany_histogram[skumulowany_histogram > 0].min()
    przeksztalcenie = (skumulowany_histogram - cmin) / (1 - cmin)
    przeksztalcenie = np.power(przeksztalcenie, 1 / (a + 1))
    przeksztalcenie = np.floor(przeksztalcenie * (G - 1)).astype(np.uint8)
    obraz_hiper = przeksztalcenie[obraz]

    return obraz_hiper, skumulowany_histogram


obraz_hiper, skumulowany_histogram = hiperbolizacja(obraz, a)

cv2.imwrite('czaszka_hiperbolizacja.png', obraz_hiper)

#plot histogram skumulowany
plt.plot(skumulowany_histogram, color='black')
plt.title('Histogram skumulowany')
plt.xlabel('Wartosc szarosci')
plt.ylabel('Skumulowane prawdopodobienstwo')
plt.savefig('histogram_skumulowany.png')



