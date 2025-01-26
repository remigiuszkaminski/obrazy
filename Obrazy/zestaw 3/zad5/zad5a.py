import numpy as np
import cv2
import matplotlib.pyplot as plt

def wyrownanie_histogramu(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    piksele = image.size
    histogram_normalized = histogram / piksele
    skumulowany_histogram = np.cumsum(histogram_normalized) # skumulowana suma
    assert np.isclose(skumulowany_histogram[-1], 1.0) 

    G = 256  
    cmin = skumulowany_histogram[skumulowany_histogram > 0].min()
    przeksztalcenie = (skumulowany_histogram - cmin) / (1 - cmin)
    przeksztalcenie = (przeksztalcenie * (G - 1)).astype(np.uint8)
    print(image.min(), image.max())
    print(przeksztalcenie.min(), przeksztalcenie.max())
    print(skumulowany_histogram.min(), skumulowany_histogram.max())
    print(histogram_normalized.min(), histogram_normalized.max())


    obraz_wyrownany = przeksztalcenie[image]

    return obraz_wyrownany, histogram, skumulowany_histogram, histogram_normalized


image = cv2.imread("czaszka.png", cv2.IMREAD_GRAYSCALE)


wyrownany_obraz, histogram, skumulowany_histogram, histogram_normalized = wyrownanie_histogramu(image)

cv2.imwrite("czaszka_wyrownanie.png", wyrownany_obraz)

# wykres histogram normalized

plt.plot(histogram_normalized, color='black')
plt.title('Histogram znormalizowany')
plt.xlabel('Wartosc szarosci')
plt.ylabel('Prawdopodobienstwo')
plt.savefig('histogram_normalized.png')




