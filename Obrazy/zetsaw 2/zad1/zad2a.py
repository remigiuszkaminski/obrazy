
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Wczytaj obraz wykresu
image_path = "PlytkaFresnela.png"
image = Image.open(image_path).convert("L")  # Skala szarości



piksele = np.array(image)
wys, ser = piksele.shape

dlugosc_rzekatnej = int(np.ceil(np.sqrt(wys**2 + ser**2)))
                        
x_cord = np.linspace(0, ser - 1, dlugosc_rzekatnej)
y_cord = np.linspace(0, wys - 1, dlugosc_rzekatnej)

liniowy = [piksele[int(round(y)), int(round(x))] for x, y in zip(x_cord, y_cord)]

x = np.arange(len(liniowy))

plt.figure(figsize=(10, 5))
plt.plot(x, liniowy)
plt.title("Profil liniowy")
plt.xlabel("Pozycja")
plt.ylabel("Jasność")
plt.grid(True)

coile = 50
dlg = x[::coile]
profil = liniowy[::coile]

plt.scatter(dlg, profil, c='r', label='Punkty profilu')
plt.legend()

plt.savefig("profil_liniowy.png")
plt.show()

