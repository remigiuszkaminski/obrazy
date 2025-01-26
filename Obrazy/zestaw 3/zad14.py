import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage import exposure

obraz = Image.open("ptaki.png").convert("L")
obraz_np = np.array(obraz, dtype=np.float32)


wys, szer = obraz_np.shape

#okno sinus
x = np.arange(szer)
y = np.arange(wys)
X, Y = np.meshgrid(x, y) 

raw_window = np.sin(np.pi * X / szer) * np.sin(np.pi * Y / wys)



obraz_windowed = obraz_np * raw_window
obraz_windowed_norm = 255 * (obraz_windowed - np.min(obraz_windowed)) / (np.max(obraz_windowed) - np.min(obraz_windowed))
obraz_windowed_norm = obraz_windowed_norm.astype(np.uint8)

kernel = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)

# konwulcja
obraz_smoothed = convolve2d(obraz_windowed_norm, kernel, mode='same', boundary='symm') #funckja convolve2d dziala zgodnie z wzorem podanym wprzyk b

obraz_smoothed = np.clip(obraz_smoothed, 0, 255).astype(np.uint8)

korekta_gamma = 0.6
srednia_po = np.mean(obraz_smoothed)
gamma_obraz = exposure.adjust_gamma(obraz_smoothed, korekta_gamma)


obraz_bezposrednio = convolve2d(obraz_np, kernel, mode='same', boundary='symm')


print("Jasnosc srednia obrazu oryginalnego: ", np.mean(obraz_np))
print("Jasnosc srednia obrazu po zastosowaniu filtru: ", np.mean(obraz_smoothed))
print("Jasnosc srednia obrazu po korekcji gamma: ", np.mean(gamma_obraz))
print("Jasnosc srednia obrazu po usrednieniu: ", np.mean(obraz_bezposrednio))
print("janość srednia obrazu po zastosowaniu okna: ", np.mean(obraz_windowed_norm))

#save images
Image.fromarray(obraz_windowed_norm).save("output_image.png")
Image.fromarray(obraz_smoothed).save("output_image_smoothed.png")
Image.fromarray(gamma_obraz).save("output_image_gamma.png")
Image.fromarray(obraz_bezposrednio.astype(np.uint8)).save("output_image_usredniony.png")

