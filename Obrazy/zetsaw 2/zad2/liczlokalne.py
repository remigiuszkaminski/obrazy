import numpy as np
from PIL import Image

def globalny(obraz):

    obrazwyj = np.array(obraz, dtype=np.float32)
    maxval = np.max(obrazwyj)
    minval = np.min(obrazwyj)
    globalnykontrast = (maxval - minval) / 255
    return globalnykontrast

def lokalny(obraz):

    obrazwyj = np.array(obraz, dtype=np.float32)
    wys, szer = obrazwyj.shape

    ilesum = 0
    suma = 0
    # sasiadujacepixele = [ obrazwyj[y - 1, x - 1], obrazwyj[y - 1, x], obrazwyj[y - 1, x + 1], obrazwyj[y, x - 1], obrazwyj[y, x + 1], obrazwyj[y + 1, x - 1], obrazwyj[y + 1, x], obrazwyj[y + 1, x + 1] ]

    for y in range(1, wys - 1):
        for x in range(1, szer - 1):
            sasiadujacepixele = [ obrazwyj[y - 1, x - 1], obrazwyj[y - 1, x], obrazwyj[y - 1, x + 1], obrazwyj[y, x - 1], obrazwyj[y, x + 1], obrazwyj[y + 1, x - 1], obrazwyj[y + 1, x], obrazwyj[y + 1, x + 1] ]
            currpix = obrazwyj[y, x]
            srednia = np.mean(sasiadujacepixele)
            suma += abs(currpix - srednia)
            ilesum += 1

    lokalnykontrast = suma / ilesum
    return lokalnykontrast

mucha1 = Image.open("muchaA.png").convert("L")
mucha2 = Image.open("muchaB.png").convert("L")
mucha3 = Image.open("muchaC.png").convert("L")

globalnyMucha1 = globalny(mucha1)
globalnyMucha2 = globalny(mucha2)
globalnyMucha3 = globalny(mucha3)

lokalnyMucha1 = lokalny(mucha1)
lokalnyMucha2 = lokalny(mucha2)
lokalnyMucha3 = lokalny(mucha3)

print("Globalny kontrast muchy A: ", globalnyMucha1)
print("Globalny kontrast muchy B: ", globalnyMucha2)
print("Globalny kontrast muchy C: ", globalnyMucha3)
print("Lokalny kontrast muchy A: ", lokalnyMucha1)
print("Lokalny kontrast muchy B: ", lokalnyMucha2)
print("Lokalny kontrast muchy C: ", lokalnyMucha3)
