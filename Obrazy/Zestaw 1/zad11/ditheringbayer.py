import numpy as np
from PIL import Image

def bayer_dith_czarnobialo(obraz):

    obraz_wyj = np.array(obraz, dtype=np.float32)
    wys, szer = obraz_wyj.shape


    bayer_matrix = np.array([
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21]
    ])

    macierz_rozm = bayer_matrix.shape[0]
    maks = np.max(bayer_matrix)

    for y in range(wys):
        for x in range(szer):
            prog = bayer_matrix[y % macierz_rozm, x % macierz_rozm] / maks * 255  
            piksel = 0 if obraz_wyj[y, x] < prog else 255
            obraz_wyj[y, x] = piksel
    
    return Image.fromarray(obraz_wyj.astype(np.uint8))

def bayern_dith_4wartosci(obraz):
    obraz_wyj = np.array(obraz, dtype=np.float32)
    wys, szer = obraz_wyj.shape

    bayer_matrix = np.array([
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21]
    ])

    macierz_rozm = bayer_matrix.shape[0]
    maks = np.max(bayer_matrix)


    wartosci = [50, 100, 150, 200]

    for y in range(wys):
        for x in range(szer):
            piksel = min(wartosci, key=lambda z: abs(z - obraz_wyj[y, x]))
            obraz_wyj[y, x] = piksel
    
    return Image.fromarray(obraz_wyj.astype(np.uint8))
            

obraz = Image.open("../stanczyk.png").convert("L")

dithering = bayer_dith_czarnobialo(obraz)

dithering.save("ditheredczarnobialo_stanczyk.png")

ditheringprzykB = bayern_dith_4wartosci(obraz)

ditheringprzykB.save("dithered4wartosci_stanczyk.png")