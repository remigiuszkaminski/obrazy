import numpy as np
from PIL import Image
dith_matrix = np.array([
    [6, 14, 2, 8],
    [4, 0, 10, 11],
    [12, 15, 5, 1],
    [9, 3, 13, 7]
])



def dithering(obraz, dith_matrix):
    
    obrazwyj = np.array(obraz, dtype=np.float32)
    wys, szer = obrazwyj.shape
    rozmiar = dith_matrix.shape[0]
    maks = np.max(dith_matrix)

    for y in range(wys):
        for x in range(szer):
            prog = dith_matrix[y % rozmiar, x % rozmiar] / maks * 255
            
            piksel = 0 if obrazwyj[y, x] < prog else 255
            obrazwyj[y, x] = piksel
    
    return Image.fromarray(obrazwyj.astype(np.uint8))

    
    

obraz = Image.open("../stanczyk.png").convert("L")

obrazzditherowany = dithering(obraz, dith_matrix)
obrazzditherowany.save("stanczyk_dithering.png")