import cv2
import numpy as np


def zhang_sueng_thinning(obraz):

    obraz = obraz.copy()
    obraz[obraz > 0] = 1

    def neighbours(x, y):
        return [ 
            obraz[x - 1, y], obraz[x - 1, y + 1], obraz[x, y + 1], obraz[x + 1, y + 1],
            obraz[x + 1, y], obraz[x + 1, y - 1], obraz[x, y - 1], obraz[x - 1, y - 1]
        ]
    
    def transitions(neighbours):
        n = neighbours + [neighbours[0]]
        return sum((n[i] == 0 and n[i + 1] == 1) for i in range(8))
    
    def zhang_suen_step1(x, y):
        n = neighbours(x, y)
        return 2 <= sum(n) <= 6 and transitions(n) == 1 and n[0] * n[2] * n[4] == 0 and n[2] * n[4] * n[6] == 0
    
    def zhang_suen_step2(x, y):
        n = neighbours(x, y)
        return 2 <= sum(n) <= 6 and transitions(n) == 1 and n[0] * n[2] * n[6] == 0 and n[0] * n[4] * n[6] == 0
    stoper = True
    iter = 0
    while stoper:
        iter += 1
        to_del = []
        stoper = False
        for i in range(1, obraz.shape[0] - 1):
            for j in range(1, obraz.shape[1] - 1):
                if obraz[i, j] == 1 and zhang_suen_step1(i, j):
                    to_del.append((i, j))
        for i, j in to_del:
            obraz[i, j] = 0

        if iter == 1:
            cv2.imwrite('zhang_suen_step1.png', (obraz * 255).astype(np.uint8))
        if to_del:
            stoper = True
        to_del = []
        for i in range(1, obraz.shape[0] - 1):
            for j in range(1, obraz.shape[1] - 1):
                if obraz[i, j] == 1 and zhang_suen_step2(i, j):
                    to_del.append((i, j))
        for i, j in to_del:
            obraz[i, j] = 0
        if iter == 1:
            cv2.imwrite('zhang_suen_step2.png', (obraz * 255).astype(np.uint8))
        if to_del:
            stoper = True
    return obraz

obraz = cv2.imread('mikolaj.png', cv2.IMREAD_GRAYSCALE)
binarny = cv2.threshold(obraz, 127, 1, cv2.THRESH_BINARY_INV)[1]

wynik = zhang_sueng_thinning(binarny)

cv2.imwrite('mikolaj_thinned.png', (wynik * 255).astype(np.uint8))

        