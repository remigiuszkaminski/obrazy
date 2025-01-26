import numpy as np
from scipy.ndimage import convolve

# Wczytanie przykładowych obrazów 2D (np. plików .txt z macierzami)
obraz1 = np.loadtxt("./g1.txt")
obraz2 = np.loadtxt("./g2.txt")
obraz3 = np.loadtxt("./g3.txt")

# -- DEFINICJE MASEK ---------------------------------------------
# Prosty filtr "gradientowy" (dwukierunkowy)
hGx = np.array([[-1, -1, -1],
                [ 0,  0,  0],
                [ 1,  1,  1]])
hGy = np.array([[-1,  0,  1],
                [-1,  0,  1],
                [-1,  0,  1]])

# Prewitt (x i y)
hPx = np.array([[-1,  0,  1],
                [-1,  0,  1],
                [-1,  0,  1]])
hPy = np.array([[-1, -1, -1],
                [ 0,  0,  0],
                [ 1,  1,  1]])

# Sobel (x i y)
hSx = np.array([[-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]])
hSy = np.array([[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]])

# Laplace (z ujemną wartością w środku)
hL = np.array([[ 0,  1,  0],
               [ 1, -4,  1],
               [ 0,  1,  0]])

# -- FUNKCJA POMOCNICZA: wylicza odpowiedź filtra gradientowego (x,y) -----------
def filtr_gradientowy(img, maskx, masky):
    gx = convolve(img, maskx)
    gy = convolve(img, masky)
    grad = np.sqrt(gx*gx + gy*gy)
    return grad

# -- DLA KAŻDEGO OBRAZU LICZYMY G, PREWITT, SOBEL, LAPLACE ----------------------

def przetworz_obraz(img):
    # 1) Gradient (dwukierunkowy)
    outG = filtr_gradientowy(img, hGx, hGy)
    
    # 2) Prewitt
    outP = filtr_gradientowy(img, hPx, hPy)
    
    # 3) Sobel
    outS = filtr_gradientowy(img, hSx, hSy)
    
    # 4) Laplace
    outL = convolve(img, hL)  # tu ewentualnie można brać np.abs() lub zero-crossing
    
    return outG, outP, outS, outL

# Przetwarzamy każdy z trzech obrazów:
g1G, g1P, g1S, g1L = przetworz_obraz(obraz1)
g2G, g2P, g2S, g2L = przetworz_obraz(obraz2)
g3G, g3P, g3S, g3L = przetworz_obraz(obraz3)

# -- OPCJONALNIE: wartość bezwzględna i przycięcie do [0..255] ------------------

g1G = np.clip(g1G, 0, 255)
g1P = np.clip(g1P, 0, 255)
g1S = np.clip(g1S, 0, 255)
g1L = np.clip(np.abs(g1L), 0, 255)  # Laplace np. w wartości bezwzględnej

# i tak samo dla g2*, g3* ...
g2G = np.clip(g2G, 0, 255)
g2P = np.clip(g2P, 0, 255)
g2S = np.clip(g2S, 0, 255)
g2L = np.clip(np.abs(g2L), 0, 255)

g3G = np.clip(g3G, 0, 255)
g3P = np.clip(g3P, 0, 255)
g3S = np.clip(g3S, 0, 255)
g3L = np.clip(np.abs(g3L), 0, 255)

# -- PROGOWANIE NA BINARY [0,255] ----------------------------------------------
def binarize(img, thresh=128):
    return ((img > thresh).astype(np.uint8))*255

g1Gbin = binarize(g1G)
g1Pbin = binarize(g1P)
g1Sbin = binarize(g1S)
g1Lbin = binarize(g1L)

g2Gbin = binarize(g2G)
g2Pbin = binarize(g2P)
g2Sbin = binarize(g2S)
g2Lbin = binarize(g2L)

g3Gbin = binarize(g3G)
g3Pbin = binarize(g3P)
g3Sbin = binarize(g3S)
g3Lbin = binarize(g3L)

# -- ZAPIS DO TEKSTOWYCH PLIKÓW (jeśli potrzebne) -------------------------------
np.savetxt("g1Gb.txt", g1Gbin, fmt="%d")
np.savetxt("g1Pb.txt", g1Pbin, fmt="%d")
np.savetxt("g1Sb.txt", g1Sbin, fmt="%d")
np.savetxt("g1Lb.txt", g1Lbin, fmt="%d")

np.savetxt("g2Gb.txt", g2Gbin, fmt="%d")
np.savetxt("g2Pb.txt", g2Pbin, fmt="%d")
np.savetxt("g2Sb.txt", g2Sbin, fmt="%d")
np.savetxt("g2Lb.txt", g2Lbin, fmt="%d")

np.savetxt("g3Gb.txt", g3Gbin, fmt="%d")
np.savetxt("g3Pb.txt", g3Pbin, fmt="%d")
np.savetxt("g3Sb.txt", g3Sbin, fmt="%d")
np.savetxt("g3Lb.txt", g3Lbin, fmt="%d")
