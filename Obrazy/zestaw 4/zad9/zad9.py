import numpy as np
from scipy.ndimage import convolve, correlate


obraz1 = np.loadtxt("./g1.txt")
obraz2 = np.loadtxt("./g2.txt")
obraz3 = np.loadtxt("./g3.txt")


hGx = np.array([[-1, 1, 0]])

hPx = np.array([[-1,  0,  1],
                [-1,  0,  1],
                [-1,  0,  1]])

hSx = np.array([[-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]])

hL = np.array([[ 0,  1,  0],
               [ 1, -4,  1],
               [ 0,  1,  0]])

def filtr(img, maskx):
    lewe = convolve(img, maskx)
    prawe = correlate(img, maskx)
    kraw = np.abs(lewe) + np.abs(prawe)
    
    return kraw

def filtr_1(img, maskx):
    kraw = convolve(img, maskx)
    kraw = np.abs(kraw)
    return kraw

def przetworz_obraz(img):
    # 1) Gradient 
    outG = filtr_1(img, hGx)
    
    # 2) Prewitt
    outP = filtr(img, hPx)
    
    # 3) Sobel
    outS = filtr(img, hSx)
    
    # 4) Laplace
    outL = convolve(img, hL) 
    outL[outL < 0] = 0 
    
    
    return outG, outP, outS, outL



g1G, g1P, g1S, g1L = przetworz_obraz(obraz1)
g2G, g2P, g2S, g2L = przetworz_obraz(obraz2)
g3G, g3P, g3S, g3L = przetworz_obraz(obraz3)

g1G = np.clip(g1G, 0, 255)
g1P = np.clip(g1P, 0, 255)
g1S = np.clip(g1S, 0, 255)
g1L = np.clip(np.abs(g1L), 0, 255) 

g2G = np.clip(g2G, 0, 255)
g2P = np.clip(g2P, 0, 255)
g2S = np.clip(g2S, 0, 255)
g2L = np.clip(np.abs(g2L), 0, 255)

g3G = np.clip(g3G, 0, 255)
g3P = np.clip(g3P, 0, 255)
g3S = np.clip(g3S, 0, 255)
g3L = np.clip(np.abs(g3L), 0, 255)

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
