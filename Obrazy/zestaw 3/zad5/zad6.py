import cv2
import numpy as np

def create_lut():
    lut_red = np.zeros(256, dtype=np.uint8)
    lut_green = np.zeros(256, dtype=np.uint8)
    lut_blue = np.zeros(256, dtype=np.uint8)

    # Kanał czerwony
    for i in range(256):
        if i <= 95:
            lut_red[i] = 0
        elif 95 < i <= 159:
            lut_red[i] = int((i - 95) / (159 - 95) * 255)
        elif 159 < i <= 223:
            lut_red[i] = 255
        elif 223 < i <= 255:
            lut_red[i] = int(255 - ((i - 223) / (255 - 223)) * (255 - 128))

    # Kanał zielony
    for i in range(256):
        if i <= 31:
            lut_green[i] = 0
        elif 31 < i <= 95:
            lut_green[i] = int((i - 31) / (95 - 31) * 255)
        elif 95 < i <= 159:
            lut_green[i] = 255
        elif 159 < i <= 223:
            lut_green[i] = int(255 - ((i - 159) / (223 - 159)) * 255)
        else:
            lut_green[i] = 0

    # Kanał niebieski
    for i in range(256):
        if i <= 31:
            lut_blue[i] = int(128 + (i / 31) * (255 - 128))
        elif 31 < i <= 95:
            lut_blue[i] = 255
        elif 95 < i <= 159:
            lut_blue[i] = int(255 - ((i - 95) / (159 - 95)) * 255)
        else:
            lut_blue[i] = 0

    return lut_blue, lut_green, lut_red

def apply_lut(obraz, lut_blue, lut_green, lut_red):
    b, g, r = cv2.split(obraz)

    b = cv2.LUT(b, lut_blue)
    g = cv2.LUT(g, lut_green)
    r = cv2.LUT(r, lut_red)

    transformed_obraz = cv2.merge([b, g, r])
    return transformed_obraz

obraz = cv2.imread("czaszka.png")

lut_blue, lut_green, lut_red = create_lut()

transformed_obraz = apply_lut(obraz, lut_blue, lut_green, lut_red)

cv2.imwrite("zad6.png", transformed_obraz)