import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

obraz = Image.open("../roze.png").convert("L")
obraz_np = np.array(obraz)

otsu_thresh = threshold_otsu(obraz_np)

binary_np = (obraz_np >= otsu_thresh).astype(np.uint8) * 255



binary_img = Image.fromarray(binary_np)
binary_img.save("roze_otsu.png")