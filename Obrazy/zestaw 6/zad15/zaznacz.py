import cv2
import numpy as np
def highlight_removed_pixels(original, modified, output_path):

    original = (original > 0).astype(np.uint8)
    modified = (modified > 0).astype(np.uint8)

    if original.shape != modified.shape:
        modified = cv2.resize(modified, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    original_color = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    

    removed_pixels = (original == 1) & (modified == 0)
    
    original_color[removed_pixels] = [0, 0, 255]
    cv2.imwrite(output_path, original_color)



pierwszy = cv2.imread('mikolaj.png', cv2.IMREAD_GRAYSCALE)
pierwszy = cv2.bitwise_not(pierwszy)
binarny = cv2.threshold(pierwszy, 127, 1, cv2.THRESH_BINARY_INV)[1]

subiter1 = cv2.imread('zhang_suen_step1.png', cv2.IMREAD_GRAYSCALE)
iter2 = cv2.imread('zhang_suen_step2.png', cv2.IMREAD_GRAYSCALE)
pelna = cv2.imread('mikolaj_thinned.png', cv2.IMREAD_GRAYSCALE)

highlight_removed_pixels(pierwszy, subiter1, 'zaznaczone1.png')
highlight_removed_pixels(pierwszy, iter2, 'zaznaczone2.png')



