import cv2
import numpy as np

def local_otsu_thresholding(image, window_size=11):
    half_size = window_size // 2
    
    padded_img = cv2.copyMakeBorder(
        image,
        half_size,
        half_size,
        half_size,
        half_size,
        cv2.BORDER_REFLECT
    )
    

    binary_output = np.zeros_like(image, dtype=np.uint8)
    
    wys, szer = image.shape
    hist = np.zeros(256, dtype=np.int32)
    
    for i in range(wys):
        for j in range(szer):
            window = padded_img[i:i+window_size, j:j+window_size]
            
            hist[:] = 0  
            for pixel in window.flat:
                hist[pixel] += 1
            
            total = window_size * window_size
            
            sum_total = 0
            for t in range(256):
                sum_total += t * hist[t]
            
            sumB = 0
            wB = 0
            maximum = 0.0
            threshold = 0
            
            for t in range(256):
                wB += hist[t]
                if wB == 0:
                    continue
                wF = total - wB
                if wF == 0:
                    break
                sumB += t * hist[t]
                mB = sumB / wB
                mF = (sum_total - sumB) / wF
                between = wB * wF * (mB - mF) ** 2
                if between > maximum:
                    maximum = between
                    threshold = t

            central_pixel = padded_img[i + half_size, j + half_size]
            binary_output[i, j] = 255 if central_pixel > threshold else 0
    
    return binary_output



img = cv2.imread("../roze.png", cv2.IMREAD_GRAYSCALE)
    
binary_img = local_otsu_thresholding(img, window_size=11)
    
output_path = 'roze_local_otsu.png'
success = cv2.imwrite(output_path, binary_img)
    
