import cv2
import numpy as np

def iterative_otsu_thresholding(obraz, delta_threshold=2):
    T0, _ = cv2.threshold(obraz, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    T_previous = float(T0) 
    delta = float('inf') 

    while delta > delta_threshold:
        class0 = obraz[obraz <= T_previous]
        class1 = obraz[obraz > T_previous]
        
        mean_class0 = np.mean(class0).item() if class0.size > 0 else 0
        mean_class1 = np.mean(class1).item() if class1.size > 0 else 0
        

        T_current = (mean_class0 + mean_class1) / 2
        delta = abs(T_current - T_previous)
        print(f"Nowy próg T: {T_current}, Delta: {delta}")
        
        T_previous = T_current

    return T_previous



obraz_path = "../roze.png" 
obraz = cv2.imread(obraz_path, cv2.IMREAD_GRAYSCALE)


final_threshold = iterative_otsu_thresholding(obraz)

_, binary_obraz = cv2.threshold(obraz, final_threshold, 255, cv2.THRESH_BINARY)
    

cv2.imwrite('roze_3klasy.png', binary_obraz)