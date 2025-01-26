import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.feature import match_template

def find_top_matches(corr_map, num_peaks=5):
    """
    Znajduje współrzędne num_peaks najwyższych wartości
    w mapie korelacji corr_map.
    Zwraca listę krotek (wartość_korelacji, wiersz, kolumna).
    """
    # Spłaszczamy mapę korelacji, aby móc łatwo wyszukać największe wartości
    flat = corr_map.flatten()
    # Indeksy posortowane malejąco według wartości korelacji
    sorted_indices = np.argsort(flat)[::-1]
    
    peaks = []
    for idx in sorted_indices[:num_peaks]:
        val = flat[idx]
        # Odtwarzamy 2D (r, c) z indeksu w tablicy spłaszczonej
        r, c = divmod(idx, corr_map.shape[1])
        peaks.append((val, r, c))
    
    return peaks

# --- Główna część skryptu ---
if __name__ == "__main__":
    # 1. Wczytanie obrazu głównego (RGB) i wzorca (RGB)
    image_path = "Webb's_First_Deep_Field.jpg"
    template_path = "wzorzecSMACS.jpg"
    
    image = imread(image_path)   # shape (H, W, 3)
    template = imread(template_path)  # shape (h, w, 3)
    
    # Upewnij się, że oba obrazy mają kanały w kolejności (R,G,B) – w scikit-image
    # jest to zazwyczaj (row, col, channel) i channel=RGB, co nam odpowiada.
    
    # 2. Rozdzielenie na kanały R, G, B
    # image[...,0] = kanał R, image[...,1] = kanał G, image[...,2] = kanał B
    image_r = image[..., 0]
    image_g = image[..., 1]
    image_b = image[..., 2]
    
    template_r = template[..., 0]
    template_g = template[..., 1]
    template_b = template[..., 2]
    
    # 3. Korelacja (match_template) dla każdego kanału osobno
    corr_r = match_template(image_r, template_r)
    corr_g = match_template(image_g, template_g)
    corr_b = match_template(image_b, template_b)
    
    # 4. Łączymy wyniki korelacji np. poprzez sumowanie albo uśrednianie
    # (np. uśrednianie to sum / 3)
    corr_combined = (corr_r + corr_g + corr_b) / 3.0
    
    # 5. Znalezienie 5 najwyższych pików korelacji
    best_peaks = find_top_matches(corr_combined, num_peaks=5)
    
    # Wypiszmy współrzędne i wartości korelacji
    print("Top 5 dopasowań (wartość_korelacji, wiersz, kolumna):")
    for val, r, c in best_peaks:
        print(f"  wartość={val:.4f}, (r={r}, c={c})")
    
    # Wizualizacja mapy korelacji i zaznaczenie najlepszego dopasowania
    # (dla uproszczenia tylko pierwsze maksimum)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("Obraz główny")
    plt.imshow(image)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Mapa korelacji (uśredniona)")
    plt.imshow(corr_combined, cmap='viridis')
    plt.axis("off")
    
    # Zaznaczymy punkt najwyższej korelacji
    best_val, best_r, best_c = best_peaks[0]
    plt.plot(best_c, best_r, 'ro')  # c - oś X (kolumna), r - oś Y (wiersz)
    
    plt.tight_layout()
    plt.show()
