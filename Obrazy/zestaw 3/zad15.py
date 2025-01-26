import cv2
import numpy as np
import matplotlib.pyplot as plt

obraz = cv2.imread('torus.png', cv2.IMREAD_GRAYSCALE)
dark = cv2.imread('Dark_Frame.png', cv2.IMREAD_GRAYSCALE)
flat = cv2.imread('Flat_Frame.png', cv2.IMREAD_GRAYSCALE)

def flat_field_korekt(obraz, dark, flat):
    obraz = obraz.astype(np.float32)
    dark = dark.astype(np.float32)
    flat = flat.astype(np.float32)

    srednia = np.mean(flat - dark)

    small = 1e-10
    korekta = ((obraz - dark) * (srednia)) / (flat - dark + small) #dzielilo przez 0 :(
    korekta = np.clip(korekta, 0, 255).astype(np.uint8)

    return korekta

def kontrast(obraz):
    obraz = obraz.astype(np.float32)
    srednia = np.mean(obraz)
    odchylenie = np.std(obraz)
    if odchylenie < 1e-10:
        odchylenie = 1e-10
    
    obraz_norm = (obraz - srednia) / odchylenie

    return obraz_norm

obraz_korekta = flat_field_korekt(obraz, dark, flat)

cv2.imwrite('torus_korekta.png', obraz_korekta)
obraz_kontrast = kontrast(obraz_korekta)


np.save('torus_korekta_kontrast.npy', obraz_kontrast)

mean_before = np.mean(obraz_korekta)
std_before = np.std(obraz_korekta)



# Obliczenie statystyk po normalizacji kontrastu
mean_after = np.mean(obraz_kontrast)
std_after = np.std(obraz_kontrast)

# Wyświetlenie statystyk
print(f'Przed normalizacją kontrastu:')
print(f'Średnia: {mean_before:.6f}')
print(f'Odchylenie standardowe: {std_before:.6f}')

print(f'\nPo normalizacji kontrastu:')
print(f'Średnia: {mean_after:.6f}')
print(f'Odchylenie standardowe: {std_after:.6f}')


data = np.load('torus_korekta_kontrast.npy')
plt.imshow(data, cmap='gray')
plt.axis('off')
plt.hist(data.flatten(), bins=256, color='blue', alpha=0.7)
plt.title('Histogram obrazu po normalizacji kontrastu')
plt.xlabel('Wartość piksela')
plt.ylabel('Liczba pikseli')
plt.savefig('histogram_kontrast.png')

