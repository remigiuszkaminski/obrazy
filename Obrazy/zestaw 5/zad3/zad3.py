import numpy as np

# policzymy tutaj sobie na szybko wartosci korzystajac ze wzoru pierwiastek z (x^2 + y^2 + z^2)

def kalkulejtor(piksel):
    return np.sqrt(piksel[0]**2 + piksel[1]**2 + piksel[2]**2)

pixele = ([[125, 130, 240], [120, 250, 75], [235, 55, 130], [130, 130, 190], [255, 240 ,0], [35, 180, 75], [165, 75, 165], [195, 195, 195], [255, 175, 200]])
wyniki = []

for piksel in pixele:
    wyniki.append(kalkulejtor(piksel))

min_wynik = wyniki[np.argmin(wyniki)]
max_wynik = wyniki[np.argmax(wyniki)]
median_wynik = wyniki[np.argsort(wyniki)[len(wyniki) // 2]]

for i in range(len(pixele)):
    print(f"Piksel {pixele[i]} ma wartosc: {wyniki[i]}")
print(f"Minimalna wartosc: {min_wynik}")
print(f"Maksymalna wartosc: {max_wynik}")
print(f"Mediana: {median_wynik}")

#max -> numer 9 min _> numer 6 mediana -> numer 2