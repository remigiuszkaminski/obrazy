from PIL import Image
import numpy as np

def sasiad_sredniainterpolacja(obraz, skala):
    szer, wys = obraz.size
    szer_wyj = int(szer * skala)
    wys_wyj = int(wys * skala)
    obraz_wyj = np.zeros((wys_wyj, szer_wyj, 3), dtype=np.uint8)
    obraz = np.array(obraz)

    for y in range(wys_wyj):
        for x in range(szer_wyj):
            x_wej = int(x * szer / szer_wyj)
            y_wej = int(y * wys / wys_wyj)
            
            x1 = x_wej
            x2 = min(x_wej + 1, szer - 1)
            y1 = y_wej
            y2 = min(y_wej + 1, wys - 1)

            sasiedzi = [obraz[y1, x1], obraz[y1, x2], obraz[y2, x1], obraz[y2, x2]]
            mocjasnosci = [0.299 * z[0] + 0.587 * z[1] + 0.114 * z[2] for z in sasiedzi]
            minmoc = sasiedzi[np.argmin(mocjasnosci)]
            maxmoc = sasiedzi[np.argmax(mocjasnosci)]
            srednia = (minmoc.astype(np.int32) + maxmoc.astype(np.int32)) // 2
            obraz_wyj[y, x] = srednia.astype(np.uint8)
            
    return Image.fromarray(obraz_wyj)
                
            

    


obraz1 = Image.open("potworek.png").convert("RGB")
obraz1skalowany = sasiad_sredniainterpolacja(obraz1, 6)
obraz1skalowany.save("potworek_sredniainterpolacja.png")
