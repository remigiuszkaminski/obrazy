from PIL import Image
import numpy as np

def sasiad_sredniaw_2najbliszyschwejsciowego(obraz, skala):
    szer, wys = obraz.size
    szer_wyj = int(szer * skala)
    wys_wyj = int(wys * skala)
    obraz_wyj = np.zeros((wys_wyj, szer_wyj, 3), dtype=np.uint8)
    obraz = np.array(obraz)

    for y in range(wys_wyj):
        for x in range(szer_wyj):
            x_wej = int(x * szer / szer_wyj)
            y_wej = int(y * wys / wys_wyj)
            sasiednepixele = []
            if x_wej < szer - 1:
                sasiednepixele.append(obraz[y_wej, x_wej + 1])
            if y_wej < wys - 1:
                sasiednepixele.append(obraz[y_wej + 1, x_wej])
            
            sasiednepixele.append(obraz[y_wej, x_wej])
            obraz_wyj[y, x] = np.mean(sasiednepixele, axis=0)

    return Image.fromarray(obraz_wyj)
                
            

    


obraz1 = Image.open("potworek.png").convert("RGB")
obraz1skalowany = sasiad_sredniaw_2najbliszyschwejsciowego(obraz1, 6)
obraz1skalowany.save("potworek_sasiad_sredniaw_2najbliszyschwejsciowego.png")
