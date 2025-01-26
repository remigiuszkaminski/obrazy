#podpunkt a 
from PIL import Image
import numpy as np


# input_image_path  = "AlbertEinstein-modified.png"
# extracted_path    = "extracted.png"


# image = Image.open(input_image_path).convert("RGB")
# img_data = np.array(image)


# lsb_data = img_data & 1



# lsb_sum = np.sum(lsb_data, axis=2)
# lsb_gray = (lsb_sum * 85).astype(np.uint8)  # 0,85,170,255


# extracted_img = Image.fromarray(lsb_gray, mode="L")
# extracted_img.save(extracted_path)
# print(f"Zapisano wyekstrahowany obraz (LSB) w pliku: {extracted_path}")


#podpunkt b


# host_image_path   = "AlbertEinstein-modified.png"
# secret_image_path = "secret.png"
# output_image_path = "stego.png"

# host_img   = Image.open(host_image_path).convert("RGB")
# secret_img = Image.open(secret_image_path).convert("RGB")

# if host_img.size != secret_img.size:
#     secret_img = secret_img.resize(host_img.size)

# host_data   = np.array(host_img)
# secret_data = np.array(secret_img)


# host_data = host_data & 0b11111100

# secret_2bits = (secret_data >> 6) & 0b00000011

# stego_data = host_data | secret_2bits

# stego_img = Image.fromarray(stego_data.astype(np.uint8), "RGB")

# stego_img.save(output_image_path)
# print(f"Zapisano obraz z ukrytym 'secret.png' jako: {output_image_path}")


#odzyskanie obrazu z b


stego_image_path     = "stego.png"       # Obraz zawierający ukryty "secret"
extracted_image_path = "extracted2.png"   # Tu zapiszemy wyodrębniony obraz


stego_img = Image.open(stego_image_path).convert("RGB")
stego_data = np.array(stego_img)


extracted_2bits = stego_data & 0b00000011  

secret_data = extracted_2bits << 6  

secret_img = Image.fromarray(secret_data.astype(np.uint8), "RGB")
secret_img.save(extracted_image_path)
print(f"Odzyskany obraz zapisano jako: {extracted_image_path}")
