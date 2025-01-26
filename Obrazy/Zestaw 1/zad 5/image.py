from PIL import Image, ImageDraw
import random

input_path = "fikakowo.jpg"  
output_path = "zmieniony.png"
image = Image.open(input_path).convert("RGB")
draw = ImageDraw.Draw(image)

num_points = 100000 
radius = 10

width, height = image.size

for _ in range(num_points):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    
    pixel_color = image.getpixel((x, y))

    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=pixel_color
    )

image.save(output_path)
image.show()
