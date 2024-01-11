from PIL import Image
from PIL import ImageOps
import os

dir_path = "data/images"

count = 0
for im_path in os.listdir(dir_path):
    im = Image.open(f"{dir_path}/{im_path}", formats=["JPEG"])
    exif_data = im.getexif()
    if exif_data:
        orientation = exif_data[274]
        if orientation != 1:
            count += 1
    im.close()
print(count)

for im_path in os.listdir(dir_path):
    im = Image.open(f"{dir_path}/{im_path}", formats=["JPEG"])
    exif_data = im.getexif()
    if exif_data and exif_data[274] != 1:
        im = ImageOps.exif_transpose(im)

        print(f"Removing EXIF data ({exif_data[274]}) for {im_path}")
        im_data = list(im.getdata())
        new_im = Image.new(im.mode, im.size)
        new_im.putdata(im_data)

        new_im.save(f"{dir_path}/{im_path}")
        
    im.close()

count = 0
