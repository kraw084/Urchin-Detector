from PIL import Image
from PIL import ImageOps
import os

def remove_exif_data(dir_path):
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


if __name__ == "__main__":
    dir_path = "data/images_helio"
    remove_exif_data(dir_path)