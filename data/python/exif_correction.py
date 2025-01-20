from PIL import Image
from PIL import ImageOps
import os
from tqdm import tqdm

def correct_exif(image_dir):
    """Rotates images with exif data and removes it from the file
    Args:
        image_dir: path to the directory where the images are stored
    """
    for im_path in tqdm(os.listdir(image_dir), desc="Correcting EXIF data", bar_format="{l_bar}{bar:30}{r_bar}"):
        #open image and read exif data
        im = Image.open(f"{image_dir}/{im_path}", formats=["JPEG"])
        exif_data = im.getexif()
        
        #check if image has exif rotation data
        if exif_data and exif_data[274] != 1:
            #rotate image acording to exif data and then strip it
            print(f"Removing EXIF data ({exif_data[274]}) from {im_path}")
            im = ImageOps.exif_transpose(im)
            im_data = list(im.getdata())
            new_im = Image.new(im.mode, im.size)
            new_im.putdata(im_data)

            new_im.save(f"{image_dir}/{im_path}")
        
    im.close()

if __name__ == "__main__":
    correct_exif("data/images")