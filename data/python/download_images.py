import urllib.request
import csv
import os
from tqdm import tqdm

def download_imgs(csv_path, image_dest_dir):
    """Use to download images from the dataset. Only images with new ids will be downloaded.
    Args:
        csv_path: path to the csv file of the dataset
        image_dest_dir: path to the directory where the images will be downloaded
    """
    
    #read dataset csv
    csv_file = open(csv_path)
    reader = list(csv.DictReader(csv_file))
    
    #create the image directory if it doesn't exist
    if not os.path.exists(image_dest_dir): os.makedirs(image_dest_dir)

    #determine the number of images to download
    for row in tqdm(reader, desc="Downloading images", bar_format="{l_bar}{bar:30}{r_bar}"): 
        #print progress

        #check if image already exists
        new_im_name = image_dest_dir + "/" + f"im{row['id']}.JPG"
        if os.path.exists(new_im_name): continue
        
        #download image
        urllib.request.urlretrieve(row["url"], new_im_name)
        
    print("----- FINISHED -----")
    csv_file.close()

if __name__ == "__main__":
    download_imgs("data/csvs/High_conf_clipped_dataset_V5.csv", "data/images")