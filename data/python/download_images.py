import urllib.request
import csv
import os

def download_imgs(csv_path, image_dest_dir, limit = None, print_every_x = 40):
    """Use to download images from the dataset. Only images with new ids will be downloaded.
    Args:
        csv_path: path to the csv file of the dataset
        image_dest_dir: path to the directory where the images will be downloaded
        limit: maximum number of images to download
        print_every_x: print the progress every x images
    """
    
    #read dataset csv
    csv_file = open(csv_path)
    reader = list(csv.DictReader(csv_file))

    #determine the number of images to download
    totalNumOfImages = len(reader)
    print(f"Downloading {limit if limit else totalNumOfImages} of {totalNumOfImages}")

    for i, row in enumerate(reader):
        
        #print progress
        if (i + 1) % print_every_x == 0:
            print(f"{i + 1}/{limit if limit else totalNumOfImages}")
        
        #check if image already exists
        new_im_name = image_dest_dir + "/" + f"im{row['id']}.JPG"
        if os.path.exists(new_im_name): continue
        
        #download image
        urllib.request.urlretrieve(row["url"], new_im_name)
        if limit and i + 1 >= limit: break

    print("----- FINISHED -----")
    csv_file.close()

if __name__ == "__main__":
    download_imgs("data/csvs/High_conf_clipped_dataset_V5.csv", "data/images")