import urllib.request
import csv
import os

def download_imgs(csv_path, image_dest_dir, limit = None, print_every_x = 40):
    """Use to download images from the dataset. Set the limit varaibles to control how many images are downloaded"""
    csv_file = open(csv_path)
    reader = list(csv.DictReader(csv_file))

    totalNumOfImages = len(reader)
    print(f"Downloading {limit if limit else totalNumOfImages} of {totalNumOfImages}")

    for i, row in enumerate(reader):
        if (i + 1) % print_every_x == 0:
            print(f"{i + 1}/{limit if limit else totalNumOfImages}")
        
        new_im_name = image_dest_dir + "/" + f"im{row['id']}.JPG"
        if os.path.exists(new_im_name): continue
        
        urllib.request.urlretrieve(row["url"], new_im_name)
        if limit and i + 1 >= limit: break



    print("----- FINISHED -----")
    csv_file.close()

if __name__ == "__main__":
    download_imgs("data/csvs/High_conf_clipped_dataset_V5.csv", "data/images")