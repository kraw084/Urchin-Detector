import urllib.request
import csv

def download_imgs(csv_path, image_dest_dir, limit = None, print_every_x = 40):
    """Use to download images from the dataset. Set the limit varaibles to control how many images are downloaded"""
    csv_file = open(csv_path)
    reader = list(csv.DictReader(csv_file))

    totalNumOfImages = len(reader)
    print(f"Downloading {limit if limit else totalNumOfImages} of {totalNumOfImages}")

    for i, row in enumerate(reader):
        urllib.request.urlretrieve(row["url"], image_dest_dir + "/" + f"im{row['id']}.JPG")
        if limit and i + 1 >= limit: break

        if (i + 1) % print_every_x == 0:
            print(f"{i + 1}/{limit if limit else totalNumOfImages}")

    print("----- FINISHED -----")
    csv_file.close()

if __name__ == "__main__":
    download_imgs("data/csvs/Complete_urchin_dataset_V3.csv", "data/images_v3")