import urllib.request
import csv
import os.path

"""Use to download images from the dataset. Set the limit varaibles to control how many images are downloaded"""

csvFilePath = os.path.abspath("data/csvs/Complete_urchin_dataset.csv")
imageOutputPath = os.path.abspath("data/images")

file = open(csvFilePath)
reader = list(csv.DictReader(file))

limit = None #how many images to download, None to download all
totalNumOfImages = len(reader)
print(f"Downloading {limit if limit else totalNumOfImages} of {totalNumOfImages}")

for i, row in enumerate(reader):
    urllib.request.urlretrieve(row["url"], imageOutputPath + "/" + f"im{row['id']}.JPG")
    if limit and i + 1 >= limit: break

    if (i + 1) % 20 == 0:
        print(f"{i + 1}/{limit if limit else totalNumOfImages}")


print("----- FINISHED -----")
file.close()