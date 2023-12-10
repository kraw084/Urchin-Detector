import urllib.request
import csv

"""Use to download images from the dataset. Set the limit varaibles to control how many images are downloaded"""

csvFilePath = "csvs/Complete_urchin_dataset.csv"
imageOutputPath = "images"

file = open(csvFilePath)
reader = list(csv.DictReader(file))

limit = 110 #how many images to download, None to download all
totalNumOfImages = len(reader)
print(f"Downloading {limit if limit else totalNumOfImages} of {totalNumOfImages}")

for i, row in enumerate(reader):
    urllib.request.urlretrieve(row["url"], imageOutputPath + "/" + str(i) + ".JPG")
    if limit and i + 1 >= limit: break

    if (i + 1) % 20 == 0:
        print(f"{i + 1}/{limit if limit else totalNumOfImages}")


print("----- FINISHED -----")
file.close()