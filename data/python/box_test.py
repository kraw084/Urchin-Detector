import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches

import csv
import random
import urllib.request
import ast

"""
This is for testing that the bounding boxes are being stored correctly and can be drawn properly.
This program will download the images it shows so make sure to delete them afterwards.
"""

NUM_IMAGES = 4

file = open("UOA_urchin_dataset.csv", "r")
reader = list(csv.DictReader(file))

indices = [random.randrange(len(reader)) for _ in range(NUM_IMAGES)]
images = []

for i in indices:
    url = reader[i]["url"]
    name = reader[i]["name"]
    urllib.request.urlretrieve(url, "test/" + name)
    images.append("test/" + name)


fig = plt.figure(figsize=(14, 7))
rows = 2
cols = 2

if not rows * cols == NUM_IMAGES: print("NUM_IMAGES doesnt match rows can cols")

for idx in range(NUM_IMAGES):
    i = indices[idx]
    imgName = images[idx]

    image = img.imread(imgName)
    ax = fig.add_subplot(rows, cols, idx + 1)
    plt.imshow(image)
    plt.axis("off")

    boxes = ast.literal_eval(reader[i]["boxes"])
    if boxes:
        plt.title("Urchins")
        h, w, _ = image.shape
        for box in boxes:
            centerPoint = (box[2] * w, box[3] * h)
            circle = patches.Circle(centerPoint, 8, edgecolor='red', linewidth=1, facecolor='none')
            ax.add_patch(circle)

            boxW = w * box[4]
            boxH = h * box[5]
            topleftPoint = (centerPoint[0] - 0.5 * boxW, centerPoint[1] - 0.5 * boxH)

            boundingBox = patches.Rectangle(topleftPoint, boxW, boxH, edgecolor='red', linewidth=1, facecolor='none')
            ax.add_patch(boundingBox)

            label = box[0] + " - " + str(round(box[1], 2))
            ax.text(topleftPoint[0], topleftPoint[1], label, fontsize=6)
    else:
        plt.title("No urchins")

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

