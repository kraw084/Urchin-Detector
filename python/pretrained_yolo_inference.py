import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
import csv

"""Run images from the dataset through a pretrained Yolov5 model to see what objects are identified"""

#import and run pretrained (COCO) Yolov5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.conf = 0.1

csvPath = "data/csvs/Complete_urchin_dataset.csv"
csvFile = open(csvPath, "r")
reader = csv.DictReader(csvFile)

for row in reader:
    imagePath = f"data/images/im{row['id']}.jpg"
    results = model(imagePath)
    results = results.pandas().xywh[0]
    #print(results)

    #plot boxes on the image
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots()
    ax.imshow(img.imread(imagePath))

    for i, row in results.iterrows():
        x = row["xcenter"]
        y = row["ycenter"]
        w = row["width"]
        h = row["height"]
        conf = row["confidence"]
        name = row["name"]

        boundingBox = patches.Rectangle((x - w/2, y - h/2), w, h, edgecolor='red', linewidth=1, facecolor='none')
        ax.add_patch(boundingBox)

        label = f"{name} - {round(conf, 2)}"
        ax.text(x - w/2,  y - h/2, label, fontsize=6)

    plt.show()