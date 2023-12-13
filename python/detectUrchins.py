import os
import csv
import ast
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches

def load_model(weights_path):
    model = torch.hub.load("yolov5", "custom", path=weights_path, source="local")
    model.cuda()
    return model

def detect(model, source):
    """Runs images through urchin detection model, returns results as a list of pandas dataframes
       source can be a single image path, list of image path or a directory path"""
    if not (isinstance(source, list) or os.path.isfile(source)):
        #source is a dir
        source = [os.path.join(source, imPath) for imPath in os.listdir(source)][:5]

    results = model(source if isinstance(source, list) else [source])
    results = [r.pandas().xywh[0] for r in results.tolist()]
    return results

def compare_to_gt(model, dir):
    csvFile = open("data/csvs/Complete_urchin_dataset.csv", "r")
    rows = list(csv.DictReader(csvFile))
    csvFile.close()

    colours = {"Evechinus chloroticus": "yellow", "Centrostephanus rodgersii": "red"}

    for imName in os.listdir(dir):
        id = int(imName.split(".")[0][2:])
        boxes = ast.literal_eval(rows[id]["boxes"])

        if not boxes: continue

        matplotlib.use('TkAgg')
        fig = plt.figure(figsize=(12, 6))
        im = img.imread(os.path.join(dir, imName))

        #plot ground truth boxes
        ax = fig.add_subplot(1, 2, 1)
        plt.title("Ground truth")
        plt.imshow(im)
        if boxes:
            h, w, _ = im.shape
            for box in boxes:
                centerPoint = (box[2] * w, box[3] * h)
                boxW = w * box[4]
                boxH = h * box[5]
                topleftPoint = (centerPoint[0] - 0.5 * boxW, centerPoint[1] - 0.5 * boxH)

                boundingBox = patches.Rectangle(topleftPoint, boxW, boxH, edgecolor=colours[box[0]], linewidth=2, facecolor='none')
                ax.add_patch(boundingBox)

                label = box[0].split(" ")[0] + " - " + str(round(box[1], 2))
                bbox_props = dict(pad=0.2, fc=colours[box[0]], edgecolor='None')
                ax.text(topleftPoint[0], topleftPoint[1], label, fontsize=7, bbox=bbox_props, c="black", family="sans-serif")

        #plot predicted boxes
        ax = fig.add_subplot(1, 2, 2)
        plt.title("Prediction")
        plt.imshow(im)
        prediction = detect(model, os.path.join(dir, imName))[0]
        for i, row in prediction.iterrows():
            x = row["xcenter"]
            y = row["ycenter"]
            w = row["width"]
            h = row["height"]
            conf = row["confidence"]
            name = row["name"]

            boundingBox = patches.Rectangle((x - w/2, y - h/2), w, h, edgecolor=colours[box[0]], linewidth=2, facecolor='none')
            ax.add_patch(boundingBox)

            label = f"{name.split(' ')[0]} - {round(conf, 2)}"
            bbox_props = dict(pad=0.2, fc=colours[box[0]], edgecolor='None')
            ax.text(x - w/2,  y - h/2, label, fontsize=7, bbox=bbox_props, c="black", family="sans-serif")

        plt.show()


if __name__ == "__main__":
    model_name = "yolov5s-fullDataset"
    weights_path = os.path.abspath(f"models/{model_name}/weights/best.pt")

    model = load_model(weights_path)
    compare_to_gt(model, "data/images")

    

