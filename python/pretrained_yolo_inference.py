import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
import python.model_utils as model_utils

"""Run images from the dataset through a pretrained Yolov5 model to see what objects are identified"""

#import and run pretrained (COCO) Yolov5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)



for row in model_utils.get_dataset_rows():
    image_path = f"data/images/im{row['id']}.jpg"
    results = model(image_path)
    results = results.pandas().xywh[0]

    #plot boxes on the image
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots()
    ax.imshow(img.imread(image_path))

    for i, row in results.iterrows():
        x = row["xcenter"]
        y = row["ycenter"]
        w = row["width"]
        h = row["height"]
        conf = row["confidence"]
        name = row["name"]

        bounding_box = patches.Rectangle((x - w/2, y - h/2), w, h, edgecolor='red', linewidth=1, facecolor='none')
        ax.add_patch(bounding_box)

        label = f"{name} - {round(conf, 2)}"
        ax.text(x - w/2,  y - h/2, label, fontsize=6)

    plt.show()