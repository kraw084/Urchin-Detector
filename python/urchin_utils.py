import os
import sys
import math
import torch
import csv
import cv2
from PIL import Image
import pandas as pd
import matplotlib.patches as patches

#Constants that can be used across files
CSV_PATH = os.path.abspath("data/csvs/Complete_urchin_dataset_V3.csv")
DATASET_YAML_PATH = os.path.abspath("data/datasets/full_dataset_v3/datasetV3.yaml")
WEIGHTS_PATH = os.path.abspath("models/yolov5s-reducedOverfitting/weights/last.pt")


def get_dataset_rows():
    csv_file = open(CSV_PATH, "r")
    reader = csv.DictReader(csv_file)
    rows = [row for row in reader]
    csv_file.close()
    return rows


def id_from_im_name(im_name):
    if "\\" in im_name: im_name = im_name.split("\\")[-1].strip("\n")
    if "/" in im_name: im_name = im_name.split("/")[-1].strip("\n")
    return int(im_name.split(".")[0][2:])


def draw_bbox(ax, bbox, im):
    """draws a bounding box on the provided matplotlib axis
       bbox can be a tuple from the csv of pandas df from model output"""
    flagged = None
    if isinstance(bbox, tuple):
        #ground truth box from csv
        im_width, im_height = im.size
        label = bbox[0]
        confidence = bbox[1]
        x_center = bbox[2] * im_width
        y_center = bbox[3] * im_height
        box_width = bbox[4] * im_width
        box_height = bbox[5] * im_height
        if len(bbox) >= 7: flagged = bbox[6]
    else: 
        #pandas pred box from model
        label = bbox["name"]
        confidence = bbox["confidence"]
        x_center = bbox["xcenter"]
        y_center = bbox["ycenter"]
        box_width = bbox["width"]
        box_height = bbox["height"]

    colours = {"Evechinus chloroticus": "#e2ed4a", "Centrostephanus rodgersii": "#cc1818"}

    top_left_point = (x_center - box_width/2, y_center - box_height/2)
    col = colours[label]
    box_patch = patches.Rectangle(top_left_point, box_width, box_height, edgecolor=col, linewidth=2, facecolor='none')
    ax.add_patch(box_patch)

    text = f"{label.split(' ')[0]} - {round(confidence, 2)}{' - F' if flagged else ''}"
    text_bbox_props = dict(pad=0.2, fc=col, edgecolor='None')
    ax.text(top_left_point[0], top_left_point[1], text, fontsize=7, bbox=text_bbox_props, c="black", family="sans-serif")


def draw_bboxes(ax, bboxes, im):
    """draws all the boxes of a single image"""
    if isinstance(bboxes, pd.DataFrame) and not bboxes.empty:
        for _, bbox in bboxes.iterrows():
            draw_bbox(ax, bbox, im)

    if isinstance(bboxes, list) and bboxes:
        for bbox in bboxes:
            draw_bbox(ax, bbox, im)


def check_cuda_availability():
    """Check if cuda is available and print relavent info"""
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def project_sys_path():
    """add the Urchin-Detector folder to the sys path so functions from yolo can be imported"""
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_dir)


def load_model(weights_path, cuda=True, verbose=True):
    """Load and return a yolo model"""
    model = torch.hub.load("yolov5", "custom", path=weights_path, source="local", _verbose=verbose)
    model.cuda() if cuda else model.cpu()
    return model


#project_sys_path()
#from yolov5.utils.augmentations import letterbox
import numpy as np

#letterbox function taken from yolov5.utils.augmentations
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def batch_inference(model, image_set, batch_size = None, conf = 0.25, nms_iou_th = 0.45, img_size = 640, tta = False, pad = False):
    """Processes images through the model in batchs to reduce memory usage
       Arguments:
            model: model object to use
            image_set: list of image paths, leave as none to use full image set as one batch
            batch_size: number of images to process at once
            conf: predictions with confidence less that this will be ignored
            nms_iou_th: iou threshold used for non-maximal supression
            img_size: size images will be rescaled to
       Returns:
            List of predictions (list of detection objects, one for each image)
       """
    if not batch_size: batch_size = len(image_set)

    model.conf = conf
    model.iou = nms_iou_th

    num_of_batches = math.ceil(len(image_set)/batch_size)
    preds = []
    for b in range(num_of_batches):
        start_index = b * batch_size
        end_index = start_index + batch_size

        if pad:
            images = [cv2.imread(im_path) for im_path in image_set[start_index:end_index]]
            images = [letterbox(im, img_size, auto=False) for im in images]
            paddingSize = [image[2] for image in images]
            images = [image[0] for image in images]
        else:
            images = image_set[start_index:end_index]

        batch_preds = model(images, size = img_size, augment=tta)
        preds += batch_preds.tolist()

        if pad: #covert padded predictions to be relative to the regular image
            adjusted_preds = []
            for pred, padSize in zip(preds, paddingSize):
                adjusted_pred = []
                boxes = pred.xywh
                if boxes[0].numel():
                    for box in boxes[0]:
                        x = ((box[0].item()) - padSize[0])/(img_size - 2 * padSize[0])
                        y = ((box[1].item()) - padSize[1])/(img_size - 2 * padSize[1])
                        w = (box[2].item())/(img_size - 2 * padSize[0])
                        h = (box[3].item())/(img_size - 2 * padSize[1])
                        c = box[4].item()

                        label = "Evechinus chloroticus" if not box[5].item() else "Centrostephanus rodgersii"
                        adjusted_pred.append((label, c, x, y, w, h))
                adjusted_preds.append(adjusted_pred)
            return adjusted_preds
        
    return preds

# ------------------ PADDING TEST MAKE SURE TO DELETE
image_paths = [f"data/images/im{i}.jpg" for i in range(6)]

model = load_model(WEIGHTS_PATH, False)
preds = batch_inference(model, image_paths, 32, pad = True)
reg_pred = batch_inference(model, image_paths, 32)

if True:
    import ast
    import matplotlib
    import matplotlib.pyplot as plt
    rows = get_dataset_rows()

    for i, im_path in enumerate(image_paths):
            id = id_from_im_name(im_path)
            boxes = ast.literal_eval(rows[id]["boxes"])
            
            matplotlib.use('TkAgg')
            fig, axes = plt.subplots(1, 3, figsize = (14, 6))

            im = Image.open(im_path)
            
            #plot ground truth boxes
            ax = axes[0]
            ax.set_title(f"Ground truth ({len(boxes)})")
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            draw_bboxes(ax, boxes, im)
                
            #plot predicted boxes
            prediction = preds[i]
            ax = axes[1]
            ax.set_title(f"Prediction ({len(prediction)})")
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            draw_bboxes(ax, prediction, im)

            prediction = reg_pred[i].pandas().xywh[0]
            ax = axes[2]
            ax.set_title(f"Prediction ({len(prediction)})")
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            draw_bboxes(ax, prediction, im)

            plt.show()