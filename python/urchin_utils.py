import os
import sys
import math
import torch
import csv
import pandas as pd
import matplotlib.patches as patches

#Constants that can be used across files
#CSV_PATH = os.path.abspath("data/csvs/Complete_urchin_dataset_V3.csv")
CSV_PATH = os.path.abspath("data/csvs/clipped_dataset.csv")
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


def draw_bbox(ax, bbox, im, correct, missed):
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
    if correct:
        col = "#58f23d"
    elif missed:
        col = "#e88c13"
    else:
        col = colours[label]

    box_patch = patches.Rectangle(top_left_point, box_width, box_height, edgecolor=col, linewidth=2, facecolor='none')
    ax.add_patch(box_patch)

    text = f"{label.split(' ')[0]} - {round(confidence, 2)}{' - F' if flagged else ''}"
    text_bbox_props = dict(pad=0.2, fc=col, edgecolor='None')
    ax.text(top_left_point[0], top_left_point[1], text, fontsize=7, bbox=text_bbox_props, c="black", family="sans-serif")


def draw_bboxes(ax, bboxes, im, correct=None, boxes_missed=None):
    """draws all the boxes of a single image"""
    if isinstance(bboxes, pd.DataFrame) and not bboxes.empty:
        i = 0
        for _, bbox in bboxes.iterrows():
            box_correct = correct[i] if not correct is None else False
            box_not_predicted = boxes_missed[i] if not boxes_missed is None else False
            draw_bbox(ax, bbox, im, correct=box_correct, missed=box_not_predicted)
            i += 1

    if isinstance(bboxes, list) and bboxes:
        for i, bbox in enumerate(bboxes):
            box_correct = correct[i] if not correct is None else False
            box_not_predicted = boxes_missed[i] if not boxes_missed is None else False
            draw_bbox(ax, bbox, im, correct=box_correct, missed=box_not_predicted)


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


def load_model(weights_path=WEIGHTS_PATH, cuda=True, verbose=True):
    """Load and return a yolo model"""
    model = torch.hub.load("yolov5", "custom", path=weights_path, source="local", _verbose=verbose)
    model.cuda() if cuda else model.cpu()
    return model



def batch_inference(model, image_set, batch_size = None, conf = 0.25, nms_iou_th = 0.45, img_size = 640, tta = False):
    """Processes images through the model in batchs to reduce memory usage
       Arguments:
            model: model object to use
            image_set: list of image paths
            batch_size: number of images to process at once, leave as none to use full image set as one batch
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
        images = image_set[start_index:end_index]
        batch_preds = model(images, size = img_size, augment=tta).tolist()
        preds += batch_preds   

    return preds


def read_txt(images_txt):
    f = open(images_txt, "r")
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    return image_paths


def process_images_input(images):
    return images if isinstance(images, list) else read_txt(images) 


def complement_image_set(images0, images1):
    """Returns all the image paths in images1 that are not in images0"""
    images0 = process_images_input(images0)
    images1 = process_images_input(images1)

    return [x for x in images1 if x not in images0]
