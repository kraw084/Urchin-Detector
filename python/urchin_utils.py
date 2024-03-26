import os
import sys
import torch
import csv
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
from PIL import Image

#Constants that can be used across files
CSV_PATH = os.path.abspath("data/csvs/High_conf_clipped_dataset_V3.csv")
DATASET_YAML_PATH = os.path.abspath("data/datasets/full_dataset_v3/datasetV3.yaml")
WEIGHTS_PATH = os.path.abspath("models/yolov5m-highRes-ro/weights/best.pt")

NUM_TO_LABEL = ["Evechinus chloroticus","Centrostephanus rodgersii"]

def dataset_by_id(csv_path=CSV_PATH):
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)
    dict = {int(row["id"]):row for row in reader}
    csv_file.close()
    return dict


def id_from_im_name(im_name):
    if "\\" in im_name: im_name = im_name.split("\\")[-1].strip("\n")
    if "/" in im_name: im_name = im_name.split("/")[-1].strip("\n")
    return int(im_name.split(".")[0][2:])


def draw_bbox(ax, bbox, im, using_alt_colours, correct, missed):
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



    top_left_point = (x_center - box_width/2, y_center - box_height/2)

    if not using_alt_colours:
        #colouring by class
        colours = {"Evechinus chloroticus": "#e2ed4a", "Centrostephanus rodgersii": "#cc1818"}
        col = colours[label]
    elif (missed is None and correct) or (correct is None and not missed):
        #green if pred is correct
        col = "#58f23d"
    else:
        col = "#cc1818"


    box_patch = patches.Rectangle(top_left_point, box_width, box_height, edgecolor=col, linewidth=2, facecolor='none')
    ax.add_patch(box_patch)

    text = f"{label.split(' ')[0]} - {round(confidence, 2)}{' - F' if flagged else ''}"
    text_bbox_props = dict(pad=0.2, fc=col, edgecolor='None')
    ax.text(top_left_point[0], top_left_point[1], text, fontsize=7, bbox=text_bbox_props, c="black", family="sans-serif")


def draw_bboxes(ax, bboxes, im, correct=None, boxes_missed=None):
    """draws all the boxes of a single image"""
    using_alt_colours = (not correct is None) or (not boxes_missed is None)

    if isinstance(bboxes, pd.DataFrame) and not bboxes.empty:
        i = 0
        for _, bbox in bboxes.iterrows():
            box_correct = correct[i] if not correct is None else None
            box_not_predicted = boxes_missed[i] if not boxes_missed is None else None
            draw_bbox(ax, bbox, im, using_alt_colours, correct=box_correct, missed=box_not_predicted)
            i += 1

    if isinstance(bboxes, list) and bboxes:
        for i, bbox in enumerate(bboxes):
            box_correct = correct[i] if not correct is None else None
            box_not_predicted = boxes_missed[i] if not boxes_missed is None else None
            draw_bbox(ax, bbox, im, using_alt_colours, correct=box_correct, missed=box_not_predicted)


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


def load_model(weights_path=WEIGHTS_PATH, cuda=True):
    """Load and return a yolo model"""
    model = torch.hub.load("yolov5", "custom", path=weights_path, source="local")
    model.cuda() if cuda else model.cpu()
    return model


class UrchinDetector:
    """Wrapper class for the yolov5 model"""
    def __init__(self, weight_path=WEIGHTS_PATH, conf=0.45, iou=0.6, img_size=1280, cuda=None):
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not cuda is None else torch.cuda.is_available()

        self.model = load_model(self.weight_path, self.cuda)
        self.model.conf = self.conf
        self.model.iou = self.iou

    def update_parameters(self, conf=0.45, iou=0.6):
        self.conf = conf
        self.model.conf = conf
        self.iou = iou
        self.model.iou = iou

    def predict(self, im):
        return self.model(im, size = self.img_size)

    def predict_batch(self, ims):
        return [self.predict(im) for im in ims]
    
    def pred_generator(self, ims):
        for im in ims:
            pred = self.predict(im)
            yield pred

    def __call__(self, im):
        return self.predict(im)


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


def annotate_images(model, image_folder, dest_folder):
    image_paths = os.listdir(image_folder)

    for im_path in image_paths:
        preds = model(image_folder + "/" + im_path)
        im = Image.open(image_folder + "/" + im_path)
        fig=plt.figure(figsize = (24, 12))

        plt.imshow(im)
        draw_bboxes(fig.axes[0], preds.pandas().xywh[0], im)
        plt.axis('off')
        plt.savefig(f'{dest_folder}/{im_path}.png', bbox_inches='tight', transparent=True, pad_inches=0)


def annotate_images2(model, image_folder, dest_folder):
    image_paths = os.listdir(image_folder)

    for im_path in image_paths:
        preds = model(image_folder + "/" + im_path)
        preds = preds.xyxy[0].cpu().numpy()

        im = cv2.imread(image_folder + "/" + im_path)
        for pred in preds:
            top_left = (round(pred[0]), round(pred[1]))
            bottom_right = (round(pred[2]), round(pred[3]))

            label = NUM_TO_LABEL[int(pred[5])]
            label = f"{label[0]}. {label.split()[1]}"

            colour = (0, 0, 255) if pred[5] else (0, 255, 255)

            im = cv2.rectangle(im, top_left, bottom_right, colour, 2)

            text_size = cv2.getTextSize(f"{label} - {pred[4]:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            padding = 1
            text_box_top_left = (top_left[0] - padding, top_left[1] - text_size[1] - padding - 5)
            text_box_bottom_right = (top_left[0] + text_size[0] + padding, top_left[1])
            im = cv2.rectangle(im, text_box_top_left, text_box_bottom_right, colour, -1)
            im = cv2.putText(im, f"{label} - {pred[4]:.2f}", (top_left[0], top_left[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(dest_folder + "/" + im_path, im)



model = UrchinDetector(cuda=False)
annotate_images2(model, "C:/Users/kelha/Desktop/test_ims", "C:/Users/kelha/Desktop/test_out")