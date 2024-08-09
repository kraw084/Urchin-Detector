import os
import sys
import torch
import csv
import pandas as pd
import matplotlib.patches as patches
import cv2
import numpy as np
import importlib

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from YOLOX.tools.demo import Predictor
except ModuleNotFoundError:
    print("YOLOX not found")
    
#Constants that can be used across files
CSV_PATH = os.path.abspath("data/csvs/High_conf_clipped_dataset_V4.csv")
DATASET_YAML_PATH = os.path.abspath("data/datasets/full_dataset_v4/datasetV4.yaml")
WEIGHTS_PATH = os.path.abspath("models/yolov5m-highRes-ro/weights/best.pt")

NUM_TO_LABEL = ["Evechinus chloroticus","Centrostephanus rodgersii", "Heliocidaris erythrogramma"]
LABEL_TO_NUM = {label: i for i, label in enumerate(NUM_TO_LABEL)}
NUM_TO_COLOUR = [(74,237,226), (24,24,204), (3,140,252)]

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
        #box is numpy array
        x_center, y_center, box_width, box_height, confidence, label = bbox
        label = NUM_TO_LABEL[int(label)]

    top_left_point = (x_center - box_width/2, y_center - box_height/2)

    if not using_alt_colours:
        #colouring by class
        col = NUM_TO_COLOUR[label]
    elif (missed is None and correct) or (correct is None and not missed):
        #green if pred is correct
        col = "#58f23d"
    else:
        col = "#cc1818"


    box_patch = patches.Rectangle(top_left_point, box_width, box_height, edgecolor=col, linewidth=2, facecolor='none')
    ax.add_patch(box_patch)

    text = f"{label.split(' ')[0]} - {round(float(confidence), 2)}{' - F' if flagged else ''}"
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


def xywh_to_xyxy(box):
    """Converts an xywh box to xyxy"""
    x, y, w, h = box[:4]
    return np.array([x - w//2, y - h//2, x + w//2, y + h//2, box[4], box[5]])


class UrchinDetector_YoloV5:
    """Wrapper class for the yolov5 model"""
    def __init__(self, weight_path=WEIGHTS_PATH, conf=0.45, iou=0.6, img_size=1280, cuda=None, classes=NUM_TO_LABEL, plat_scaling = False):
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not (cuda is None) else torch.cuda.is_available()
        self.scaling = plat_scaling

        self.model = load_model(self.weight_path, self.cuda)
        self.model.conf = self.conf
        self.model.iou = self.iou
        
        self.classes = classes

    def update_parameters(self, conf=0.45, iou=0.6):
        self.conf = conf
        self.model.conf = conf
        self.iou = iou
        self.model.iou = iou

    def predict(self, im):
        results = self.model(im, size = self.img_size)
        if self.scaling:
            with torch.inference_mode():
                for pred in results.pred[0]:
                    pred[4] = plat_scaling(pred[4])
            results.__init__(results.ims, pred=results.pred, files=results.files, times=results.times, names=results.names, shape=results.s)
        return results

    def __call__(self, im):
        return self.xywhcl(im)
    
    def xywhcl(self, im):
        pred = self.predict(im).xywh[0].cpu().numpy()
        return [box for box in pred]
    
class UrchinDetector_YOLOX:
    """Wrapper class for the yolov5 model"""
    def __init__(self, weight_path=WEIGHTS_PATH, conf=0.2, iou=0.6, img_size=1280, cuda=None, exp_file_name="yolox_urchin_m", classes=NUM_TO_LABEL):
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not cuda is None else torch.cuda.is_available()

        yolox_exp_module = importlib.import_module(f"yolox.exp.custom.{exp_file_name}")
        exp_class = getattr(yolox_exp_module, "Exp")
        self.exp = exp_class()
        self.exp.test_conf = self.conf
        self.exp.nmsthre = self.iou
        self.exp.input_size = (img_size, img_size)
        self.exp.test_size = self.exp.input_size

        model = self.exp.get_model()
        ckpt_file = self.weight_path
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        device = "gpu" if self.cuda else "cpu"
        if device == "gpu":
            model.cuda()
        model.eval()

        self.model = Predictor(model, self.exp, NUM_TO_LABEL, device=device, legacy=False)
        self.update_parameters(self.conf, self.iou)
        self.model.test_size = self.exp.input_size
        
        self.classes = classes

    def update_parameters(self, conf=0.2, iou=0.6):
        self.conf = conf
        self.exp.test_conf = conf
        self.model.confthre = conf

        self.iou = iou
        self.exp.nmsthre = iou
        self.model.nmsthre = iou

    def predict(self, im):
        im = cv2.imread(im)
        pred = self.model.inference(im)[0][0]
        if pred is None:
            return []
        pred = pred.cpu().numpy()

        im_size_ratio = min(self.exp.test_size[0] / im.shape[0], self.exp.test_size[1] / im.shape[1])
        pred[:, :4] = pred[:, :4] / im_size_ratio
        
        return pred

    def __call__(self, im):
        return self.xywhcl(im)
    
    def xywhcl(self, im):
        pred = self.predict(im)
        formatted_pred = []
        for bbox in pred:            
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            conf = bbox[4] * bbox[5]
            label = bbox[6]
            formatted_pred.append(np.array([x_center, y_center, w, h, conf, label]))

        return formatted_pred


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


def filter_txt(txt_path, txt_output_name, var_name, exclude=None):
    if exclude is None: exclude = []
    im_paths = process_images_input(txt_path)
    dataset = dataset_by_id()
    kept_images = []
    for im in im_paths:
        id = id_from_im_name(im)
        data_row = dataset[id]

        if data_row[var_name] not in exclude: kept_images.append(im)

    f = open(txt_output_name, "w")
    f.write("\n".join(kept_images))
    f.close()


def plat_scaling(x):
    #Platt scaling function for highres-ro v3 model
    cubic = -7.3848* x**3 +13.5284 * x**2 -6.2952 *x + 1.0895
    linear = 0.566 * x + 0.027

    return cubic if x >=0.45 else linear
    

def annotate_image(im, prediction, num_to_label, num_to_colour, draw_labels=True):
        """Draws xywhcl boxes onto a single image. Colours are BGR"""
        thickness = 2
        font_size = 0.75

        label_data = []
        for pred in prediction:
            top_left = (int(pred[0]) - int(pred[2])//2, int(pred[1]) - int(pred[3])//2)
            bottom_right = (top_left[0] + int(pred[2]), top_left[1] + int(pred[3]))
            label = num_to_label[int(pred[5])]
            label = f"{label[0]}. {label.split()[1]}"

            colour = num_to_colour[int(pred[5])]

            #Draw boudning box
            im = cv2.rectangle(im, top_left, bottom_right, colour, thickness)

            label_data.append((f"{label} - {float(pred[4]):.2f}", top_left, colour))

        #Draw text over boxes
        if draw_labels:
            for data in label_data:
                text_size = cv2.getTextSize(data[0], cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
                text_box_top_left = (data[1][0], data[1][1] - text_size[1])
                text_box_bottom_right = (data[1][0] + text_size[0], data[1][1])
                im = cv2.rectangle(im, text_box_top_left, text_box_bottom_right, data[2], -1)
                im = cv2.putText(im, data[0], data[1], cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness - 1, cv2.LINE_AA)


def annotate_preds_on_folder(model, input_folder, output_folder, draw_labels=True):
    for im_name in os.listdir(input_folder):
        preds = model.xywhcl(input_folder + "/" + im_name)
        im = cv2.imread(input_folder + "/" + im_name)
        annotate_image(im, preds, NUM_TO_LABEL, NUM_TO_COLOUR, draw_labels=draw_labels)
        cv2.imwrite(output_folder + "/" + im_name, im)