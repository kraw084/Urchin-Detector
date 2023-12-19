import ast
import torch
from PIL import Image
import pandas as pd
import urchin_utils

urchin_utils.project_sys_path()
from yolov5.val import process_batch
from yolov5.utils.metrics import ap_per_class

def get_metrics(model, image_set):
    """Computes metrics of provided image set
        model: model to get predictions from
        image_set: list of image paths"""

    gt_boxes = [ast.literal_eval(row["boxes"]) for row in urchin_utils.get_dataset_rows()]
    pred_boxes = model(image_set).tolist()
    class_to_num = {"Evechinus chloroticus": 0, "Centrostephanus rodgersii": 1}

    num_iou_vals = 10
    iou_vals = torch.linspace(0.5, 0.95, num_iou_vals)
    stats = [] #(num correct, confidence, predicated classes, target classes)

    #get the relavent stats for each images predictions
    for im_path, pred in zip(image_set, pred_boxes):
        id = urchin_utils.id_from_im_name(im_path)
        num_of_labels = len(gt_boxes[id])
        num_of_preds = len(pred)

        target_classes = [class_to_num[box[0]] for box in gt_boxes[id]]
        correct = torch.zeros(num_of_preds, num_iou_vals, dtype=torch.bool)

        if num_of_preds == 0 and num_of_labels:
            stats.append((correct, [], [], target_classes))
            continue

        if num_of_labels:
            im = Image.open(im_path)
            w, h = im.size

            labels = torch.zeros((num_of_labels, 5)) #(class, x1, y1, x2, y2)
            for i, box in enumerate(gt_boxes[id]):
                x_center = box[2] * w
                y_center = box[3] * h
                box_width = box[4] * w
                box_height = box[5] * h

                labels[i][0] = class_to_num[box[0]]
                labels[i][1] = x_center - box_width/2
                labels[i][2] = y_center - box_height/2
                labels[i][3] = x_center + box_width/2
                labels[i][4] = y_center + box_height/2

            correct = process_batch(pred.xyxy[0], labels, iou_vals)

        stats.append((correct, pred.xyxy[0][:, 4], pred.xyxy[0][:, 5], torch.tensor(target_classes)))
    
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]

    num_to_class = {0:"Evechinus chloroticus", 1:"Centrostephanus rodgersii"}
    true_pos, false_pos, precision, recall, f1, ap_across_iou_th, ap_class = ap_per_class(*stats, names=num_to_class)
    ap50, ap = ap_across_iou_th[:, 0], ap_across_iou_th.mean(1)
    mean_precision, mean_recall, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()

    return precision, mean_precision, recall, mean_recall, f1, ap50, map50, ap, map

def print_metrics(precision, mean_precision, recall, mean_recall, f1, ap50, map50, ap, map):
    cols = {"P": [precision[0], precision[1], mean_precision],
            "R": [recall[0], recall[1], mean_recall],
            "F1": [f1[0], f1[1], (f1[0] + f1[1])/2],
            "ap50": [ap50[0], ap[1], map50],
            "ap": [ap[0], ap[1], map]
            }
    
    df = pd.DataFrame(cols)
    row_headers = ["Evechinus chloroticus", "Centrostephanus rodgersii", "Avg"]
    df["Rows"] = row_headers
    df.set_index("Rows", inplace=True)
    df = df.round(3)

    print(df.to_string(index=True, index_names=False))

if __name__ == "__main__":
    model = urchin_utils.load_model(urchin_utils.WEIGHTS_PATH, False)
    metrics = get_metrics(model, 
                ["C:\\Users\\kelha\\Documents\\Uni\\Summer Research\\Urchin-Detector\\data\\images\\im81.JPG",
                 "C:\\Users\\kelha\\Documents\\Uni\\Summer Research\\Urchin-Detector\\data\\images\\im128.JPG",
                 "C:\\Users\\kelha\\Documents\\Uni\\Summer Research\\Urchin-Detector\\data\\images\\im2436.JPG",
                 "C:\\Users\\kelha\\Documents\\Uni\\Summer Research\\Urchin-Detector\\data\\images\\im3796.JPG"])

    print_metrics(*metrics)

