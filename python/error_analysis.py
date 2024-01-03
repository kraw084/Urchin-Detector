import ast
import torch
from PIL import Image
import pandas as pd
import urchin_utils

urchin_utils.project_sys_path()
from yolov5.val import process_batch
from yolov5.utils.metrics import ap_per_class

def get_metrics(model, image_set, img_size = 640, cuda=True):
    """Computes metrics of provided image set. Based on the code from yolov5/val.py.
       Arguments:
                model: model to get predictions from
                image_set: list of image paths
       Returns: 
                precision, mean precision, recall, mean recall, f1 score, ap50, 
                map50, ap, and map of the provided images and ap_classes, a list
                with class indices that occurs in the given image set e.g. [], [0], [1], [0, 1]
        """

    device = torch.device("cuda") if cuda else torch.device("cpu")

    gt_boxes = [ast.literal_eval(row["boxes"]) for row in urchin_utils.get_dataset_rows()]
    pred_boxes = urchin_utils.batch_inference(model, image_set, 32, img_size = img_size)
    class_to_num = {"Evechinus chloroticus": 0, "Centrostephanus rodgersii": 1}
    instance_counts = [0, 0, 0] #num of kina boxes, num of centro boxes, num of empty images

    num_iou_vals = 10
    iou_vals = torch.linspace(0.5, 0.95, num_iou_vals, device=device)
    stats = [] #(num correct, confidence, predicated classes, target classes)

    #get the relavent stats for each images predictions
    for im_path, pred in zip(image_set, pred_boxes):
        id = urchin_utils.id_from_im_name(im_path)
        num_of_labels = len(gt_boxes[id])
        num_of_preds = pred.xyxy[0].shape[0]

        if num_of_labels == 0: instance_counts[2] += 1

        target_classes = [class_to_num[box[0]] for box in gt_boxes[id]]
        correct = torch.zeros(num_of_preds, num_iou_vals, dtype=torch.bool, device=device)

        if num_of_preds == 0:
            if num_of_labels:
                stats.append((correct, torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor(target_classes, device=device)))
            continue

        if num_of_labels:
            im = Image.open(im_path)
            w, h = im.size

            labels = torch.zeros((num_of_labels, 5), device=device) #(class, x1, y1, x2, y2)
            for i, box in enumerate(gt_boxes[id]):
                #convert csv label to xyxy label as required by process_batch()
                x_center = box[2] * w
                y_center = box[3] * h
                box_width = box[4] * w
                box_height = box[5] * h

                labels[i][0] = class_to_num[box[0]]
                labels[i][1] = x_center - box_width/2
                labels[i][2] = y_center - box_height/2
                labels[i][3] = x_center + box_width/2
                labels[i][4] = y_center + box_height/2

                instance_counts[class_to_num[box[0]]] += 1

            #get true positive counts at different iou thresholds
            correct = process_batch(pred.xyxy[0], labels, iou_vals)

        stats.append((correct, pred.xyxy[0][:, 4], pred.xyxy[0][:, 5], torch.tensor(target_classes, device=device)))
    
    #get metrics and calc averages
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    num_to_class = {0:"Evechinus chloroticus", 1:"Centrostephanus rodgersii"}
    true_pos, false_pos, precision, recall, f1, ap_across_iou_th, ap_class = ap_per_class(*stats, names=num_to_class)
    ap50, ap = ap_across_iou_th[:, 0], ap_across_iou_th.mean(1)
    mean_precision, mean_recall, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()

    return precision, mean_precision, recall, mean_recall, f1, ap50, map50, ap, map, ap_class, instance_counts


def print_metrics(precision, mean_precision, recall, mean_recall, f1, ap50, map50, ap, map, classes, counts):
    """Print metrics in a table, seperated by class"""
    headers = ["Class", "Instances", "P", "R", "F1", "ap50", "ap"]
    df = pd.DataFrame(columns=headers)
    #parameters may only have stats for 1 class so add those rows seperatly 
    for c in classes:
        if c == 0: 
            label = "Evechinus chloroticus"
            count = counts[0]
        if c == 1:
            label = "Centrostephanus rodgersii"
            count = counts[1]
            if len(classes) == 1: c = 0

        df_row = [label, count, precision[c], recall[c], f1[c], ap50[c], ap[c]]
        df.loc[len(df)] = df_row
 
    if len(classes) == 2: #if the stats include both classes and add an average row
        df_row = ["Avg", "-", mean_precision, mean_recall, (f1[0] + f1[1])/2, map50, map]
        df.loc[len(df)] = df_row
    
    #df formatting and printing
    df.set_index("Class", inplace=True)
    df = df.round(3)
    print(df.to_string(index=True, index_names=False))
    print(f"{counts[2]} images with no labels")


def metrics_by_var(model, images_txt, var_name, var_func = None, img_size = 640):
    """Seperate the given dataset by the chosen variable and get metrics on each partition
       Arguments:
            model: model to run
            image_txt: txt file of image paths
            var_name: csv header name to filter by
            var_func: optional func to run the value of var_name through, useful for discretization"""
    
    #read image paths from txt file
    f = open(images_txt, "r")
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    dataset_rows = urchin_utils.get_dataset_rows()
    splits = {}

    #split data by var_name
    for image_path in image_paths:
        id = urchin_utils.id_from_im_name(image_path)
        value = dataset_rows[id][var_name]

        if var_func: value = var_func(value)

        if value in splits:
            splits[value].append(image_path)
        else:
            splits[value] = [image_path]

    #print header
    print("----------------------------------------------------")
    print(f"Getting metrics by {var_name}{' and ' + var_func.__name__ if var_func else ''}")
    print(f"Values: {', '.join([str(k) for k in sorted(splits.keys())])}")
    print("----------------------------------------------------")

    #print metrics for each split
    for value in sorted(splits):
        print(f"Metrics for {value} ({len(splits[value])} images):\n")
        metrics = get_metrics(model, splits[value], img_size = img_size)
        print_metrics(*metrics)
        print("----------------------------------------------------")
    print("FINISHED")


def depth_discretization(depth):
    depth = float(depth)
    minVal = 0
    maxVal = 60
    step = 2

    for i in range(minVal, maxVal, step):
        if depth >= i and depth < i + step: return i


def contains_low_prob_box(boxes):
    boxes = ast.literal_eval(boxes)
    conf_values = [int(box[1]) for box in boxes]
    return any([val < 1 for val in conf_values])


def compare_models(weights_paths, images_txt, cuda=True):
    """Used to compare models by getting and printing metrics on each
       Arguments:
            weights_path: list of file paths to weight.pt files of the models to be compared
            images_txt: a txt file of image paths
    """
    f = open(images_txt, "r")
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    for weights_path in weights_paths:
        model = urchin_utils.load_model(weights_path, cuda, verbose=False)
        metrics = get_metrics(model, image_paths, cuda)

        print("------------------------------------------------")
        print(f"Model: {weights_path}\n")
        print_metrics(*metrics)

    print("------------------------------------------------")
    print("FINISHED")

if __name__ == "__main__":
    #model = urchin_utils.load_model(urchin_utils.WEIGHTS_PATH, True)
    #metrics_by_var(model, "data/datasets/full_dataset_v2/val.txt", "depth", depth_discretization)

    f = open("data/datasets/full_dataset_v2/val.txt", "r")
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    model = urchin_utils.load_model("models/yolov5s-fullDatasetV2-new/weights/best.pt", True)
    metrics = get_metrics(model, image_paths)
    print_metrics(*metrics)
