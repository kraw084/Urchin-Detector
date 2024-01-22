import ast
import torch
from PIL import Image
import pandas as pd
import urchin_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

urchin_utils.project_sys_path()
from yolov5.val import process_batch
from yolov5.utils.metrics import ap_per_class


def get_metrics(model, image_set, img_size = 640, conf = 0.25, iou = 0.45, tta = False, cuda=True):
    """Computes metrics of provided image set. Based on the code from yolov5/val.py
        Arguments:
                model: model to get predictions from
                image_set: list of image paths
                img_size: the size images will be reduced to
                conf: prediction confidence threshold
                iou: nms iou threshold
                tta: set to true to enable test time augmentation
                cuda: enable cuda
       Returns: 
                precision, mean precision, recall, mean recall, f1 score, ap50, 
                map50, ap, and map of the provided images and ap_classes, a list
                with class indices that occurs in the given image set e.g. [], [0], [1], [0, 1]
        """

    device = torch.device("cuda") if cuda else torch.device("cpu")

    gt_boxes = [ast.literal_eval(row["boxes"]) for row in urchin_utils.get_dataset_rows()]
    pred_boxes = urchin_utils.batch_inference(model, image_set, 32, conf=conf, nms_iou_th=iou, img_size = img_size, tta=tta)
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

        if num_of_labels == 0:
            instance_counts[2] += 1
        else:
            for box in gt_boxes[id]:
                instance_counts[class_to_num[box[0]]] += 1

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


def metrics_by_var(model, images_txt, var_name, var_func = None, img_size = 640, cuda=True):
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
        metrics = get_metrics(model, splits[value], img_size = img_size, cuda=cuda)
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
    conf_values = [float(box[1]) for box in boxes]
    return any([val < 0.7 for val in conf_values])


def contains_low_prob_box_or_flagged(boxes):
    boxes = ast.literal_eval(boxes)
    conf_values = [float(box[1]) for box in boxes]
    return any([val < 0.7 for val in conf_values]) or any([box[6] for box in boxes])


def compare_models(weights_paths, images_txt, cuda=True, conf_values = None, iou_values = None):
    """Used to compare models by getting and printing metrics on each
       Arguments:
            weights_path: list of file paths to weight.pt files of the models to be compared
            images_txt: a txt file of image paths
            cuda: run the model on gpu
            conf_values: list of confidence values to run each model on (should have 1 number of each model)
            iou_values: list of iou thresholds to run each model on (should have 1 number of each model)
    """
    f = open(images_txt, "r")
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    if conf_values is None: conf_values = [0.25] * len(image_paths)
    if iou_values is None: iou_values = [0.45] * len(image_paths)

    prev_weight_path = None
    prev_model = None
    for i, weights_path in enumerate(weights_paths):
        if weights_path == prev_weight_path:
            model = prev_model
        else:
            model = urchin_utils.load_model(weights_path, cuda, verbose=False)
            prev_weight_path = weights_path
            prev_model = model

        metrics = get_metrics(model, image_paths, cuda=cuda, conf=conf_values[i], iou=iou_values[i], tta = True)

        print("------------------------------------------------")
        print(f"Model: {weights_path}\n")
        print_metrics(*metrics)

    print("------------------------------------------------")
    print("FINISHED")


def train_val_metrics(model, dataset_path, limit = None):
    """Print the metrics from the training and validation set, userful for detecting overfitting"""
    for set_name in ("train", "val"):
        f = open(f"{dataset_path}/{set_name}.txt", "r")
        image_paths = [line.strip("\n") for line in f.readlines()]
        f.close()

        if limit:
            random.shuffle(image_paths)
            image_paths = image_paths[:limit]

        print(f"Metrics on {set_name} set ({len(image_paths)} images) --------------------------------------")
        metrics = get_metrics(model, image_paths)
        print_metrics(*metrics)


def compare_to_gt(model, txt_of_im_paths, label = "urchin", conf = 0.25, save_path = False, limit = None, filter_var = None, filter_func = None):
    """Creates figures to visually compare model predictions to the actual labels
        model: yolo model to run
        txt_of_im_paths: path of a txt file containing image paths
        label: "all", "empty", "urchin", "kina", "centro", used to filter what kind of images are compared
        save_path: if this is a file path, figures will be saved instead of shown
        limit: number of figures to show/save, leave as none for no limit
        filter_var: csv var to be passed as input to filter function
        filter_func: function to be used to filter images, return false to skip an image
    """
    if label not in ("all", "empty", "urchin", "kina", "centro"):
        raise ValueError(f'label must be in {("all", "empty", "urchin", "kina", "centro")}')

    rows = urchin_utils.get_dataset_rows()

    txt_file = open(txt_of_im_paths, "r")
    im_paths = txt_file.readlines()
    filtered_paths = []

    for path in im_paths:
        id = urchin_utils.id_from_im_name(path)
        if filter_var and filter_func and not filter_func(rows[id][filter_var]): continue

        boxes = ast.literal_eval(rows[id]["boxes"])

        if label == "empty":
            if boxes: continue
        elif label == "urchin":
            if not boxes: continue
        elif label == "kina":
            if not boxes or boxes[0][0] == "Centrostephanus rodgersii": continue
        elif label == "centro":
            if not boxes or boxes[0][0] == "Evechinus chloroticus": continue

        filtered_paths.append(path)
    

    for i, im_path in enumerate(filtered_paths):
        id = urchin_utils.id_from_im_name(im_path)
        boxes = ast.literal_eval(rows[id]["boxes"])
        
        matplotlib.use('TkAgg')
        fig, axes = plt.subplots(1, 2, figsize = (14, 6))
        fig.suptitle(f"{im_path}\n{i + 1}/{len(filtered_paths)}")

        #format image meta data to display at the bottom of the fig
        row_dict = rows[id]
        row_dict.pop("boxes", None)
        row_dict.pop("url", None)
        row_dict.pop("id", None)
        text = str(row_dict)[1:-1].replace("'", "").split(",")
        text = f"{'    '.join(text[:3])} \n {'    '.join(text[3:])}"
        fig.text(0.5, 0.05, text, ha='center', fontsize=10)

        im = Image.open(im_path.strip("\n"), formats=["JPEG"])
        #deal with EXIF rotation
        #exif_data = im.getexif()
        #if exif_data:
        #    orientation = exif_data[274]
        #    rotations = {3: 180, 6: 270, 8: 90}
        #    if orientation in rotations:
        #        im = im.rotate(rotations[orientation])

        #plot ground truth boxes
        ax = axes[0]
        ax.set_title(f"Ground truth ({len(boxes)})")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        urchin_utils.draw_bboxes(ax, boxes, im)
            
        #plot predicted boxes
        prediction = urchin_utils.batch_inference(model, [im_path.strip("\n")], conf=conf)[0].pandas().xywh[0]
        ax = axes[1]
        ax.set_title(f"Prediction ({len(prediction)})")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        urchin_utils.draw_bboxes(ax, prediction, im)

        if not save_path:
            plt.show()
        else:
            fig.savefig(f"{save_path}/fig-{id}.png", format="png", bbox_inches='tight')

        if limit:
            limit -= 1
            if limit <= 0: break


def urchin_count_stats(model, images_txt):
    f = open(images_txt, "r")
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    rows = urchin_utils.get_dataset_rows()
    preds = urchin_utils.batch_inference(model, image_paths, 32)

    contains_urchin_correct = 0
    count_errors = []
    min_err_id = 0
    max_err_id = 0
    for im_path, pred in zip(image_paths, preds):
        id = urchin_utils.id_from_im_name(im_path)
        boxes = ast.literal_eval(rows[id]["boxes"])
        num_of_pred_boxes = len(pred.pandas().xyxy[0])
        if bool(boxes) == bool(num_of_pred_boxes): contains_urchin_correct += 1
        error = num_of_pred_boxes - len(boxes)
        count_errors.append(error)

        if max(count_errors) == error: max_err_id = id
        if min(count_errors) == error: min_err_id = id


    print(f"Proportion of images correctly classifed as containing urchins: {round(contains_urchin_correct/len(image_paths), 3)}")
    print("Count error stats:")
    print(f"mean: {np.mean(count_errors)}")
    print(f"median: {np.median(count_errors)}")
    print(f"std: {np.std(count_errors)}")
    print(f"min: {min(count_errors)} (id: {min_err_id})")
    print(f"max: {max(count_errors)} (id: {max_err_id})")

 
    matplotlib.use('TkAgg')
    freq = np.unique(count_errors, return_counts=True)
    points = []
    for value, count in zip(freq[0], freq[1]):
        points += [(value, i) for i in range(1, count + 1)]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, y, facecolors='none', edgecolors='black')
    ax.set_xticks(np.arange(min(count_errors), max(count_errors) + 1))
    plt.title("Urchin count errors")
    plt.xlabel("Count error (num of preds - num of true boxes)")
    plt.ylabel("Number of images")
    plt.show()


if __name__ == "__main__":
    weight_path = "models/yolov5s-reducedOverfitting/weights/last.pt"
    txt = "data/datasets/full_dataset_v3/val.txt"

    model = urchin_utils.load_model(weight_path, False)
    #model = urchin_utils.load_model("models/yolov5s-highConfNoFlagBoxes/weights/last.pt", cuda=False)

    #urchin_count_stats(model, txt)

    #metrics_by_var(model, "data/datasets/full_dataset_v3/val.txt", var_name="boxes", var_func=contains_low_prob_box, cuda=False)
    #metrics_by_var(model, "data/datasets/full_dataset_v3/val.txt", var_name="flagged", cuda=False)
    #metrics_by_var(model, "data/datasets/full_dataset_v3/val.txt", var_name="boxes", var_func=contains_low_prob_box_or_flagged, cuda=False)
    
    compare_to_gt(model, txt, "all", conf=0.4, filter_var= "campaign", filter_func= lambda x: x == "2019-Sydney")
 
    #compare_models(["models/yolov5s-reducedOverfitting/weights/last.pt"], txt, cuda=False)

    #train_val_metrics(model, "data/datasets/full_dataset_v3", 400)




