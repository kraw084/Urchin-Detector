import ast
import torch
from PIL import Image
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import cv2

from urchin_utils import (dataset_by_id, UrchinDetector, UrchinDetector_YOLOX, process_images_input, 
                          project_sys_path, id_from_im_name, draw_bboxes,
                          filter_txt)

project_sys_path()
from yolov5.val import process_batch
from yolov5.utils.metrics import ap_per_class, box_iou

 
def correct_predictions(im_path, gt_box, pred, iou_vals = None, boxes_missed = False, cuda = True):
    """Determines what predictions are correct at different iou thresholds
        Arguments:
            im_path: image path
            gt_box: ground truth boxes from csv
            pred: model prediction
            boxes_missed: set to True to return an additional array that indicated if the ith gt box was missed
            iou_vals: array of iou thresholds
            cuda: enable cuda
        Returns:
            A Nx10 array where N is the number of predicted boxes. [n, i] is true if the nth box has the same class
            and iou greater than iou_vals[i] with the gt_box that it has the highest iou with.
    """
    if iou_vals is None: iou_vals = torch.linspace(0.5, 0.95, 10)
    iou_vals = iou_vals.cpu()
 
    im = Image.open(im_path)
    w, h = im.size
    class_to_num = {"Evechinus chloroticus": 0, "Centrostephanus rodgersii": 1}

    labels = torch.zeros((len(gt_box), 5)) #(class, x1, y1, x2, y2)

    for i, box in enumerate(gt_box):
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
    if not type(pred) is list:
        correct = process_batch(pred.xyxy[0].cpu(), labels, iou_vals)
    else:
        pred_xyxy = np.zeros((len(pred), 6))
        for i, bbox in enumerate(pred):
            x = bbox["xcenter"]
            y = bbox["ycenter"]
            w = bbox["width"]
            h = bbox["height"]
            pred_xyxy[i, :] = np.array([x - w//2, y - h//2, x + w//2, y + h//2, bbox["confidence"], class_to_num[bbox["name"]]])
        pred_xyxy = torch.from_numpy(pred_xyxy)

        correct = process_batch(pred_xyxy, labels, iou_vals)

    if boxes_missed:
        #find all the boxes where all the iou values are less than the threshold
        if not type(pred) is list:
            iou = box_iou(labels[:, 1:], pred.xyxy[0][:, :4])
        else:
            iou = box_iou(labels[:, 1:], pred_xyxy[:, :4])

        gt_box_missed = np.all(a=(iou < iou_vals[0]).numpy(force=True), axis=1)
        return correct, gt_box_missed
    
    return correct


def get_metrics(model, image_set, cuda=True, min_iou_val = 0.5, dataset_path=None):
    """Computes metrics of provided image set. Based on the code from yolov5/val.py
        Arguments:
                model: model to get predictions from
                image_set: list of image paths
                img_size: the size images will be reduced to
                cuda: enable cuda
                min_iou_val: the smallest iou value used when determine prediction correctness
       Returns: 
                precision, mean precision, recall, mean recall, f1 score, ap50, 
                map50, ap, and map of the provided images and ap_classes, a list
                with class indices that occurs in the given image set e.g. [], [0], [1], [0, 1]
        """
    
    if len(image_set) == 0: 
        return [0], 0, [0], 0, [0], [0], 0, [0], 0, [], [0, 0, 0]

    cuda = False
    device = torch.device("cuda") if cuda else torch.device("cpu")

    class_to_num = {"Evechinus chloroticus": 0, "Centrostephanus rodgersii": 1}
    instance_counts = [0, 0, 0] #num of kina boxes, num of centro boxes, num of empty images
    dataset = dataset_by_id() if dataset_path is None else dataset_by_id(dataset_path)
    num_iou_vals = 10
    iou_vals = torch.linspace(min_iou_val, 0.95, num_iou_vals, device=device)
    stats = [] #(num correct, confidence, predicated classes, target classes)

    #get the relavent stats for each images predictions
    for im_path in image_set:
        id = id_from_im_name(im_path)
        boxes = ast.literal_eval(dataset[id]["boxes"])
        pred = model(im_path)
        num_of_labels = len(boxes)

        if not type(pred) is list:
            num_of_preds = pred.xyxy[0].shape[0]
        else: 
            num_of_preds = len(pred)
            pred_xyxy = np.zeros((len(pred), 6))
            for i, bbox in enumerate(pred):
                x = bbox["xcenter"]
                y = bbox["ycenter"]
                w = bbox["width"]
                h = bbox["height"]
                pred_xyxy[i, :] = np.array([x - w//2, y - h//2, x + w//2, y + h//2, bbox["confidence"], class_to_num[bbox["name"]]])
            pred_xyxy = torch.from_numpy(pred_xyxy)
            pred_xyxy = pred_xyxy.to(device)

        if num_of_labels == 0:
            instance_counts[2] += 1
        else:
            for box in boxes:
                instance_counts[class_to_num[box[0]]] += 1

        target_classes = [class_to_num[box[0]] for box in boxes]
        correct = torch.zeros(num_of_preds, num_iou_vals, dtype=torch.bool, device=device)

        if num_of_preds == 0:
            if num_of_labels:
                stats.append((correct, torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor(target_classes, device=device)))
            continue

        if num_of_labels:
            correct = correct_predictions(im_path, boxes, pred, iou_vals, cuda=cuda)
            
        if not type(pred) is list:
            pred_confs = pred.xyxy[0][:, 4].to(device)
            pred_labels = pred.xyxy[0][:, 5].to(device)
        else:
            pred_confs = pred_xyxy[:, 4].to(device)
            pred_labels = pred_xyxy[:, 5].to(device)

        stats.append((correct, pred_confs, pred_labels, torch.tensor(target_classes, device=device)))

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


def metrics_by_var(model, images, var_name, var_func = None, min_iou_val=0.5, cuda=True, dataset_path=None):
    """Seperate the given dataset by the chosen variable and print the metrics of each partition
       Arguments:
            model: model to run
            images: txt file of image paths or list or image paths
            var_name: csv header name to filter by
            var_func: optional func to run the value of var_name through, useful for discretization"""
    
    #read image paths from txt file
    image_paths = process_images_input(images)

    dataset = dataset_by_id() if dataset_path is None else dataset_by_id(dataset_path)
    splits = {}

    #split data by var_name
    for image_path in image_paths:
        id = id_from_im_name(image_path)
        if var_name == "im":
            value = var_func(cv2.imread(image_path))
        else:
            value = dataset[id][var_name]
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
        metrics = get_metrics(model, splits[value], cuda=cuda, min_iou_val=min_iou_val, dataset_path=dataset_path)
        print_metrics(*metrics)
        print("----------------------------------------------------")
    print("FINISHED")


def validiate(model, images, cuda = True, min_iou_val = 0.5, dataset_path=None):
    """Calculate and prints the metrics on a single dataset
        Arguments:
            model: the model to generate predictions with
            images: txt file of image paths or list or image paths
            cuda: enable cuda
            min_iou_val: the smallest iou value used when determine prediction correctness
            """
    image_paths = process_images_input(images)
    metrics = get_metrics(model, image_paths, cuda=cuda, min_iou_val=min_iou_val, dataset_path=dataset_path)
    print_metrics(*metrics)


def compare_to_gt(model, images, label = "urchin", save_path = False, limit = None, 
                  filter_var = None, filter_func = None, display_correct = False, cuda=True,
                  min_iou_val = 0.5):
    """Creates figures to visually compare model predictions to the actual labels
        model: yolo model to run
        images: txt of list of image paths
        label: "all", "empty", "urchin", "kina", "centro", used to filter what kind of images are compared
        save_path: if this is a file path, figures will be saved instead of shown
        limit: number of figures to show/save, leave as none for no limit
        filter_var: csv var to be passed as input to filter function
        filter_func: function to be used to filter images, return false to skip an image
    """
    if label not in ("all", "empty", "urchin", "kina", "centro"):
        raise ValueError(f'label must be in {("all", "empty", "urchin", "kina", "centro")}')

    dataset = dataset_by_id()

    image_paths = process_images_input(images)
    filtered_paths = []

    #filter paths using label parameter and filter_var and filter_func
    for path in image_paths:
        id = id_from_im_name(path)
        im_data = dataset[id]
        if filter_var and filter_func and not filter_func(cv2.imread(f"data/images/im{id}.JPG") if filter_var == "im" else im_data[filter_var]): continue

        if label == "empty":
            if im_data["count"] != "0": continue
        elif label == "urchin":
            if im_data["count"] == "0": continue
        elif label == "kina":
            if im_data["Evechinus"].upper() == "FALSE": continue
        elif label == "centro":
             if im_data["Centrostephanus"].upper() == "FALSE": continue

        filtered_paths.append(path)
    
    #loop through all the filtered images and display them with gt and predictions drawn
    for i, im_path in enumerate(filtered_paths):
        id = id_from_im_name(im_path)
        boxes = ast.literal_eval(dataset[id]["boxes"])
        
        matplotlib.use('TkAgg')
        fig, axes = plt.subplots(1, 2, figsize = (14, 6))
        fig.suptitle(f"{im_path}\n{i + 1}/{len(filtered_paths)}")

        #format image meta data to display at the bottom of the fig
        row_dict = dataset[id]
        row_dict.pop("boxes", None)
        row_dict.pop("url", None)
        row_dict.pop("id", None)
        text = str(row_dict)[1:-1].replace("'", "").split(",")

        cutoff_index = 7
        text = f"{'    '.join(text[:cutoff_index])} \n {'    '.join(text[cutoff_index:])}"
        fig.text(0.5, 0.05, text, ha='center', fontsize=10)

        im = Image.open(im_path.strip("\n"), formats=["JPEG"])

        #Generate predictions
        prediction = model(im_path)
        if type(model) is UrchinDetector:
            num_of_preds = len(prediction.pandas().xywh[0])
        else:
            num_of_preds = len(prediction)

        #Determine predicition correctness if display_correct is True
        correct = None
        boxes_missed = None
        if display_correct:
            iou_vals = torch.linspace(min_iou_val, 0.95, 10, device=torch.device("cuda") if cuda else torch.device("cpu"))
            correct, boxes_missed = correct_predictions(im_path, boxes, prediction, boxes_missed=True, cuda=cuda, iou_vals=iou_vals)
            correct = correct[:, 0]

        #plot ground truth boxes
        ax = axes[0]
        ax.set_title(f"Ground truth ({len(boxes)})")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        draw_bboxes(ax, boxes, im, boxes_missed=boxes_missed)
            
        #plot predicted boxes
        ax = axes[1]
        ax.set_title(f"Prediction ({num_of_preds})")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        if type(model) is UrchinDetector:
            draw_bboxes(ax, prediction.pandas().xywh[0], im, correct=correct)
        else:
            draw_bboxes(ax, prediction, im, correct=correct)

        #Save or show figure
        if not save_path:
            plt.show()
        else:
            fig.savefig(f"{save_path}/fig-{id}.png", format="png", bbox_inches='tight')

        if limit:
            limit -= 1
            if limit <= 0: break


def compare_models(model1, model2, dataset1, dataset2, images, label = "urchin", 
                  filter_var = None, filter_func = None, cuda=True):
    
    if label not in ("all", "empty", "urchin", "kina", "centro"):
        raise ValueError(f'label must be in {("all", "empty", "urchin", "kina", "centro")}')

    image_paths = process_images_input(images)
    filtered_paths = []

    #filter paths using label parameter and filter_var and filter_func
    for path in image_paths:
        id = id_from_im_name(path)
        im_data = dataset1[id]
        if filter_var and filter_func and not filter_func(cv2.imread(f"data/images/im{id}.JPG") if filter_var == "im" else im_data[filter_var]): continue

        if label == "empty":
            if im_data["count"] != "0": continue
        elif label == "urchin":
            if im_data["count"] == "0": continue
        elif label == "kina":
            if im_data["Evechinus"].upper() == "FALSE": continue
        elif label == "centro":
             if im_data["Centrostephanus"].upper() == "FALSE": continue

        filtered_paths.append(path)
 
    #loop through all the filtered images and display them with gt and predictions drawn
    for i, im_path in enumerate(filtered_paths):
        id = id_from_im_name(im_path)
        boxes1 = ast.literal_eval(dataset1[id]["boxes"])
        boxes2 = ast.literal_eval(dataset2[id]["boxes"])
        
        matplotlib.use('TkAgg')
        fig, axes = plt.subplots(2, 2, figsize = (16, 8))
        fig.suptitle(f"{im_path}\n{i + 1}/{len(filtered_paths)}")

        im = Image.open(im_path.strip("\n"), formats=["JPEG"])

        #Generate predictions
        prediction1 = model1(im_path)
        prediction2 = model2(im_path)
        if not type(prediction1) is list:
            num_of_preds1 = len(prediction1.pandas().xywh[0])
        else:
            num_of_preds1 = len(prediction1)

        if not type(prediction2) is list:
            num_of_preds2 = len(prediction2.pandas().xywh[0])
        else:
            num_of_preds2 = len(prediction2)

        #Determine predicition correctness if display_correct is True

        correct1, boxes_missed1 = correct_predictions(im_path, boxes1, prediction1, boxes_missed=True, cuda=cuda)
        correct1 = correct1[:, 0]
        correct2, boxes_missed2 = correct_predictions(im_path, boxes2, prediction2, boxes_missed=True, cuda=cuda)
        correct2 = correct2[:, 0]

        #plot ground truth boxes
        ax = axes[0][0]
        ax.set_title(f"M1 - Ground truth ({len(boxes1)})")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        draw_bboxes(ax, boxes1, im, boxes_missed=boxes_missed1)
            
        #plot predicted boxes
        ax = axes[0][1]
        ax.set_title(f"M1 - Prediction ({num_of_preds1})")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        if not type(prediction1) is list:
            draw_bboxes(ax, prediction1.pandas().xywh[0], im, correct=correct1)
        else:
            draw_bboxes(ax, prediction1, im, correct=correct1)

        #plot ground truth boxes
        ax = axes[1][0]
        ax.set_title(f"M2 - Ground truth ({len(boxes2)})")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        draw_bboxes(ax, boxes2, im, boxes_missed=boxes_missed2)
            
        #plot predicted boxes
        ax = axes[1][1]
        ax.set_title(f"M2 - Prediction ({num_of_preds2})")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        if not type(prediction2) is list:
            draw_bboxes(ax, prediction2.pandas().xywh[0], im, correct=correct2)
        else:
            draw_bboxes(ax, prediction2, im, correct=correct2)

        plt.show()


def urchin_count_stats(model, images):
    """Get stats on urchin count predictions (how many urchins are in the image). This ignores wether the predictions are correct"""
    image_paths = process_images_input(images)

    dataset = dataset_by_id()

    count_errors = []
    min_err_id = 0
    max_err_id = 0
    total_count = 0
    total_predicted = 0
    #Loop through predictions and calculate the count error (num of predictions - num of true labels)
    for im_path in image_paths:
        id = id_from_im_name(im_path)
        num_of_true_boxes = int(dataset["count"])
        pred = model(im_path)
        num_of_pred_boxes = len(pred.pandas().xyxy[0])

        error = num_of_pred_boxes - num_of_true_boxes
        count_errors.append(error)
        total_count += num_of_true_boxes
        total_predicted += num_of_pred_boxes

        if max(count_errors) == error: max_err_id = id
        if min(count_errors) == error: min_err_id = id

    #print stats
    print("Count error stats:")
    print(f"mean: {np.mean(count_errors)}")
    print(f"median: {np.median(count_errors)}")
    print(f"std: {np.std(count_errors)}")
    print(f"min: {min(count_errors)} (id: {min_err_id})")
    print(f"max: {max(count_errors)} (id: {max_err_id})")
    print(f"Total urchin count: {total_count}")
    print(f"Total predicted urchin count: {total_predicted}")
    print(f"Total error: {np.sum(count_errors)}")

    #Create dot plot
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


def detection_accuracy(model, images, num_iou_vals = 10, cuda = True, min_iou_val = 0.5):
    """Evaluates detection accuracy at different iou thresholds
        Arguments:
            model: model to use for preedictions
            images: txt of image file paths or list of image file paths
            num_iou_vals: number of iou values to test at (evenly spaced between 0.5 and 0.95)
            cuda: enable cuda
        Returns:
            perfect_detection_accuracy: 1xnum_iou_vals array where each value is the proportion of images the model perfectly detected
            at_least_one_accuracy: 1xnum_iou_vals array where each value is the proportion of images with at least 1 correct prediction
            perfect_images: list of image paths of images that were perfectly detected (at iou th = 0.5)
            at_least_one_images: list of image paths of images with at least one correct prediciton (at iou th = 0.5)
    """
    image_paths = process_images_input(images)

    dataset = dataset_by_id()
    perfect_detection_count = np.zeros(num_iou_vals, dtype=np.int32)
    at_least_one_correct_count = np.zeros(num_iou_vals, dtype=np.int32)
    perfect_images, at_least_one_images = [], []

    for im_path in image_paths:
        id = id_from_im_name(im_path)
        boxes = ast.literal_eval(dataset[id]["boxes"])
        pred = model(im_path)
        num_of_preds = len(pred.pandas().xyxy[0])
        num_of_true_boxes = len(boxes)
  
        #if the image is correctly predicted as containing no urchins
        if num_of_true_boxes == 0 and num_of_preds == 0:
            perfect_detection_count += 1
            at_least_one_correct_count += 1
            perfect_images.append(im_path)
            at_least_one_images.append(im_path)
            continue

        #Calculate number of correct predictions at each iou threshold
        correct = correct_predictions(im_path, boxes, pred, iou_vals=torch.linspace(min_iou_val, 0.95, num_iou_vals), cuda=cuda).numpy()
        number_correct = np.sum(correct, axis=0, dtype=np.int32)

        #check for perfect prediction
        if num_of_preds == num_of_true_boxes: 
            perfect_detection_count += (number_correct == num_of_preds).astype(np.int32)
            if number_correct[0] == num_of_preds: perfect_images.append(im_path)

        #At least one correct prediction
        at_least_one_correct_count += (number_correct >= 1).astype(np.int32)
        if number_correct[0] >= 1: at_least_one_images.append(im_path)

    #Print stats
    print("iou thresh values:".ljust(20), torch.linspace(min_iou_val, 0.95, num_iou_vals).numpy())
    print("Perfect detections:".ljust(20), perfect_detection_count/len(image_paths))
    print("At least 1 correct:".ljust(20), at_least_one_correct_count/len(image_paths))

    return perfect_images, at_least_one_images
        

def classification_over_frames(model, images, seconds_threshold = 5):
    """Binary classification accuracy (urchin or no urchins) calculated on groupings of consecutive frames"""
    dataset = dataset_by_id()
    image_paths = process_images_input(images)

    #group images by time
    print("Started grouping")
    time_strings = [(dataset[id_from_im_name(x)]["time"], str(id_from_im_name(x))) for x in image_paths]
    datetimes = [] 
    for value, id in time_strings:
        try:
            datetimes.append((datetime.strptime(value, "%Y-%m-%d %H:%M:%S"), id))
        except:
            datetimes.append((datetime.strptime(value, "%d/%m/%Y %H:%M"), id))

    datetimes.sort()
    frame_groupings = [[]]
    for t, id in datetimes:
        for other_t in frame_groupings[-1]:
            if abs((t - other_t[0]).total_seconds()) <= seconds_threshold and dataset[int(id)]["deployment"] == dataset[int(id)]["deployment"]:
                frame_groupings[-1].append((t, id))
                break

        if (t, id) not in frame_groupings[-1]:
            frame_groupings.append([(t, id)])

    frame_groupings.pop(0)
    frame_groupings = [[f"data/images/im{x[1]}.JPG" for x in group] for group in frame_groupings]
    frame_groupings = [group for group in frame_groupings if len(group) > 1]
    print(len(frame_groupings))

    #classify frame groupings by majority vote
    print("Started classifying")
    correct_classification = 0
    for group in frame_groupings:
        preds = model.batch_predict(group)
        urchin_votes = 0
        empty_votes = 0
        urchin_true = 0
        empty_true = 0
        for im, pred in zip(group, preds):
            num_of_preds = len(pred.pandas().xyxy[0])
            id = id_from_im_name(im)
            num_of_labels = int(dataset[id]["count"])

            if num_of_labels:
                urchin_true += 1
            else:
                empty_true += 1

            if num_of_preds:
                urchin_votes += 1
            else:
                empty_votes += 1

        group_contains_urchins = urchin_true >= empty_true

        if urchin_votes >= empty_votes and group_contains_urchins: correct_classification += 1
        if empty_votes > urchin_votes and not group_contains_urchins: correct_classification += 1

    print("Finished")
    print(correct_classification/len(frame_groupings))
        

def image_rejection_test(model, images, image_score_funcs, image_score_ths):
        """Displays plots of different metrics and test coverage when using various image scoring function and thresholds to reject images
            Arguments:
                model: model to use
                images: txt or list of image paths
                image_score_funcs: list of functions that take an image as an input and return a single number
                image_score_ths: list of lists of thresholds to calculate metrics at (one list of thresholds per image score func)
        """

        image_paths = process_images_input(images)

        #Calculate scores for each image
        images_with_scores = []
        for i, path in enumerate(image_paths):
            im = cv2.imread(path)
            scores = [func(im) for func in image_score_funcs]
            images_with_scores.append([path] + scores)

        matplotlib.use('TkAgg')
        fig, axes = plt.subplots(1, len(image_score_funcs), figsize = (14, 6))
        if len(image_score_funcs) == 1: axes = [axes]

        #Create a plot for each image score function
        for i, score_func in enumerate(image_score_funcs):
            mapScores = []
            pScores = []
            rScores = []
            f1Scores = []
            coverage = []
            #get metrics at each threshold
            for th in image_score_ths[i]:
                filtered_images = [x[0] for x in images_with_scores if x[i + 1] >= th]

                metrics = get_metrics(model, filtered_images)
                pScores.append(metrics[1])
                rScores.append(metrics[3])
                mapScores.append(metrics[6])
                f1Scores.append((metrics[4][0] + metrics[4][1])/2 if len(metrics[4]) == 2 else metrics[4][0])
                coverage.append(len(filtered_images)/len(image_paths))

            #draw plots
            ax = axes[i]
            scores = (mapScores, f1Scores, coverage, pScores, rScores)
            colours = ("blue", "red", "orange", "green", "purple")
            labels = ("mAP50", "F1", "Coverage", "Precision", "Recall")

            for values, colour, label in zip(scores, colours, labels):
                ax.scatter(image_score_ths[i], values, c = colour, label = label)
                ax.plot(image_score_ths[i], values, c = colour)

            ax.legend()
            ax.grid(True)
            ax.set_yticks(np.arange(0, 1.001, 0.05))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))
            ax.set_title(f"Metrics by {score_func.__name__} threshold")

        plt.show()


def bin_by_count(model, images, bin_width, cuda=True, seperate_empty_images=False):
    """Bin images by urchin count and calculate metrics on each bin
        Arguments:
            model: model to generate predictions with
            images: txt file of image paths or list or image paths
            bin_width: determines the number of counts per bin e.g. 5 means bins [0, 5), [5, 10) etc
            seperate_empty_images: put images with count 0 into there own seperate bin"""
    
    image_paths = process_images_input(images)
    dataset = dataset_by_id()

    #get the count from each image
    counts = []
    for im in image_paths:
        id = id_from_im_name(im)
        counts.append(int(dataset[id]["count"]))

    #bin images by count
    bin_starts = list(range(0, max(counts) + bin_width, bin_width))
    bins = [[] for i in bin_starts]

    if seperate_empty_images:
        empty_bin = []

    for i, im in enumerate(image_paths):
        if seperate_empty_images and counts[i] == 0:
            empty_bin.append(im)
            continue
        for j, bin_start in enumerate(bin_starts):
            if counts[i] >= bin_start and counts[i] < bin_start + bin_width:
                bins[j].append(im)

    if seperate_empty_images:
        print(f"Empty images - {len(empty_bin)}:")
        print("\n")

    #Validate each bin
    for i in range(len(bin_starts)):
        if bins[i]:
            print(f"Bin [{1 if seperate_empty_images and bin_starts[i] == 0 else bin_starts[i]}, {bin_starts[i] + bin_width}) - {len(bins[i])} images:")
            validiate(model, bins[i], cuda)
            print("\n")


def missed_boxes_ids(model, images, filter_var, filter_func, min_iou=0.5, cuda=True):
    """Generates a list of annotation ids that the model did not detect (i.e. false negatives)
        Arguments:
            model: model to generate predictions with
            images: txt of list of image paths
            filter_var: csv var to be passed as input to filter function
            filter_func: function to be used to filter images, return false to skip an image
            min_iou_val: the smallest iou value used when determine prediction correctness
            cuda: enable cuda
        Returns:
            list of squidle annotation ids corrosponding to missed boxes from the filtered images"""
    
    image_paths = process_images_input(images)
    rows = dataset_by_id()

    #Filtering images
    image_paths = [im for im in image_paths if 
                   filter_func(rows[id_from_im_name(im)][filter_var]) and 
                   len(ast.literal_eval(rows[id_from_im_name(im)]["boxes"])) != 0]
    
    #Finding ids
    ids = []
    im_paths = []
    images_count = 0
    for i in range(len(image_paths)):
        boxes = ast.literal_eval(rows[id_from_im_name(image_paths[i])]["boxes"])
        pred = model(image_paths[i])

        _, missed = correct_predictions(image_paths[i], boxes, pred, 
                                              iou_vals=torch.linspace(min_iou, 0.95, 10), 
                                              boxes_missed=True, cuda=cuda)
        
        has_missed_box = False
        for j in range(len(boxes)):
            if missed[j]:
                ids.append(boxes[j][-1])
                has_missed_box = True

        if has_missed_box:
            images_count += 1
            im_paths.append(image_paths[i])

    print(f"Number of missed annotations: {len(ids)}")
    print(f"Images with a FN: {images_count}")

    return ids


def compare_dataset_annotations(d_path1, d_path2, d_name1, d_name2):
    """Compare the annotations of the same images across two versions of the dataset"""
    d1 = dataset_by_id(d_path1)
    d2 = dataset_by_id(d_path2)
    for im_path in process_images_input("data/datasets/full_dataset_v3/val.txt"):
        matplotlib.use('TkAgg')
        fig, axes = plt.subplots(1, 2, figsize = (14, 6))

        id = id_from_im_name(im_path)
        im = Image.open(im_path.strip("\n"), formats=["JPEG"])

        ax = axes[0]
        ax.set_title(d_name1)
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        draw_bboxes(ax, ast.literal_eval(d1[id]["boxes"]), im)

        if id in d2:
            ax = axes[1]
            ax.set_title(d_name2)
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            draw_bboxes(ax, ast.literal_eval(d2[id]["boxes"]), im)

        plt.show()


def calibration_curve(model, images, conf_step=0.1):
    image_paths = process_images_input(images)
    rows = dataset_by_id()
    model.update_parameters(0.05)
    conf_bins = np.arange(0, 1, conf_step) #bins are [conf_bins[i], conf_bins[i+1])

    tp = np.zeros_like(conf_bins)
    totals = np.zeros_like(conf_bins)

    def f(x):
        return min(-2.696556590256292*(x**3) + 4.262217397630296*(x**2) -0.643336789510566*x + 0.093656077697483, 1)

    for im in image_paths:
        id = id_from_im_name(im)
        preds = model(im)
    
        correct_preds = correct_predictions(im, ast.literal_eval(rows[id]["boxes"]), preds)

        for i in range(len(preds.xywh[0])):
            pred =  preds.xywh[0][i]
            conf = pred[4].item()
            bin_index = int(conf//conf_step)
            totals[bin_index] += 1
            if correct_preds[i][0]: tp[bin_index] += 1

    print(conf_bins)
    print((2 * conf_bins + conf_step)/2)
    print(tp/totals)

    #points = [(str(x[0]), str(x[1])) for x in zip((2 * conf_bins + conf_step)/2, tp/totals)]
    #for x, y in points:
    #    print(x +","+ y)



    matplotlib.use('TkAgg')
    plt.figure(figsize=(8, 6))
    plt.plot((2 * conf_bins + conf_step)/2, tp/totals, marker='o', label='Calibration Curve', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Calibration')
    plt.xlabel('Average confidence')
    plt.ylabel('Fraction of TP')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
  
   
if __name__ == "__main__":
    weight_path = "models/yolov5m-highRes-ro/weights/best.pt"
    txt = "data/datasets/full_dataset_v4/val.txt"
    test_txt = "data/datasets/full_dataset_v4/test.txt"
    d = dataset_by_id("data/csvs/High_conf_clipped_dataset_V4.csv")
    cuda = torch.cuda.is_available()

    modelV4 = UrchinDetector("models/yolov5m-highRes-ro-v4/weights/best.pt")
    #yolox_model1 = UrchinDetector_YOLOX("models/yolox-m/yolox-m-v1.pth", img_size=640, conf=0.2)
    yolox_model2 = UrchinDetector_YOLOX("models/yolox-m/yolox-m-v2.pth", img_size=1280, conf=0.2)

    #compare_to_gt(yolox_model, txt, "all", display_correct=True, cuda=True)
    #compare_models(modelV4, yolox_model2, d, d, txt)

    validiate(modelV4, txt)
    validiate(yolox_model2, txt)

    #modelV3 = UrchinDetector("models/yolov5m-highRes-ro/weights/best.pt")
    #modelV4 = UrchinDetector("models/yolov5m-highRes-ro-v4/weights/best.pt")

    #joint_val_dataset = [im for im in process_images_input(txt) if im in process_images_input("data/datasets/full_dataset_v4/val.txt")]
    #joint_test_dataset = [im for im in process_images_input(test_txt) if im in process_images_input("data/datasets/full_dataset_v4/test.txt")]
    #print(len(joint_test_dataset))

    #metrics_by_var(modelV3, test_txt, "source", None, 0.3, cuda, dataset_path="data/csvs/High_conf_clipped_dataset_V3.csv")
    #metrics_by_var(modelV4, "data/datasets/full_dataset_v4/test.txt", "source", None, 0.3, cuda, dataset_path="data/csvs/High_conf_clipped_dataset_V4.csv")

    #d1 = dataset_by_id("data/csvs/High_conf_clipped_dataset_V3.csv")
    #d2 = dataset_by_id("data/csvs/High_conf_clipped_dataset_V4.csv")
    #compare_models(modelV3, modelV4, d1, d2, joint_test_dataset, filter_var="source", filter_func=lambda x: x == "NSW DPI Urchins")


    #validiate(modelV3, joint_test_dataset, cuda, 0.5, "data/csvs/High_conf_clipped_dataset_V3.csv")
    #validiate(modelV4, joint_test_dataset, cuda, 0.5, "data/csvs/High_conf_clipped_dataset_V4.csv")

    #perfect_images, at_least_one_images =  detection_accuracy(model, txt, cuda=cuda, min_iou_val=0.3)
    
    #metrics_by_var(model, txt, "source", None, cuda)


    #compare_to_gt(model, txt, "all", display_correct=True, cuda=cuda, filter_var="source",
    #              filter_func=lambda x: x == "NSW DPI Urchins", min_iou_val= 0.3)
    #NSW DPI Urchins
    #UoA Sea Urchin
    #Urchins - Eastern Tasmania

    #metrics_by_var(model, txt, "source", cuda = cuda)
