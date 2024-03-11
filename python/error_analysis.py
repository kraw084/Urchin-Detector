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

from urchin_utils import dataset_by_id, UrchinDetector, process_images_input, project_sys_path, id_from_im_name, draw_bboxes, annotate_images

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

    im = Image.open(im_path)
    w, h = im.size
    class_to_num = {"Evechinus chloroticus": 0, "Centrostephanus rodgersii": 1}

    labels = torch.zeros((len(gt_box), 5), device= torch.device("cuda") if cuda else torch.device("cpu")) #(class, x1, y1, x2, y2)
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
    correct = process_batch(pred.xyxy[0], labels, iou_vals)

    if boxes_missed:
        #find all the boxes where all the iou values are less than the threshold
        iou = box_iou(labels[:, 1:], pred.xyxy[0][:, :4])
        print(iou)
        gt_box_missed = np.all(a=(iou < iou_vals[0]).numpy(force=True), axis=1)
        print(gt_box_missed)
        print(iou_vals[0])
        return correct, gt_box_missed
    
    return correct


def get_metrics(model, image_set, cuda=True, min_iou_val = 0.5):
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

    device = torch.device("cuda") if cuda else torch.device("cpu")

    class_to_num = {"Evechinus chloroticus": 0, "Centrostephanus rodgersii": 1}
    instance_counts = [0, 0, 0] #num of kina boxes, num of centro boxes, num of empty images
    dataset = dataset_by_id()
    num_iou_vals = 10
    iou_vals = torch.linspace(min_iou_val, 0.95, num_iou_vals, device=device)
    stats = [] #(num correct, confidence, predicated classes, target classes)

    #get the relavent stats for each images predictions
    for im_path in image_set:
        id = id_from_im_name(im_path)
        boxes = ast.literal_eval(dataset[id]["boxes"])
        pred = model(im_path)
        num_of_labels = len(boxes)
        num_of_preds = pred.xyxy[0].shape[0]

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


def metrics_by_var(model, images, var_name, var_func = None, min_iou_val=0.5, cuda=True):
    """Seperate the given dataset by the chosen variable and print the metrics of each partition
       Arguments:
            model: model to run
            images: txt file of image paths or list or image paths
            var_name: csv header name to filter by
            var_func: optional func to run the value of var_name through, useful for discretization"""
    
    #read image paths from txt file
    image_paths = process_images_input(images)

    dataset = dataset_by_id()
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
        metrics = get_metrics(model, splits[value], cuda=cuda, min_iou_val=min_iou_val)
        print_metrics(*metrics)
        print("----------------------------------------------------")
    print("FINISHED")


def validiate(model, images, cuda = True, min_iou_val = 0.5):
    """Calculate and prints the metrics on a single dataset
        Arguments:
            model: the model to generate predictions with
            images: txt file of image paths or list or image paths
            cuda: enable cuda
            min_iou_val: the smallest iou value used when determine prediction correctness
            """
    image_paths = process_images_input(images)
    metrics = get_metrics(model, image_paths, cuda=cuda, min_iou_val=min_iou_val)
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
        num_of_preds = len(prediction.pandas().xywh[0])

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
        draw_bboxes(ax, prediction.pandas().xywh[0], im, correct=correct)

        #Save or show figure
        if not save_path:
            plt.show()
        else:
            fig.savefig(f"{save_path}/fig-{id}.png", format="png", bbox_inches='tight')

        if limit:
            limit -= 1
            if limit <= 0: break


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


if __name__ == "__main__":
    weight_path = "models/yolov5m-highRes-ro/weights/best.pt"
    txt = "data/datasets/full_dataset_v3/val.txt"
    cuda = torch.cuda.is_available()

    model = UrchinDetector(weight_path)

    #bin_by_count(model, txt, 5, cuda, seperate_empty_images=True)

    #perfect_images, at_least_one_images =  detection_accuracy(model, txt, cuda=cuda, min_iou_val=0.3)

    compare_to_gt(model, txt, "all", display_correct=True, cuda=cuda, filter_var="source",
                  filter_func=lambda x: x == "NSW DPI Urchins", min_iou_val= 0.3)
    #NSW DPI Urchins
    #UoA Sea Urchin
    #Urchins - Eastern Tasmania
  
    #validiate(model, txt, cuda)