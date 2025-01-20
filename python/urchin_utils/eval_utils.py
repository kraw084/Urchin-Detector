import torch
import numpy as np
import pandas as pd
from tqdm import tqdm   

from urchin_utils.model_utils import LABEL_TO_NUM, NUM_TO_LABEL, project_sys_path, gt_to_detection
from urchin_utils.data_utils import id_from_im_name, process_images_input

project_sys_path(2)
from yolov5.utils.metrics import box_iou, ap_per_class


def iou_matrix(boxes1, boxes2):
    """"Calculates iou between every box in boxes1 and every box in boxes2
        Args:
            boxes1: detection object with N boxes
            boxes2: detection object with M boxes
        Returns:
            NxM iou matrix where matrix[i][j] is the iou between box i in boxes1 and box j in boxes2"""
    
    #transform boxes to the reqiured format for the yolov5 iou function
    boxes1 = torch.tensor([box[:4] for box in boxes1.gen(box_format="xyxycl")])
    boxes2 = torch.tensor([box[:4] for box in boxes2.gen(box_format="xyxycl")])

    return box_iou(boxes1, boxes2).numpy()


def match_preds(gt, preds, iou_th=0.5):
    """Determines which predictions are correct and returns the matching (Based on code from yolov5 repo)
        Args:
            gt: ground truth boxes as a detection object
            preds: predicted boxes as a detection object
        Returns:
            correct: array of len(preds) where the ith element is true if pred[i] has a match
            match_indices: Nx2 array of box index matchings (gt box, pred box)
    """
    #convert detection objects to np mats
    gt_arr = np.array(gt)
    preds_arr = np.array(preds)
    
    correct = np.zeros((preds_arr.shape[0],)).astype(bool)

    #if either is empty, end early
    if gt_arr.shape[0] == 0 or preds_arr.shape[0] == 0: return correct, None

    #calculate iou matrix
    iou_mat = iou_matrix(gt, preds)
    
    #find all cases where the class labels match
    correct_label = gt_arr[:, 5:6] == preds_arr[:, 5].T

    #find all possible cases where label matchs and the iou is above the threshold
    match_indices = np.argwhere((iou_mat >= iou_th) & correct_label)
    
    #deterime the best match for each box
    if match_indices.shape[0]:
        iou_of_matches = iou_mat[match_indices[:, 0], match_indices[:, 1]]
        match_indices = match_indices[iou_of_matches.argsort()[::-1]]
        match_indices = match_indices[np.unique(match_indices[:, 1], return_index=True)[1]]
        match_indices = match_indices[np.unique(match_indices[:, 0], return_index=True)[1]]

        correct[match_indices[:, 1]] = True

    return correct, match_indices


def correct_predictions(gt_boxes, pred, iou_val = 0.5):
    """Determines what predictions are correct at different iou thresholds
        Args:
            gt_boxes: N ground truth boxes from csv as a detection object
            pred: M model prediction as a detection object 
            iou_val: iou threshold to determine correct predictions
        Returns:
            correct_preds: A np array of length N where the ith element is true if the ith predicted box has a gt box
            missed_gts: A np array of length M where the ith element is true if the ith gt box was missed (had no match)
    """
    #determine which predictions are correct
    correct_preds, match_indices = match_preds(gt_boxes, pred, iou_th=iou_val)
    
    #determine which gt boxes were missed
    missed_gts = np.ones(len(gt_boxes), dtype=bool)
    missed_gts[match_indices[:, 0]] = False
    
    return correct_preds, missed_gts 


def get_metrics(model, image_set, dataset, min_iou_val = 0.5, max_iou_val = 0.95, num_iou_vals = 10, ):
    """Computes metrics of provided image set. Based on the code from yolov5/val.py
        Args:
                model: model to get predictions from
                image_set: list of image paths
                dataset: dataset to get gt boxes from (dict with id as key and row as value)
                min_iou_val: the smallest iou value used when determine prediction correctness
                max_iou_val: the largest iou value used when determine prediction correctness
                num_iou_vals: the number of iou between the chosen min and max
        Returns: 
                precision, mean precision, recall, mean recall, f1 score, ap50, 
                map50, ap, and map of the provided images and ap_classes, a list
                with class indices that occurs in the given image set e.g. [], [0], [1], [0, 1]
        """
    
    if len(image_set) == 0: 
        return [0], 0, [0], 0, [0], [0], 0, [0], 0, [], [0, 0, 0]

    device = torch.device("cuda") if model.cuda else torch.device("cpu")

    instance_counts = [0 for i in range(len(LABEL_TO_NUM) + 1)] #num of boxes for each species plus empty images
    
    iou_vals = torch.linspace(min_iou_val, max_iou_val, num_iou_vals, device=device)
    
    stats = [] #(num correct, confidence, predicated classes, target classes)

    #get the relavent stats for each images predictions
    for im_path in tqdm(image_set, desc="Computing metrics", bar_format="{l_bar}{bar:30}{r_bar}"):
        #get gt and predicted boxes
        id = id_from_im_name(im_path)
        gt = gt_to_detection(dataset[id])
        pred = model(im_path)
        
        num_of_gts = len(gt)
        num_of_preds = len(pred)

        #find ground truth instance counts
        if num_of_gts == 0:#empty image
            instance_counts[-1] += 1
        else:
            for box in gt:#add number of boxes of each species
                instance_counts[int(box[5])] += 1

        target_classes = [int(box[5]) for box in gt]
        correct = torch.zeros(num_of_preds, num_iou_vals, dtype=torch.bool, device=device)

        if num_of_preds == 0:
            if num_of_gts: #if there are no predictions but there are gts
                stats.append((correct, torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor(target_classes, device=device)))
            continue

        #determine which predictions are correct at each iou val
        if num_of_gts:            
            for i in range(len(iou_vals)):
                correct[:, i] = torch.from_numpy(correct_predictions(gt, pred, iou_vals[i].item())[0]).to(device)
            
        xyxy_preds = torch.from_numpy(np.vstack([box for box in pred.gen(box_format="xyxycl")])).to(device)
        stats.append((correct, xyxy_preds[:, 4], xyxy_preds[:, 5], torch.tensor(target_classes, device=device)))
    
    #get metrics and calc averages
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    true_pos, false_pos, precision, recall, f1, ap_across_iou_th, ap_class = ap_per_class(*stats, names={c:i for i, c in enumerate(model.classes)})
    ap50, ap = ap_across_iou_th[:, 0], ap_across_iou_th.mean(1)
    mean_precision, mean_recall, map50, map50_95 = precision.mean(), recall.mean(), ap50.mean(), ap.mean()

    return precision, mean_precision, recall, mean_recall, f1, ap50, map50, ap, map50_95, ap_class, instance_counts


def print_metrics(precision, mean_precision, recall, mean_recall, f1, ap50, map50, ap, map, classes, counts):
    """Print metrics in a table, seperated by class"""
    headers = ["Class", "Instances", "P", "R", "F1", "ap50", "ap"]
    df = pd.DataFrame(columns=headers)
    
    #parameters may only have stats for 1 class so add those rows seperatly 
    for i, c in enumerate(classes):
        label = NUM_TO_LABEL[c]
        count = counts[c]

        df_row = [label, count, precision[i], recall[i], f1[i], ap50[i], ap[i]]
        df.loc[len(df)] = df_row
 
    if len(classes) > 1: #if the stats include multiple classes and add an average row
        df_row = ["Avg", "-", mean_precision, mean_recall, (f1[0] + f1[1])/2, map50, map]
        df.loc[len(df)] = df_row
    
    #df formatting and printing
    df.set_index("Class", inplace=True)
    df = df.round(3)
    print(df.to_string(index=True, index_names=False))
    print(f"{counts[-1]} images with no urchins")
    
    
def validiate(model, images, dataset):
    """Calculate and prints the metrics on a single dataset
    Args:
        model: the model to generate predictions with
        images: txt file of image paths or list of image paths
        dataset: 
    """
    image_paths = process_images_input(images)
    metrics = get_metrics(model, image_paths, dataset)
    print_metrics(*metrics)
    
    

