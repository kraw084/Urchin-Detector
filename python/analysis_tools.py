import csv

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from urchin_utils.data_utils import process_images_input, id_from_im_name, filter_images, dataset_by_id
from urchin_utils.model_utils import gt_to_detection, NUM_TO_LABEL
from urchin_utils.eval_utils import correct_predictions, validiate
from urchin_utils.vis_utils import plot_im_and_boxes

def compare_to_gt(model, 
                  images, 
                  dataset,
                  label = "urchin", 
                  save_path = False, 
                  limit = None, 
                  filter_var = None, 
                  filter_func = None, 
                  display_correct = False, 
                  iou_val = 0.5, 
                  ):
    """Creates figures to visually compare model predictions to the actual annotations
    Args:
        model: model object to generate predictions with
        images: txt or list of image paths
        label: "all", "empty", "urchin", "kina", "centro", "helio" used to filter what kind of images are compared
        save_path: if this is a file path, figures will be saved instead of shown
        limit: number of figures to show/save, leave as none for no limit
        filter_var: csv var to be passed as input to filter function
        filter_func: function to be used to filter images, return false to skip an image
    """
    image_paths = process_images_input(images)
    filtered_paths = filter_images(image_paths, dataset, label, filter_var, filter_func, limit)

    #loop through all the filtered images and display them with gt and predictions drawn
    for i, im_path in enumerate(filtered_paths):
        id = id_from_im_name(im_path)
        gt = gt_to_detection(dataset[id])
        
        #Set up figure and subplots
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
        
        #read image and convert to RGB for matplotlib display
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        #Generate predictions
        prediction = model(im_path)
  

        #Determine predicition correctness if display_correct is True
        if display_correct:
            correct, boxes_missed = correct_predictions(gt, prediction, iou_val=iou_val)
        else:
            correct = None
            boxes_missed = None
        
        #plot ground truth boxes
        ax = axes[0]
        ax.set_title(f"Ground truth ({len(gt)})")
        plot_im_and_boxes(im, gt, ax, correct=None, boxes_missed=boxes_missed)
            
        #plot predicted boxes
        ax = axes[1]
        ax.set_title(f"Prediction ({len(prediction)})")
        plot_im_and_boxes(im, prediction, ax, correct=correct, boxes_missed=None)
  
        #Save or show figure
        if not save_path:
            plt.show()
        else:
            fig.savefig(f"{save_path}/fig-{id}.png", format="png", bbox_inches='tight')


def compare_models(model1, 
                   model2, 
                   dataset, 
                   images, 
                   label = "urchin", 
                   limit = None,
                   filter_var = None, 
                   filter_func = None):
    """Creates figures to visually compare the predictions of two models on the same image
    Args:
        model1: model object to generate predictions with
        model2: another model object to generate predictions with
        images: txt or list of image paths
        label: "all", "empty", "urchin", "kina", "centro", "helio" used to filter what kind of images are compared
        save_path: if this is a file path, figures will be saved instead of shown
        limit: number of figures to show/save, leave as none for no limit
        filter_var: csv var to be passed as input to filter function
        filter_func: function to be used to filter images, return false to skip an image
    """
    
    image_paths = process_images_input(images)
    filtered_paths = filter_images(image_paths, dataset, label, filter_var, filter_func, limit)
 
    #loop through all the filtered images and display them with gt and predictions drawn
    for i, im_path in enumerate(filtered_paths):
        id = id_from_im_name(im_path)
        gt = gt_to_detection(dataset[id])
        
        matplotlib.use('TkAgg')
        fig, axes = plt.subplots(2, 2, figsize = (16, 8))
        fig.suptitle(f"{im_path}\n{i + 1}/{len(filtered_paths)}")

        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        #Generate predictions
        prediction1 = model1(im_path)
        prediction2 = model2(im_path)

        #Determine predicition correctness if display_correct is True
        correct1, boxes_missed1 = correct_predictions(gt, prediction1)
        correct2, boxes_missed2 = correct_predictions(gt, prediction2)

        #plot ground truth boxes
        ax = axes[0][0]
        ax.set_title(f"M1 - Ground truth ({len(gt)})")
        plot_im_and_boxes(im, gt, ax, None, boxes_missed1)
            
        #plot predicted boxes
        ax = axes[0][1]
        ax.set_title(f"M1 - Prediction ({len(prediction1)})")
        plot_im_and_boxes(im, prediction1, ax, correct1, None)

        #plot ground truth boxes
        ax = axes[1][0]
        ax.set_title(f"M2 - Ground truth ({len(gt)})")
        plot_im_and_boxes(im, gt, ax, None, boxes_missed2)
            
        #plot predicted boxes
        ax = axes[1][1]
        ax.set_title(f"M2 - Prediction ({len(prediction2)})")
        plot_im_and_boxes(im, prediction2, ax, correct2, None)

        plt.show()
        

def bin_by_count(model, images, dataset, bin_width, cuda=True, seperate_empty_images=False):
    """Bin images by urchin count and calculate metrics on each bin
    Args:
        model: model to generate predictions with
        images: txt file of image paths or list or image paths
        dataset: dict of the dataset with id as the key and the row (another dict) as the value
        bin_width: determines the number of counts per bin e.g. 5 means bins [0, 5), [5, 10) etc
        seperate_empty_images: put images with count 0 into there own seperate bin
    """

    image_paths = process_images_input(images)

    #get the count from each image
    counts = []
    for im in image_paths:
        id = id_from_im_name(im)
        counts.append(int(dataset[id]["count"]))

    #bin images by count
    bin_starts = list(range(0, max(counts) + bin_width, bin_width))
    bins = [[] for _ in bin_starts]

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
            
            
def calibration_curve(model, images, dataset, conf_step=0.1):
    """Creates a calibration curve to show how prediction confidence relates to true accuracy
    Args:
        model: model to generate predictions with
        images: txt file of image paths or list or image paths
        dataset: dict of the dataset with id as the key and the row (another dict) as the value
        conf_step: step size to generate confidence bins (smaller step will give a more detailed curve)
    """
    image_paths = process_images_input(images)
    model.update_parameters(0.01)
    conf_bins = np.arange(0, 1, conf_step) #bins are [conf_bins[i], conf_bins[i+1])

    #arrays for keeping track of num of true positives and total predictions
    tp = np.zeros_like(conf_bins)
    totals = np.zeros_like(conf_bins)

    for im in image_paths:
        #generate predictions and determine correctness
        id = id_from_im_name(im)
        preds = model(im)
        correct_preds = correct_predictions(gt_to_detection(dataset[id]), preds)

        #for each bounding box place it in the bin corresponding to its confidence and update the tp and total arrays
        for i in range(len(preds)):
            pred =  preds[i]
            conf = pred[4]
            bin_index = int(conf//conf_step)
            totals[bin_index] += 1
            if correct_preds[i][0]: tp[bin_index] += 1

    #print(conf_bins)
    #print((2 * conf_bins + conf_step)/2)
    #print(tp/totals)

    #Display calibration curve
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
    
    
def save_detections(model, images, output_csv):
    """Export model predictions of a set of images to a csv with each row being one bounding box
    Args:
        model: model to generate predictions with
        images: txt file of image paths or list or image paths
        output_csv: path to the output csv file
    """
    image_paths = process_images_input(images)
    
    csv_rows = []
    for im in tqdm(image_paths, desc="Generating predictions",  bar_format="{l_bar}{bar:30}{r_bar}"):
        #generate predictions
        det = model(im)
        
        #add each box as a new row to the csv
        for box in det.gen(box_format="xyxycl"):
            csv_rows.append({"image": im.split("\\")[-1],
                             "x_top_left": box[0],
                             "y_top_left": box[1],
                             "x_bottom_right": box[2],
                             "y_bottom_right": box[3],
                             "conf": box[4],
                             "class": NUM_TO_LABEL[int(box[5])]})
            
    #write rows to csv
    f = open(output_csv, "w", newline="")
    csv_writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    csv_writer.writeheader()
    csv_writer.writerows(csv_rows)
    f.close()
    
    
def metrics_by_var(model, images, dataset, var_name, var_func = None):
    """Seperate the given dataset by the chosen variable and print the metrics of each partition
    Args:
        model: model to run
        images: txt file of image paths or list or image paths
        var_name: csv header name to filter by
        var_func: optional func to run the value of var_name through, useful for discretization"""
    
    #read image paths from txt file
    image_paths = process_images_input(images)

    #split data by var_name
    splits = {}
    for image_path in image_paths:
        #read value from dataset and apply func is given
        id = id_from_im_name(image_path)
        value = dataset[id][var_name]
        if var_func: value = var_func(value)

        #add image path to dict of splits based on the value
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
        validiate(model, splits[value], dataset)
        print("----------------------------------------------------")
        
    print("FINISHED")