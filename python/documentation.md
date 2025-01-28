# UrchinBot Documentation
This documentation provides a quick overview of the code in this repository. We give a brief explanation of what each function, class and script does and what it is used for. For more details about each function and its arguments and output refer to the doc strings in the code.

## Overview
1. **cvat:** contains scripts used for processsing and analysising annotations from CVAT
    - **cvat_stats.py:** script that reads through a cvat xml annotation file and outputs some basic stats and metrics
    - **get_annotation.py:** contains a function for reading and formating a cvat xml annotation file into a more readable dict
2. **urchin_utils:** contains various helper functions and utilities
    - **data_utils.py:** functions for reading and dealing with the dataset
    - **eval_utils.py:** functions for evaluating model performance and calculating metrics
    - **model_utils.py:** functions and classes for loading and using object detection models
    - **vis_utils.py:** functions for various ways of drawing and visualing model predictions
3. **analysis_tools.py**: uses components from urchin_utils to create more complex functions to analyse and evaluate models
4. **run_analysis.py**: script for running functions from analysis_tools
5. **train_urchin_model.py**: script for training a yolov5 model on the urchin dataset
6. **val_urchin_model.py**: script for validating/testing a yolov5 model on the urchin dataset

### Data Utils

```dataset_by_id(csv_path=CSV_PATH)```: Creates a dictionary of the dataset with the id as the key and the row (another dict) as the value. The dict is creating using the formated csv specified by *csv_path*. The keys of the row dicts are the csv headers (see data/data_documentation.md). The dictionary created by this function is the main way all other function utilise ground truth data. Note that the boxes are stored as a string but using *ast.literal_eval(row["boxes"])* will convert the string to a list of lists.

```id_from_im_name(im_name)```: Extracts an integer id from the image name (i.e. "im{id}.JPG"). Useful for finding the id of the image for indexing a dataset dict.

```read_txt(images_txt)```: Reads in a text file of image paths and returns a list of image paths.

```process_images_input(images)```: takes either a txt file of image paths or a list of image paths and returns a list of image paths. This is used so other function can recieve either format of images as input and process them the same way.

```complement_image_set(images0, images1)```: returns a list of all the image paths in images1 that are not in images0.

```filter_images(image_paths, dataset, label="all", filter_var=None, filter_func=None, limit=None)```: Filters a txt or list of image path and returns a list. Setting *label* to one of "all", "empty", "urchin", "kina", "centro", "helio" allows for a simple way or broadly filtering the images. *fitler_var* and *filter_func* allow for a more fine grained filtering of the images. *limit* allows for a maximum number of images to keep. This is useful for when you want to analyse a small subset of the dataset.

### Evaluation Utils
```iou_matrix(boxes1, boxes2)```: creates a matrix of the IoU values between all boxes in boxes1 and boxes2. Each row of the matrix corresponds to a box in boxes1 and each column corresponds to a box in boxes2. Uses the yolov5 iou function.

```match_preds(gt, preds, iou_th=0.5)```: matches bounding box predictions to the groundtruth data and determines the correctness of predictions (at the specified IoU threshold). returns a bool array indicated which predictions are correct and a matrix of index pairs of the correct matches.

```correct_predictions(gt_boxes, pred, iou_val = 0.5)```: determines what predictions are correct and which gt boxes were missed at the specified iou thresholds. Returns two bool arrays.

```get_metrics(model, image_set, dataset, min_iou_val = 0.5, max_iou_val = 0.95, num_iou_vals = 10)```: Runs the model on the given image set and computes precision, mean precision, recall, mean recall, f1 score, ap50, map50, ap, and map. Uses code from yolov5.

```print_metrics(precision, mean_precision, recall, mean_recall, f1, ap50, map50, ap, map, classes, counts):```: prints the output of *get_metrics* in a table.

```validiate(model, images, dataset)```: this is the main function to call when you want to evaluate a model. It runs the model on the provided images and prints the metrics.

### Model Utils


