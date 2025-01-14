import os
import csv

#File paths
CSV_PATH = os.path.abspath("data/csvs/High_conf_clipped_dataset_V5.csv")
DATASET_YAML_PATH = os.path.abspath("data/datasets/full_dataset_v5/datasetV5.yaml")
WEIGHTS_PATH = os.path.abspath("models/yolov5m-helio/weights/best.pt")


def dataset_by_id(csv_path=CSV_PATH):
    """Creates a dictionary of the dataset with the id as the key and the row (another dict) as the value
    Args:
        csv_path: path to the csv file of the dataset
    Returns:
        dict: dictionary of the dataset
    """
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)
    dict = {int(row["id"]):row for row in reader}
    csv_file.close()
    return dict

def id_from_im_name(im_name):
    """Extracts the image id from the image name
    Args:
        im_name: name of the image
    Returns:
        image id as an int
    """
    if "\\" in im_name: im_name = im_name.split("\\")[-1].strip("\n")
    if "/" in im_name: im_name = im_name.split("/")[-1].strip("\n")
    return int(im_name.split(".")[0][2:])
