import os
import csv

#File paths
CSV_PATH = os.path.abspath("data/csvs/High_conf_clipped_dataset_V5.csv")
DATASET_YAML_PATH = os.path.abspath("data/datasets/full_dataset_v5/datasetV5.yaml")



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


def read_txt(images_txt):
    """Reads a txt file of image paths and returns the list
    Args: 
        images_txt: path to the txt file of image names, each separated by a newline
    Returns:
        list: list of image paths
    """
    f = open(images_txt, "r")
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    return image_paths


def process_images_input(images):
    """Used to generalise image input so that it can be either a txt file of image paths or a list of image paths
    Args:
        images: either a txt file of image paths or a list of image paths
    Returns:
        list: list of image paths
    """
    return images if isinstance(images, list) else read_txt(images) 


def complement_image_set(images0, images1):
    """Returns all the image paths in images1 that are not in images0"""
    images0 = process_images_input(images0)
    images1 = process_images_input(images1)

    return [x for x in images1 if x not in images0]


