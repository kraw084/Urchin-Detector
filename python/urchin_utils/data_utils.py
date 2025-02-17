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


def filter_images(image_paths, dataset, label="all", filter_var=None, filter_func=None, limit=None):
    """Filters a list of images path using a label, filter_var, and filter_func
    Args:
        image_paths: list of image paths
        dataset: dict of the dataset with id as the key and the row (another dict) as the value
        label: "all", "empty", "urchin", "kina", "centro", "helio" used to as a broad/basic filter
        limit: maximum number of images to keep, leave as none for no limit
        filter_var: csv var (from the dataset)to be passed as input to the filter function
        filter_func: function to be used to filter images, return false to skip an image
    """
    if label not in ("all", "empty", "urchin", "kina", "centro", "helio"):
        raise ValueError(f'label must be in {("all", "empty", "urchin", "kina", "centro", "helio")}')
    
    filtered_paths = []
    #filter paths using label parameter and filter_var and filter_func
    kept_count = 0  
    for path in image_paths:
        id = id_from_im_name(path)
        im_data = dataset[id]
        
        #if using filtering and the func returns false, skip this image
        if filter_var and filter_func and not filter_func(im_data[filter_var]): continue

        #if using label filtering and the label does not match, skip this image
        if label == "empty":
            if im_data["count"] != "0": continue
        elif label == "urchin":
            if im_data["count"] == "0": continue
        elif label == "kina":
            if im_data["Evechinus"].upper() == "FALSE": continue
        elif label == "centro":
             if im_data["Centrostephanus"].upper() == "FALSE": continue
        elif label == "helio":
            if im_data["Heliocidaris"].upper() == "FALSE": continue

        filtered_paths.append(path)
        kept_count += 1
        
        if limit and kept_count >= limit: break
        
    return filtered_paths


def overlapping_ids(csv1, csv2):
    """Finds and returns all the image id shared between two formatted csv datasets"""
    d1 = dataset_by_id(csv1)
    d2 = dataset_by_id(csv2)
    
    shared_ids = [id for id in d1.keys() if id in d2.keys()]
    return shared_ids


def remove_rows_from_csv(csv_path, new_csv_path, ids_to_remove):
    """Removes rows from a csv file based on a list of ids to remove
    Args:
        csv_path: path to the csv file
        ids_to_remove: list of ids to remove
    """
    #read csv file 
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)
    rows = [r for r in reader]
    csv_file.close()
    
    #remove rows with the target ids
    for i in range(len(rows) - 1, -1, -1):
        if int(rows[i]["id"]) in ids_to_remove:
            rows.pop(i)
    
    #save as a new csv
    with open(new_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    