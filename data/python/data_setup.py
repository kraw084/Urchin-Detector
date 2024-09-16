import os
from download_images import download_imgs
from exif_correction import remove_exif_data
from create_yolo_labels import yolo_labels

os.mkdir("data/images")
download_imgs("data/csvs/High_conf_clipped_dataset_V5.csv", "data/images")

remove_exif_data("data/images")

os.mkdir("data/labels")
yolo_labels("data/csvs/High_conf_clipped_dataset_V5.csv", "data/labels", 0.7)
