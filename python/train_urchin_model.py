import sys
import os
from urchin_utils import project_sys_path, DATASET_YAML_PATH

project_sys_path()
import yolov5


if __name__ == "__main__":
    yolov5.train.run(img = 640, epochs = 150, data = DATASET_YAML_PATH, weights = "yolov5s.pt", save_period = 40)
    #yolov5.val.run(data = dataPath, weights = "yolov5/runs/train/exp4/weights/best.pt", task = "val")