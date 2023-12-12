import sys
import os

#add the Urchin-Detector folder to the sys path so functions from yolo can be imported
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import yolov5.train
import yolov5.val


if __name__ == "__main__":
    yolov5.train.run(img = 640, epochs = 150, data = "datasetOverfit.yaml", weights = "yolov5s.pt")

    #yolov5.val.run(data = "yolov5/data/datasetOverfit.yaml", weights = "yolov5/runs/train/exp2/weights/best.pt", task = "test")