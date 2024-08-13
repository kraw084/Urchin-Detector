# Urchin-Detector
A repository for training, testing and analysing sea urchin object detection models. All models are availble in the models folder.

The current best model (models/yolov5m-highRes-ro-v4) can accurately detect and classify and locate three sea urchin species found in Austrailia and New Zealand.

Currently, the error analysis code supports YoloV5 and YoloX models but custom models can be easily added by creating wrapper classes that implement the xywhcl() function to produce a list of numpy array bounding boxes with the form x_center, y_center, width, height, confidence, and label (see python/urchin_utils.py for more details).

**Performance on the test set:**
|         | P     | R     | F1    | mAP50 | mAP50:95 |
| ------- | ----- | ----- | ----- | ----- | -------- |
| **Kina**    | 0.914 | 0.903 | 0.908 | 0.935 | 0.575    |
| **Centro**  | 0.877 | 0.837 | 0.857 | 0.887 | 0.492    |
| **Helio**   | 0.921 | 0.827 | 0.871 | 0.892 | 0.580    |
| **Average** | 0.904 | 0.856 | 0.879 | 0.905 | 0.549    |

*Evaluated with conf=0.45 and nms_iou=0.6*


## First time set up

#### 1. Clone the Yolov5 repo
This Project used the ultralytics implementation of the yolo model to perform urchin detection. This requires the yolov5 repo to cloned into this repo. This can be achieved by running the following command:

```
git clone https://github.com/ultralytics/yolov5
```
#### 2. Create a virtual enviroment
Create and activate a venv using the following commands:
```
python -m venv venv
venv\scripts\activate
```
#### 3. Download requirements
If your GPU supports CUDA install pytorch by using the following command:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
If your GPU doesnt support CUDA then pytorch will be installed when downloading the rest of the requirements, as detailed below. 

Install all the other project requirements using the following command (this includes the yolov5 requirments):
```
pip install -r requirments.txt
```
If there are any import issues (e.g. modules not found) try installing yolov5 requirements directly using ```pip install -r yolov5/requirments.txt```



### 