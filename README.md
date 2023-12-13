# Urchin-Detector

## First time set up

### 1. Clone the Yolov5 repo
This Project used the ultralytics implementation of the yolo model to perform urchin detection. This requires the yolov5 repo to cloned into this repo. This can be achieved by running the following command:

```
git clone https://github.com/ultralytics/yolov5
```
### 2. Add \_\_init\_\_.py to yolov5
Add an empty file called \_\_init\_\_.py into the yolov5 folder. This allows us to import functions from yolo into our code.

### 3. Create a virtual enviroment
Create and activate a venv using the following commands:
```
python -m venv venv
venv\scripts\activate
```

### 3. Download requirements
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