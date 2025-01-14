import os
import sys
import importlib

import torch
import cv2
import numpy as np


try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from YOLOX.tools.demo import Predictor
except ModuleNotFoundError:
    print("YOLOX not found")
    

#index to species name translation and vice versa
NUM_TO_LABEL = ["Evechinus chloroticus","Centrostephanus rodgersii", "Heliocidaris erythrogramma"]
LABEL_TO_NUM = {label: i for i, label in enumerate(NUM_TO_LABEL)}

#Path to the current best model
WEIGHTS_PATH = os.path.abspath("models/yolov5m-helio/weights/best.pt")

def project_sys_path():
    """Add the Urchin-Detector folder to the sys path so functions from yolo can be imported"""
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_dir)


def check_cuda_availability():
    """Check if cuda is available and print relavent info"""
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    

def load_yolov5_model(weights_path=WEIGHTS_PATH, cuda=True):
    """Load and return a yolov5 model
    Args:
        weights_path: path to the weights file
        cuda: whether to load the model on the gpu
    Returns:
        YOLOv5 model
    """
    model = torch.hub.load("yolov5", "custom", path=weights_path, source="local")
    model.cuda() if cuda else model.cpu()
    return model

    
class Detection:
    """Object representing a collection of bounding boxes (e.g. an output from a model). 
    Should create a subclass for each type of model and define convert_pred_to_array method.
    bounding boxes are stored as a list of np arrays of the form [x_center, y_center, w, h, conf, class] in pixels
    but can be retrieved in many differet formats.
    """
    def __init__(self, model_pred):
        """
        Args:
            model_pred: output from a model
        """
        self.dets = self.convert_pred_to_array(model_pred)
        self.count = len(self.dets)
        
    def convert_pred_to_array(self):
        """Should be implemented in a subclass. Should convert the output from a model to 
        a list of np arrays of the form [x_center, y_center, w, h, conf, class] in pixels"""
        pass

    def xywhcl(self, index):
        """Returns a single box as an np array of the form [x_center, y_center, w, h, conf, class] in pixels"""
        return self.dets[index]
    
    def xyxycl(self, index):
        """Returns a single box as an np array of the form [x_top_left, y_top_left, x_bottom_right, y_bottom_right, conf, class] in pixels"""
        box = self.dets[index][:4]
        x, y, w, h = box[:4]
        return np.array([x - w//2, y - h//2, x + w//2, y + h//2, box[4], box[5]])
    
    
    def gen(self, box_format="xywhcl"):
        """Used to create an iterator that returns the boxes in the chosen format"""
        box_get_method = None
        if box_format == "xywhcl":
            box_get_method = self.xywhcl
        elif box_format == "xyxycl":
            box_get_method = self.xyxycl
            
        for i in range(self.count):
            yield box_get_method(i)
    
    
    def __getitem__(self, index):
        """Use for indexing using [ ] notation. Returns boxes in the xywhcl format"""
        return self.dets[index]
    
    def __iter__(self):
        """Defines the iterator for the class which returns boxes in the xywhcl format"""
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < self.count:
            i = self.i
            self.i += 1
            return self.dets[i]
        else:
            raise StopIteration


class Detection_YoloV5(Detection):
    """Subclass of Detection for yolov5 models"""
    def convert_pred_to_array(self, model_pred):
        return [box for box in model_pred.xywh[0].cpu().numpy()]


class Detection_YoloX(Detection):
    def __init__(self, model_pred, im_ratio):
        """
        Args:
            model_pred: output from a model
            im_ratio: ratio of the size of the orignal image to the input image size, used for scaling
        """
        self.dets = self.convert_pred_to_array(model_pred, im_ratio)
        self.count = len(self.dets)
    
    """Subclass of Detection for yolox models"""
    def convert_pred_to_array(self, model_pred, ratio):
        if pred is None:
            return []
        pred = pred.cpu().numpy()
        pred[:, :4] = pred[:, :4] / ratio
        
        formatted_pred = []
        for bbox in pred:            
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            conf = bbox[4] * bbox[5]
            label = bbox[6]
            formatted_pred.append(np.array([x_center, y_center, w, h, conf, label]))
            
        return formatted_pred


class UrchinDetector_YoloV5:
    """Wrapper class for the yolov5 model"""
    def __init__(self, 
                 weight_path=WEIGHTS_PATH, 
                 conf=0.45, 
                 iou=0.6, 
                 img_size=1280, 
                 cuda=None, 
                 classes=NUM_TO_LABEL
                 ):
        """
        Args:
            weight_path: path to the weights file
            conf: confidence threshold
            iou: nms iou threshold
            img_size: image size that the model takes as input
            cuda: whether to load the model on the gpu
            classes: list of classes that the model detects
        """
        
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not (cuda is None) else torch.cuda.is_available()

        self.model = load_yolov5_model(self.weight_path, self.cuda)
        self.model.conf = self.conf
        self.model.iou = self.iou
        
        self.classes = classes

    def update_parameters(self, conf=0.45, iou=0.6):
        """Update the two parameters of the model
        Args:
            conf: new confidence threshold
            iou: new nms iou threshold
        """
        self.conf = conf
        self.model.conf = conf
        self.iou = iou
        self.model.iou = iou

    def predict(self, im):
        """Runs the yolov5 model on a single image and returns the detection
        Args:
            im: image to run the model on
        Returns:
            Detection_YoloV5 object
        """
        results = self.model(im, size = self.img_size)
        det = Detection_YoloV5(results)
        return det

    def __call__(self, im):
        """Shorthand for calling the predict method"""
        return self.predict(im)
    
    
class UrchinDetector_YOLOX:
    """Wrapper class for the yoloX model"""
    def __init__(self, 
                 weight_path=WEIGHTS_PATH, 
                 conf=0.2, 
                 iou=0.6, 
                 img_size=1280, 
                 cuda=None, 
                 exp_file_name="yolox_urchin_m", 
                 classes=NUM_TO_LABEL
                 ):
        """
        Args:
            weight_path: path to the weights file
            conf: confidence threshold
            iou: nms iou threshold
            img_size: image size that the model takes as input
            cuda: whether to load the model on the gpu
            exp_file_name: name of the YOLOX experiment file
            classes: list of classes that the model detects
        """
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not cuda is None else torch.cuda.is_available()

        #find the yoloX experiment file and set parameters
        yolox_exp_module = importlib.import_module(f"YOLOX.exps.custom.{exp_file_name}")
        exp_class = getattr(yolox_exp_module, "Exp")
        self.exp = exp_class()
        self.exp.test_conf = self.conf
        self.exp.nmsthre = self.iou
        self.exp.input_size = (img_size, img_size)
        self.exp.test_size = self.exp.input_size

        #load the model
        model = self.exp.get_model()
        ckpt_file = self.weight_path
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        device = "gpu" if self.cuda else "cpu"
        if device == "gpu":
            model.cuda()
        model.eval()

        self.model = Predictor(model, self.exp, NUM_TO_LABEL, device=device, legacy=False)
        self.update_parameters(self.conf, self.iou)
        self.model.test_size = self.exp.input_size
        
        self.classes = classes

    def update_parameters(self, conf=0.2, iou=0.6):
        """Update the two parameters of the model
        Args:
            conf: new confidence threshold
            iou: new nms iou threshold
        """
        self.conf = conf
        self.exp.test_conf = conf
        self.model.confthre = conf

        self.iou = iou
        self.exp.nmsthre = iou
        self.model.nmsthre = iou

    def predict(self, im):
        """Runs the yoloX model on a single image and returns the detection
        Args:
            im: path to image to run the model on
        Returns:
            Detection_YoloX object
        """
        im = cv2.imread(im)
        pred = self.model.inference(im)[0][0]
        im_size_ratio = min(self.exp.test_size[0] / im.shape[0], self.exp.test_size[1] / im.shape[1])
        
        return Detection_YoloX(pred, im_size_ratio)
  
    def __call__(self, im):
        """Shorthand for calling the predict method"""
        return self.predict(im)
    



