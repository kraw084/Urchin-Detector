import os
import sys
import torch
import cv2
import numpy as np
import importlib

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from YOLOX.tools.demo import Predictor
except ModuleNotFoundError:
    print("YOLOX not found")
    

#index to species name translation and vice versa
NUM_TO_LABEL = ["Evechinus chloroticus","Centrostephanus rodgersii", "Heliocidaris erythrogramma"]
LABEL_TO_NUM = {label: i for i, label in enumerate(NUM_TO_LABEL)}

class Detection:
    def __init__(self, model_pred, im_name):
        self.dets = self.convert_pred_to_array(model_pred)
        self.count = len(self.dets)
        self.image_name = im_name
        
    def convert_pred_to_array(self):
        pass

    
    def xywhcl(self, index):
        return self.dets[index]
    
    def xyxycl(self, index):
        box = self.dets[index][:4]
        x, y, w, h = box[:4]
        return np.array([x - w//2, y - h//2, x + w//2, y + h//2, box[4], box[5]])
    
    
    def gen(self, box_format="xywhcl"):
        box_get_method = None
        if box_format == "xywhcl":
            box_get_method = self.xywhcl
        elif box_format == "xyxycl":
            box_get_method = self.xyxycl
            
        for i in range(self.count):
            yield box_get_method(i)
    
    
    def __getitem__(self, index):
        return self.dets[index]
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < self.count:
            i = self.i
            self.i += 1
            return self.dets[i]
        else:
            raise StopIteration



def check_cuda_availability():
    """Check if cuda is available and print relavent info"""
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def project_sys_path():
    """add the Urchin-Detector folder to the sys path so functions from yolo can be imported"""
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_dir)


def load_model(weights_path=WEIGHTS_PATH, cuda=True):
    """Load and return a yolo model"""
    model = torch.hub.load("yolov5", "custom", path=weights_path, source="local")
    model.cuda() if cuda else model.cpu()
    return model




class UrchinDetector_YoloV5:
    """Wrapper class for the yolov5 model"""
    def __init__(self, weight_path=WEIGHTS_PATH, conf=0.45, iou=0.6, img_size=1280, cuda=None, classes=NUM_TO_LABEL, plat_scaling = False):
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not (cuda is None) else torch.cuda.is_available()
        self.scaling = plat_scaling

        self.model = load_model(self.weight_path, self.cuda)
        self.model.conf = self.conf
        self.model.iou = self.iou
        
        self.classes = classes

    def update_parameters(self, conf=0.45, iou=0.6):
        self.conf = conf
        self.model.conf = conf
        self.iou = iou
        self.model.iou = iou

    def predict(self, im):
        results = self.model(im, size = self.img_size)
        if self.scaling:
            with torch.inference_mode():
                for pred in results.pred[0]:
                    pred[4] = plat_scaling(pred[4])
            results.__init__(results.ims, pred=results.pred, files=results.files, times=results.times, names=results.names, shape=results.s)
        return results

    def __call__(self, im):
        return self.xywhcl(im)
    
    def xywhcl(self, im):
        pred = self.predict(im).xywh[0].cpu().numpy()
        return [box for box in pred]
    
class UrchinDetector_YOLOX:
    """Wrapper class for the yolov5 model"""
    def __init__(self, weight_path=WEIGHTS_PATH, conf=0.2, iou=0.6, img_size=1280, cuda=None, exp_file_name="yolox_urchin_m", classes=NUM_TO_LABEL):
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not cuda is None else torch.cuda.is_available()

        
        yolox_exp_module = importlib.import_module(f"YOLOX.exps.custom.{exp_file_name}")
        exp_class = getattr(yolox_exp_module, "Exp")
        self.exp = exp_class()
        self.exp.test_conf = self.conf
        self.exp.nmsthre = self.iou
        self.exp.input_size = (img_size, img_size)
        self.exp.test_size = self.exp.input_size

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
        self.conf = conf
        self.exp.test_conf = conf
        self.model.confthre = conf

        self.iou = iou
        self.exp.nmsthre = iou
        self.model.nmsthre = iou

    def predict(self, im):
        im = cv2.imread(im)
        pred = self.model.inference(im)[0][0]
        if pred is None:
            return []
        pred = pred.cpu().numpy()

        im_size_ratio = min(self.exp.test_size[0] / im.shape[0], self.exp.test_size[1] / im.shape[1])
        pred[:, :4] = pred[:, :4] / im_size_ratio
        
        return pred

    def __call__(self, im):
        return self.xywhcl(im)
    
    def xywhcl(self, im):
        pred = self.predict(im)
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


def read_txt(images_txt):
    f = open(images_txt, "r")
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    return image_paths


def process_images_input(images):
    return images if isinstance(images, list) else read_txt(images) 


def complement_image_set(images0, images1):
    """Returns all the image paths in images1 that are not in images0"""
    images0 = process_images_input(images0)
    images1 = process_images_input(images1)

    return [x for x in images1 if x not in images0]


def filter_txt(txt_path, txt_output_name, var_name, exclude=None):
    if exclude is None: exclude = []
    im_paths = process_images_input(txt_path)
    dataset = dataset_by_id()
    kept_images = []
    for im in im_paths:
        id = id_from_im_name(im)
        data_row = dataset[id]

        if data_row[var_name] not in exclude: kept_images.append(im)

    f = open(txt_output_name, "w")
    f.write("\n".join(kept_images))
    f.close()


def plat_scaling(x):
    #Platt scaling function for highres-ro v3 model
    cubic = -7.3848* x**3 +13.5284 * x**2 -6.2952 *x + 1.0895
    linear = 0.566 * x + 0.027

    return cubic if x >=0.45 else linear
    

