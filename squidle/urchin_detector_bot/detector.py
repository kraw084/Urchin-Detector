from sqapi.annotate import Annotator
from sqapi.media import SQMediaObject
import torch
from PIL import Image
import cv2

class UrchinDetectorBot(Annotator):
    def __init__(self, weights_path: str, device: str = "cpu", img_size: int = 1280, conf: float = 0.45, iou_th: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.model = torch.hub.load("yolov5", "custom", path=weights_path)
        self.model.cpu() if device == "cpu" else self.model.cuda()
        self.model.conf = conf
        self.model.iou = iou_th
        self.img_size = img_size
        self.device = device

    def tensor2polygon(self, box):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])

        return [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]


    def detect_points(self, mediaobj: SQMediaObject):
        if not mediaobj.is_processed:
            orig_image = mediaobj.data()
            img = mediaobj.data(Image.fromarray(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)))
        else:
            img = mediaobj.data()

        predictions = self.model(img, size = self.img_size).xyxy[0]

        points = []
        for pred in predictions:
            polygon = self.tensor2polygon(pred)
            likelihood = pred[4]
            class_code = int(pred[5])
            p = self.create_annotation_label_point_px(class_code, likelihood, polygon=polygon, width=mediaobj.width, height=mediaobj.height)
            points.append(p)

        return points