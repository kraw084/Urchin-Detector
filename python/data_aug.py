import cv2
import numpy as np
import random


hyp = {"blur": 0.5}
img = cv2.imread("data/images/im0.JPG")
h, w, _ = img.shape
img = cv2.resize(img, (w//5, h//5))
cv2.imshow("img", img)
cv2.waitKey(0)

# copy and past this into yolov5/utils/dataloaders.py in LoadImagesAndLabels.__getitem__ along side the other augmentations
if "blur" in hyp and random.random() < hyp['blur']:
    h, w, _ = img.shape
    kernal_size = w//10 if w//10 % 2 == 1 else w//10 + 1
    sigma = (random.random() * 1.9) + 0.1
    img = cv2.GaussianBlur(img, (kernal_size, kernal_size), sigmaX=sigma, sigmaY=sigma)

cv2.imshow("img", img)
cv2.waitKey(0)