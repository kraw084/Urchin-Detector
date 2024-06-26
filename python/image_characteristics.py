import ast
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def depth_discretization(depth):
    depth = float(depth)
    minVal = 0
    maxVal = 60
    step = 2

    for i in range(minVal, maxVal, step):
        if depth >= i and depth < i + step: return i

def alt_discretization(alt):
    if alt == "": return "Empty"
    if alt == "1.0": return "1"

    alt = float(alt)
    for i in range(0, 7, 1):
        if alt >= i and alt < i + 1: return f"[{i}, {i+1})"


def contains_low_prob_box(boxes):
    boxes = ast.literal_eval(boxes)
    conf_values = [float(box[1]) for box in boxes]
    return any([val < 0.7 for val in conf_values])


def contains_low_prob_box_or_flagged(boxes):
    boxes = ast.literal_eval(boxes)
    conf_values = [float(box[1]) for box in boxes]
    return any([val < 0.7 for val in conf_values]) or any([box[6] for box in boxes])


def blur_score(im):
    h, w, _ = im.shape
    im = cv2.resize(im, (w//5, h//5))
    return cv2.Laplacian(im, cv2.CV_64F).var()

def contrast_score(im):
    h, w, _ = im.shape
    im = cv2.resize(im, (w//5, h//5))
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).std()

def brightness_score(im):
    h, w, _ = im.shape
    im = cv2.resize(im, (w//5, h//5))
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean()

def image_quality_check(im, blur_th = 300, contrast_th = 20):
    #h, w, _ = im.shape
    #im = cv2.resize(im, (w//5, h//5))
    return (blur_score(im) < blur_th) or (contrast_score(im) < contrast_th)



if __name__ == "__main__":
    f = open(f"data\datasets/full_dataset_v3/val.txt", "r")  
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    values = []
    for i, path in enumerate(image_paths):
        im = cv2.imread(path)
        h, w, _ = im.shape
        #im = cv2.resize(im, (w//5, h//5))
        score = blur_score(im)
        values.append(score)

        if score < 50:
            cv2.imshow(str(score), im)
            cv2.waitKey(0)

        print(i)


    plt.hist(values, bins=100, density=False, alpha=0.75, color='blue')
    plt.title("Histogram")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()