import os

import matplotlib.patches as patches
import cv2

from model_utils import NUM_TO_LABEL, LABEL_TO_NUM

#species bounding box colours
NUM_TO_COLOUR = [(74,237,226), (24,24,204), (3,140,252)] #yellow, red, orange in BGR
NUM_TO_COLOUR_HEX = ["#e2ed4a", "#cc1818", "#fc8c03"] #yellow, red, orange in hex


def draw_bbox(ax, bbox, im, colour):
    """draws a single bounding box on the provided matplotlib axis
    Args:
        ax: matplotlib axis to draw on
        bbox: bounding box to draw
        im: np array of image to draw on
        colour: colour of bounding box
        """
    flagged = None
    
    #extract box data based on its format
    if isinstance(bbox, tuple):
        #ground truth box from csv
        im_width, im_height = im.size
        label = bbox[0]
        confidence = bbox[1]
        x_center = bbox[2] * im_width
        y_center = bbox[3] * im_height
        box_width = bbox[4] * im_width
        box_height = bbox[5] * im_height
        if len(bbox) >= 7: flagged = bbox[6]
    else: 
        #box is an iterable with pixel coordinates in the form xywhcl
        x_center, y_center, box_width, box_height, confidence, label = bbox
        label = NUM_TO_LABEL[int(label)]

    top_left_point = (x_center - box_width/2, y_center - box_height/2)

    #draw box
    box_patch = patches.Rectangle(top_left_point, box_width, box_height, edgecolor=colour, linewidth=2, facecolor='none')
    ax.add_patch(box_patch)

    #draw label
    text = f"{label.split(' ')[0]} - {round(float(confidence), 2)}{' - F' if flagged else ''}"
    text_bbox_props = dict(pad=0.2, fc=colour, edgecolor='None')
    ax.text(top_left_point[0], top_left_point[1], text, fontsize=7, bbox=text_bbox_props, c="black", family="sans-serif")


def determine_box_colour(label, correct, missed):
    correct_colour = "#58f23d" #green
    incorrect_colour = "#cc1818" #red
    
    if correct is None and missed is None:
        #colouring by class
        col = NUM_TO_COLOUR_HEX[LABEL_TO_NUM[label]]
    elif correct or (missed is False):
        #green if pred is correct
        col = correct_colour
    elif missed or (correct is False):
        #red if pred is incorrect/missed
        col = incorrect_colour
        
    return col


def draw_bboxes(ax, bboxes, im, correct=None, boxes_missed=None):
    """draws all a set of boxes onto an image
    Args:
        ax: matplotlib axis to draw on
        bboxes: detection object (from model_utils.py)
        im: np array of image to draw on
        correct: list of bools indicating if the prediction is correct (will draw the box in green)
        boxes_missed: list of bools indicating if the prediction is missed (will draw the box in red)
    """
    #draw each box onto the axis
    for i, bbox in enumerate(bboxes.gen(box_format="xywhcl")):
        col = determine_box_colour(bbox[0], correct[i], boxes_missed[i])
        draw_bbox(ax, bbox, im, col)
    
        
def annotate_image(im, bboxes, thickness=2, font_size=0.75, draw_labels=True):
        """Draws bounding boxes onto a single cv2 image that can be saved
        Args:
            im: np array of image to draw on
            bboxes: detection object (from model_utils.py)
            thickness: thickness of the bounding box and label text
            font_size: font size of the label text
            draw_labels: boolean indicating if the label should be drawn (or just the boxes)
        """

        label_data = []
        for box in bboxes.gen(box_format="xyxycl"):
            x_top_left, y_top_left, x_bottom_right, y_bottom_right = box[:4]
            label = NUM_TO_LABEL[int(box[5])]
            label = f"{label[0]}. {label.split()[1]}"

            colour = NUM_TO_COLOUR[int(box[5])]

            #Draw boudning box
            im = cv2.rectangle(im, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), colour, thickness)

            #Add label to draw later, after all boxes have been drawn
            label_data.append((f"{label} - {float(box[4]):.2f}", (x_top_left, y_top_left), colour))

        #Draw text over boxes
        if draw_labels:
            for data in label_data:
                #calculate text dimensions
                text_size = cv2.getTextSize(data[0], cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
                text_box_top_left = (data[1][0], data[1][1] - text_size[1])
                text_box_bottom_right = (data[1][0] + text_size[0], data[1][1])
                
                #draw text box and text
                im = cv2.rectangle(im, text_box_top_left, text_box_bottom_right, data[2], -1)
                im = cv2.putText(im, data[0], data[1], cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness - 1, cv2.LINE_AA)


def annotate_preds_on_folder(model, input_folder, output_folder, draw_labels=True):
    """Runs the model on all images in a folder and saves the annotated images to a new folder"""  
    for im_name in os.listdir(input_folder):
        preds = model(input_folder + "/" + im_name)
        im = cv2.imread(input_folder + "/" + im_name)
        annotate_image(im, preds, draw_labels=draw_labels)
        cv2.imwrite(output_folder + "/" + im_name, im)