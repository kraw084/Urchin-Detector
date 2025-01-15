
import cv2
import matplotlib
import matplotlib.pyplot as plt

from utils.data_utils import process_images_input, id_from_im_name, filter_images
from utils.model_utils import gt_to_detection
from utils.eval_utils import correct_predictions
from utils.vis_utils import plot_im_and_boxes

def compare_to_gt(model, 
                  images, 
                  dataset,
                  label = "urchin", 
                  save_path = False, 
                  limit = None, 
                  filter_var = None, 
                  filter_func = None, 
                  display_correct = False, 
                  cuda=True,
                  iou_val = 0.5, 
                  ):
    """Creates figures to visually compare model predictions to the actual annotations
        model: model object to generate predictions with
        images: txt or list of image paths
        label: "all", "empty", "urchin", "kina", "centro", "helio" used to filter what kind of images are compared
        save_path: if this is a file path, figures will be saved instead of shown
        limit: number of figures to show/save, leave as none for no limit
        filter_var: csv var to be passed as input to filter function
        filter_func: function to be used to filter images, return false to skip an image
    """
    if label not in ("all", "empty", "urchin", "kina", "centro", "helio"):
        raise ValueError(f'label must be in {("all", "empty", "urchin", "kina", "centro", "helio")}')

    image_paths = process_images_input(images)
    filtered_paths = filter_images(image_paths, dataset, label, filter_var, filter_func, limit)

    #loop through all the filtered images and display them with gt and predictions drawn
    for i, im_path in enumerate(filtered_paths):
        id = id_from_im_name(im_path)
        gt = gt_to_detection(dataset[id])
        
        #Set up figure and subplots
        matplotlib.use('TkAgg')
        fig, axes = plt.subplots(1, 2, figsize = (14, 6))
        fig.suptitle(f"{im_path}\n{i + 1}/{len(filtered_paths)}")

        #format image meta data to display at the bottom of the fig
        row_dict = dataset[id]
        row_dict.pop("boxes", None)
        row_dict.pop("url", None)
        row_dict.pop("id", None)
        text = str(row_dict)[1:-1].replace("'", "").split(",")

        cutoff_index = 7
        text = f"{'    '.join(text[:cutoff_index])} \n {'    '.join(text[cutoff_index:])}"
        fig.text(0.5, 0.05, text, ha='center', fontsize=10)
        
        #read image and convert to RGB for matplotlib display
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        #Generate predictions
        prediction = model(im_path)
  

        #Determine predicition correctness if display_correct is True
        if display_correct:
            correct, boxes_missed = correct_predictions(gt, prediction, iou_val=iou_val)
        else:
            correct = None
            boxes_missed = None
        
        #plot ground truth boxes
        ax = axes[0]
        ax.set_title(f"Ground truth ({len(gt)})")
        plot_im_and_boxes(im, gt, ax, correct=None, boxes_missed=boxes_missed)
            
        #plot predicted boxes
        ax = axes[1]
        ax.set_title(f"Prediction ({len(prediction)})")
        plot_im_and_boxes(im, prediction, ax, correct=correct, boxes_missed=None)
  
        #Save or show figure
        if not save_path:
            plt.show()
        else:
            fig.savefig(f"{save_path}/fig-{id}.png", format="png", bbox_inches='tight')
