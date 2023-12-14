import os
import ast
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img

from urchin_utils import get_dataset_rows, id_from_im_name, draw_bboxes, load_model, WEIGHTS_PATH

def detect(model, source):
    """Runs images through urchin detection model, returns results as a list of pandas dataframes
       source can be a single image path, list of image path or a directory path"""
    if not (isinstance(source, list) or os.path.isfile(source)):
        #source is a dir
        source = [os.path.join(source, im_name) for im_name in os.listdir(source)]

    results = model(source if isinstance(source, list) else [source])
    results = [r.pandas().xywh[0] for r in results.tolist()]
    return results

def compare_to_gt(model, txt_of_im_paths, label = "urchin", save_path = False, limit = None):
    """Creates figures to compare model predictions to the actual labels
        model: yolo model to run
        txt_of_im_paths: path of a txt file containing image paths
        label: "all", "empty", "urchin", "kina", "centro", used to filter what kind of images are compared
        save_path: if this is a file path, figures will be saved instead of shown
        limit: number of figures to show/save, leave as none for no limit
    """
    if label not in ("all", "empty", "urchin", "kina", "centro"):
        raise ValueError(f'label must be in {("all", "empty", "urchin", "kina", "centro")}')

    rows = get_dataset_rows()

    txt_file = open(txt_of_im_paths, "r")
    im_paths = txt_file.readlines()

    for im_path in im_paths:
        id = id_from_im_name(im_path)
        boxes = ast.literal_eval(rows[id]["boxes"])

        if label == "empty":
            if boxes: continue
        elif label == "urchin":
            if not boxes: continue
        elif label == "kina":
            if not boxes or boxes[0][0] == "Centrostephanus rodgersii": continue
        elif label == "centro":
            if not boxes or boxes[0][0] == "Evechinus chloroticus": continue

        matplotlib.use('TkAgg')
        fig = plt.figure(figsize=(14, 8))
        im = img.imread(im_path.strip("\n"))

        #plot ground truth boxes
        ax = fig.add_subplot(1, 2, 1)
        plt.title("Ground truth")
        plt.imshow(im)
        draw_bboxes(ax, boxes, im)
            
        #plot predicted boxes
        ax = fig.add_subplot(1, 2, 2)
        plt.title("Prediction")
        plt.imshow(im)
        prediction = detect(model, im_path.strip("\n"))[0]
        draw_bboxes(ax, prediction, im)

        if not save_path:
            plt.show()
        else:
            fig.savefig(f"{save_path}/fig-{id}.png", format="png", bbox_inches='tight')

        if limit:
            limit -= 1
            if limit <= 0: break

if __name__ == "__main__":
    model = load_model(WEIGHTS_PATH)

    compare_to_gt(model, 
                  "data/datasets/full_dataset/val.txt", 
                  label ="centro", 
                  save_path = "C:\\Users\\kelha\\Documents\\Uni\\Summer Research\\Urchin-Detector",
                  limit = 10)

    

