import ast
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from urchin_utils.data_utils import dataset_by_id

from get_annotation import get_annotation_dict


def get_stats(image_comments):
    """Uses comments from a manually checked validation set to calculate metrics
    Args:
        image_comments: a list of lists of comments for each image. All comments should be false-positve, missed, miss-classified, or None (if its a true positive)
    """
    num_of_images = len(image_comments)

    num_of_pred_boxes = 0
    num_of_gt_boxes = 0
    num_of_tps = 0
    num_of_fns = 0
    num_of_fps = 0
    num_of_miss_class = 0

    for comments in image_comments:
        #get counts based on tags
        tp_count = sum([1 if t is None else 0 for t in comments])
        fp_count = sum([1 if t == "false-positive" else 0 for t in comments])
        fn_count = sum([1 if t == "missed" else 0 for t in comments])
        miss_class_count = sum([1 if t == "miss-classified" else 0 for t in comments])
        total_boxes = len(comments)
        pred_boxes_count = total_boxes - fn_count
        gt_boxes_count = total_boxes - fp_count
        
        #adjust total counts
        num_of_pred_boxes += pred_boxes_count
        num_of_gt_boxes += gt_boxes_count
        num_of_tps += tp_count
        num_of_fps += fp_count
        num_of_fns += fn_count
        num_of_miss_class += miss_class_count

    #print counts
    print(f"Number of images: {num_of_images}")
    print(f"Number of true urchins: {num_of_gt_boxes}")
    print(f"Number of predicted urchins: {num_of_pred_boxes}")
    print(f"Number of true positives: {num_of_tps}")
    print(f"Number of false positives: {num_of_fps + num_of_miss_class}")
    print(f"Number of false negatives: {num_of_fns + num_of_miss_class}")
    print()

    #calculate and print metrics
    p = num_of_tps / (num_of_tps + (num_of_fps + num_of_miss_class))
    r = num_of_tps / (num_of_tps + (num_of_fns + num_of_miss_class))
    f1 = 2 * p * r / (p + r)

    print(f"Precision: {round(p, 3)}")
    print(f"Recall: {round(r, 3)}")
    print(f"F1: {round(f1, 3)}")

    mc_rate = num_of_miss_class / (num_of_tps + num_of_miss_class)
    print(f"Miss-classification rate: {round(mc_rate, 3)}")
    print()


def cvat_stats(xml_path):
    """Calculates validation stats using annotations from a cvat xml file"""
    #read xml file and format as dict
    annot_dict = get_annotation_dict(xml_path)    
    
    #extract comments from dict
    comments = []
    for k in annot_dict.keys():
        comments.append([t[0] if len(t)>0 else None for t in annot_dict[k]["tags"]])
    
    get_stats(comments)
    

def classify_comment(c):
    """Sorts the various manual comments from the big validation set into distinct categories
    Args:
        c: a string comment
    Returns:
        missed, false-positive, miss-classified, or None based on the contents of the comment
    """
    
    #defining terms that indicate each category
    fp_terms = ["false", "falst", "duplicate", "maybe"]
    missed_terms = ["missed", "misse"]
    missclass_terms = ["missclassifed", "missclassified", "misclassified", "evechinus", "centro", "kina", "helio"]  

    #check if any of the terms are in the comment, selecting the first that appears
    if any(t in c.lower() for t in missed_terms): 
        return "missed"
    elif any(t in c.lower() for t in fp_terms): 
        return "false-positive"
    elif any(t in c.lower() for t in missclass_terms): 
        return "miss-classified"
    else:
        return None


def csv_stats(csv_path):
    """Calculates validation stats using comments from a formated csv file from squidle"""
    #read dataset csv
    d = dataset_by_id(csv_path)
    
    #extract comments and apply classification function
    comments = [ast.literal_eval(row["comments"]) for row in d.values()]
    comments = [[classify_comment(c) for c in row] for row in comments]
    get_stats(comments)
    

if __name__ == "__main__":
    #cvat_stats("annotations1.xml")
    cvat_stats("annotations2.xml")
    csv_stats("data/csvs/big_val/big_val_urchin_dataset.csv")    
    