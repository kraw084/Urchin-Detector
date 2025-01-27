from get_annotation import get_annotation_dict

#format cvat xml annotations as a dict
annotation_data_dict = get_annotation_dict ('annotations.xml')

num_of_images = len(annotation_data_dict.keys())

num_of_pred_boxes = 0
num_of_gt_boxes = 0
num_of_tps = 0
num_of_fns = 0
num_of_fps = 0
num_of_miss_class = 0

for im in annotation_data_dict.keys():
    boxes = annotation_data_dict[im]["boxes"]
    labels = annotation_data_dict[im]["labels"]
    tags = annotation_data_dict[im]["tags"]
    
    formated_tags = [t[0] if len(t)>0 else None for t in tags]
    
    #get counts based on tags
    tp_count = sum([1 if t is None else 0 for t in formated_tags])
    fp_count = sum([1 if t == "false-positive" else 0 for t in formated_tags])
    fn_count = sum([1 if t == "missed" else 0 for t in formated_tags])
    miss_class_count = sum([1 if t == "miss-classified" else 0 for t in formated_tags])
    total_boxes = len(boxes)
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

#calculate and print metrics
p = num_of_tps / (num_of_tps + (num_of_fps + num_of_miss_class))
r = num_of_tps / (num_of_tps + (num_of_fns + num_of_miss_class))
f1 = 2 * p * r / (p + r)
print()
print(f"Precision: {round(p, 3)}")
print(f"Recall: {round(r, 3)}")
print(f"F1: {round(f1, 3)}")

mc_rate = num_of_miss_class / num_of_pred_boxes
print(f"Miss-classification rate: {round(mc_rate, 3)}")