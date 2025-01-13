import csv
import ast

urchin_species = ["Evechinus chloroticus", "Centrostephanus rodgersii", "Heliocidaris erythrogramma"]

def yolo_labels(csv_path, label_dest_dir, conf_thresh = 0, include_flagged = False):
    """Used to create the label files in the format specified by the Yolov5 Implementation
    Args:
        csv_path: path to the csv file of the dataset
        label_dest_dir: path to the directory where the label files will be created
        conf_thresh: minimum confidence threshold for a box to be included
    """
    
    #read dataset csv
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)

    for row in reader:
        id = row["id"]
        boxes = ast.literal_eval(row["boxes"])
        
        #if at least one box meets the confidence threshold and is not flagged
        if boxes and any([box[1] >= conf_thresh for box in boxes]) and (include_flagged or any([not box[6] for box in boxes])):
            print(f"Create label txt for im{id}")
            label_file = open(f"{label_dest_dir}/im{id}.txt", "w")
            for box in set(boxes):
                if box[1] < conf_thresh or box[6]: continue #skip boxes with low confidence

                #yolo format: class xCenter yCenter width height
                class_label = 0 
                
                #determine class id from species name
                class_label = urchin_species.index(box[0])
                    
                #write to label file
                to_write = f"{class_label} {box[2]} {box[3]} {box[4]} {box[5]}\n"
                label_file.write(to_write)
            label_file.close()
    
    print("----- FINISHED -----")
    csv_file.close()

if __name__ == "__main__":
    yolo_labels("data/csvs/High_conf_clipped_dataset_V5.csv", "data/labels", 0.7)