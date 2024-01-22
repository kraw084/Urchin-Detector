import csv
import ast

def yolo_labels(csv_path, label_dest_dir, conf_thresh = 0):
    """Used to create the label files in the format specified by the Yolov5 Implementation"""
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)

    for row in reader:
        id = row["id"]
        boxes = ast.literal_eval(row["boxes"])
        if boxes and any([box[1] >= conf_thresh for box in boxes]) and any([not box[6] for box in boxes]):
            print(f"Create label txt for im{id}")
            label_file = open(f"{label_dest_dir}/im{id}.txt", "w")
            for box in set(boxes):
                if box[1] < conf_thresh or box[6]: continue #skip boxes with low confidence

                #yolo format: class xCenter yCenter width height
                class_label = 0 if box[0] == "Evechinus chloroticus" else 1
                to_write = f"{class_label} {box[2]} {box[3]} {box[4]} {box[5]}\n"
                label_file.write(to_write)
            label_file.close()
    print("----- FINISHED -----")

    csv_file.close()

if __name__ == "__main__":
    yolo_labels("data/csvs/Complete_urchin_dataset_V3.csv", "data/test_labels", 0.7)