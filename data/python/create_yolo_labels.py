import csv
import ast

def yolo_labels(csv_path, label_dest_dir):
    """Used to create the label files in the format specified by the Yolov5 Implementation"""
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)

    for row in reader:
        id = row["id"]
        boxes = ast.literal_eval(row["boxes"])
        if boxes:
            print(f"Create label txt for im{id}")
            label_file = open(f"{label_dest_dir}/im{id}.txt", "w")
            for box in boxes:
                #yolo format: class xCenter yCenter width height
                class_label = 0 if box[0] == "Evechinus chloroticus" else 1
                to_write = f"{class_label} {box[2]} {box[3]} {box[4]} {box[5]}\n"
                label_file.write(to_write)
            label_file.close()
    print("----- FINISHED -----")

    csv_file.close()

if __name__ == "__main__":
    yolo_labels("data/csvs/Complete_urchin_dataset.csv", "data/labels")