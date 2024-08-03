import csv
import ast

def get_stats(csv_path, classes=["Evechinus chloroticus", "Centrostephanus rodgersii"]):
    """Get stats on the data set (stats for the complete set are written in the data_info file)"""
    csv_file = open(csv_path, "r")
    reader = list(csv.DictReader(csv_file))

    print(f"Number of images: {len(reader)}")

    image_counts = [0] * (len(classes) + 1)
    box_counts = [0] * (len(classes))
    prob_less_than_1 = 0
    for row in reader:
        boxes = ast.literal_eval(row["boxes"])
        contains = [False] * len(classes)
        if boxes:
            for box in boxes:
                for i, c in enumerate(c):
                    if box[1] < 1: prob_less_than_1 += 1

                    if box[0] == c:
                        box_counts[i] += 1
                        contains[i] = True
                        break
                    

        if not any(contains):
            image_counts[-1] += 1

        for i, contain in enumerate(contain):
            if contain:
                image_counts[i] += 1




    for i, c in enumerate(classes):
        print(f"Number of images that contain {c}: {image_counts[i]}")
        
    print(f"Number of images that contain no urchins: {image_counts[-1]}")

    print(f"Number of bounding boxes: {sum(box_counts)}")

    for i, c in enumerate(classes):
        print(f"Number of {c} boxes: {box_counts[i]}")

    print(f"Number of boxes with liklihood less than 1: {prob_less_than_1}")

    csv_file.close()

def split_instance_count(csv_path, txt):
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)
    dict = {int(row["id"]):row for row in reader}
    csv_file.close()

    f = open(txt, "r")
    txt_ids = [int(row.split(".")[0].split("/")[-1][2:]) for row in f.readlines()]

    kina_count = 0
    centro_count = 0

    for id in txt_ids:
         data_row = dict[id]
         for box in ast.literal_eval(data_row["boxes"]):
              if box[0] == "Evechinus chloroticus": kina_count += 1
              if box[0] == "Centrostephanus rodgersii": centro_count += 1

    print(f"Kina instances: {kina_count}")
    print(f"Centro instances: {centro_count}")

if __name__ == "__main__":
    #get_stats("data/csvs/Complete_urchin_dataset_V4.csv")

    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "train0.txt")
    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "val0.txt")
    print("--------")
    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "train1.txt")
    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "val1.txt")
    print("--------")
    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "train2.txt")
    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "val2.txt")
    print("--------")
    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "train3.txt")
    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "val3.txt")
    print("--------")
    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "train4.txt")
    split_instance_count("data/csvs/High_conf_clipped_dataset_V4.csv", "val4.txt")
    print("--------")
   