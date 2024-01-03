import csv
import ast

def get_stats(csv_path):
    """Get stats on the data set (stats for the complete set are written in the data_info file)"""
    csv_file = open(csv_path, "r")
    reader = list(csv.DictReader(csv_file))

    print(f"Number of images: {len(reader)}")

    boxes_col = [ast.literal_eval(row["boxes"]) for row in reader]
    box_count = sum([len(boxes) for boxes in boxes_col])
    kina_count = sum([1 for boxes in boxes_col for box in boxes if box and box[0] == "Evechinus chloroticus"])
    centro_count = sum([1 for boxes in boxes_col for box in boxes if box and box[0] == "Centrostephanus rodgersii"])

    counts = [0] * 3
    for row in reader:
            boxes = ast.literal_eval(row["boxes"])
            if boxes:
                kina = False
                centro = False
                for box in boxes:
                    if box[0] == "Evechinus chloroticus": 
                        kina = True
                    else:
                        centro = True
                counts[0] += int(kina)
                counts[1] += int(centro)
                counts[2] += int(kina and centro)

    print(f"Number of images that contain kina: {counts[0]}")
    print(f"Number of images that contain centrostephanus: {counts[1]}")
    print(f"Number of images that contain both: {counts[2]}")
    print(f"Number of images with no boxes: {sum([1 for boxes in boxes_col if not boxes])}\n")

    print(f"Number of bounding boxes: {box_count}")
    print(f"Number of Kina boxes: {kina_count}")
    print(f"Number of centrostephanus boxes: {centro_count}")

    csv_file.close()

if __name__ == "__main__":
     get_stats("data/csvs/Complete_urchin_dataset_V2.csv")