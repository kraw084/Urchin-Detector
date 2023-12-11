import csv
import ast

"""Get stats on the data set (stats for the complete set are written in the data_info file)"""

f = open("data/csvs/Complete_urchin_dataset.csv", "r")
reader = list(csv.DictReader(f))

print(f"Number of images: {len(reader)}")

boxesCol = [ast.literal_eval(row["boxes"]) for row in reader]
boxCount = sum([len(boxes) for boxes in boxesCol])
kinaCount = sum([1 for boxes in boxesCol for box in boxes if box and box[0] == "Evechinus chloroticus"])
centroCount = sum([1 for boxes in boxesCol for box in boxes if box and box[0] == "Centrostephanus rodgersii"])

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
print(f"Number of images with no boxes: {sum([1 for boxes in boxesCol if not boxes])}\n")

print(f"Number of bounding boxes: {boxCount}")
print(f"Number of Kina boxes: {kinaCount}")
print(f"Number of centrostephanus boxes: {centroCount}")



f.close()