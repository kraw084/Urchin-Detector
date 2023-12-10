import csv
import ast

"""Get stats on the data set (stats for the complete set are written in the data_info file)"""

f = open("Complete_urchin_dataset.csv", "r")
reader = list(csv.DictReader(f))

print(f"Number of images: {len(reader)}")

boxesCol = [ast.literal_eval(row["boxes"]) for row in reader]
boxCount = sum([len(boxes) for boxes in boxesCol])
kinaCount = sum([1 for boxes in boxesCol for box in boxes if box and box[0] == "Evechinus chloroticus"])
centroCount = sum([1 for boxes in boxesCol for box in boxes if box and box[0] == "Centrostephanus rodgersii"])

print(f"Number of bounding boxes: {boxCount}")
print(f"Number of Kina boxes: {kinaCount}")
print(f"Number of centrostephanus boxes: {centroCount}")

print(f"Number of images with no boxes: {sum([1 for boxes in boxesCol if not boxes])}")

f.close()