import csv
import ast

"""Used to create the label files in the format specified by the Yolov5 Implementation"""

csvPath = "data/csvs/Complete_urchin_dataset.csv"
csvFile = open(csvPath, "r")
reader = csv.DictReader(csvFile)

for row in reader:
    id = row["id"]
    boxes = ast.literal_eval(row["boxes"])
    if boxes:
        print(f"Create label txt for im{id}")
        labelFile = open(f"data/labels/im{id}.txt", "w")
        for box in boxes:
            #yolo format: class xCenter yCenter width height
            classLabel = 0 if box[0] == "Evechinus chloroticus" else 1
            toWrite = f"{classLabel} {box[2]} {box[3]} {box[4]} {box[5]}\n"
            labelFile.write(toWrite)
        labelFile.close()
print("----- FINISHED -----")

csvFile.close()