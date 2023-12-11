import os
import random
import csv
import ast

"""Seperate the dataset into train, validation and test sets"""

imageNames = os.listdir("data/images")

csvFile = open("data/csvs/Complete_urchin_dataset.csv")
reader = csv.DictReader(csvFile)
csvList = list(reader)

#stratify by class
kinaImages = []
centroImages = []
emptyImages = []

for image in imageNames:
    id = int(image.split(".")[0][2:])
    boxes = ast.literal_eval(csvList[id - 1]["boxes"])
    if boxes:
        firstLabel = boxes[0][0]
        if firstLabel == "Evechinus chloroticus":
            kinaImages.append(image)
        else:
            centroImages.append(image)
    else:
        emptyImages.append(image)



#partition the dataset
setsToPartition = [kinaImages, centroImages, emptyImages]

trainSize = 0.8
valSize = 0.1
testSize = 0.1

trainSet = []
valSet = []
testSet = []

if trainSize + valSize + testSize != 1: raise ValueError("Set proportions dont add to 1")

for data in setsToPartition:
    testIndexCutoff = int(len(data) * testSize) - 1
    valIndexCutoff = testIndexCutoff + int(len(data) * valSize) - 1

    indices = list(range(len(data)))
    random.shuffle(indices)

    for i, randIndex in enumerate(indices):
        if i <= testIndexCutoff:
            testSet.append(data[randIndex])
        elif i <= valIndexCutoff:
            valSet.append(data[randIndex])
        else:
            trainSet.append(data[randIndex])

print(f"train set size: {len(trainSet)} - {round(len(trainSet)/len(imageNames), 4)}")
print(f"val set size: {len(valSet)} - {round(len(valSet)/len(imageNames), 4)}")
print(f"test set size: {len(testSet)} - {round(len(testSet)/len(imageNames), 4)}")

#get stats on the partition
kinaTotal = 0
centroTotal = 0
emptyTotal = 0
for name, data in zip(["train", "val", "test"], [trainSet, valSet, testSet]):
    print(f"------- {name} -------")
    kinaCount = 0
    centroCount = 0
    emptyCount = 0
    for image in data:
        id = int(image.split(".")[0][2:])
        row = csvList[id]
        boxes = ast.literal_eval(row["boxes"])
        if len(boxes) == 0:
            emptyCount += 1
        else:
            classLabel = boxes[0][0]
            if classLabel == "Evechinus chloroticus": kinaCount += 1
            if classLabel == "Centrostephanus rodgersii": centroCount += 1

    print(f"Total classes: {kinaCount + centroCount + emptyCount}")
    print(f"Kina count: {kinaCount} - {round(kinaCount/(kinaCount + centroCount + emptyCount), 4)}")
    print(f"Centro count: {centroCount} - {round(centroCount/(kinaCount + centroCount + emptyCount), 4)}")
    print(f"Empty count: {emptyCount} - {round(emptyCount/(kinaCount + centroCount + emptyCount), 4)}")

    kinaTotal += kinaCount
    centroTotal += centroCount
    emptyTotal += emptyCount

print("--------------------------------")
print(f"Total classes: {kinaTotal + centroTotal + emptyTotal}")
print(f"Total Kina count: {kinaTotal}")
print(f"Total Centro count: {centroTotal}")
print(f"Total Empty count: {emptyTotal}")

csvFile.close()

#write sets to txt file
for name, data in zip(["train", "val", "test"], [trainSet, valSet, testSet]):
    f = open(f"data/{name}.txt", "w")
    f.write("\n".join(data))
    f.close()