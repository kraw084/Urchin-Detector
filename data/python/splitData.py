import os
import random
import csv
import ast

"""Seperate the dataset into train, validation and test sets"""

imageNames = os.listdir("data/images")

trainSize = 0.8
valSize = 0.1
testSize = 0.1

trainSet = []
valSet = []
testSet = []

if trainSize + valSize + testSize != 1: raise ValueError("Set proportions dont add to 1")


#partition the dataset
testIndexCutoff = int(len(imageNames) * testSize) - 1
valIndexCutoff = testIndexCutoff + int(len(imageNames) * valSize) - 1

indices = list(range(len(imageNames)))
random.shuffle(indices)

for i, randIndex in enumerate(indices):
    if i <= testIndexCutoff:
        testSet.append(imageNames[randIndex])
    elif i <= valIndexCutoff:
        valSet.append(imageNames[randIndex])
    else:
        trainSet.append(imageNames[randIndex])

print(f"train set size: {len(trainSet)} - {round(len(trainSet)/len(imageNames), 4)}")
print(f"val set size: {len(valSet)} - {round(len(valSet)/len(imageNames), 4)}")
print(f"test set size: {len(testSet)} - {round(len(testSet)/len(imageNames), 4)}")

#get stats on the partition
csvFile = open("data/csvs/Complete_urchin_dataset.csv")
reader = csv.DictReader(csvFile)
csvList = [row for row in reader][1:]

totalCounts = [0] * 3
for name, data in zip(["train", "val", "test"], [trainSet, valSet, testSet]):
    counts = [0] * 3
    for image in data:
        id = int(image.split(".")[0][2:])
        boxes = ast.literal_eval(csvList[id - 1]["boxes"])
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
        else:
            counts[2] += 1

    print(f"------- {name} set -------")
    print(f"Total classes: {sum(counts)}")
    print(f"Kina count: {counts[0]} - {round(counts[0]/sum(counts), 4)}")
    print(f"Centro count: {counts[1]} - {round(counts[1]/sum(counts), 4)}")
    print(f"Empty count: {counts[2]} - {round(counts[2]/sum(counts), 4)}")

    totalCounts[0] += counts[0]
    totalCounts[1] += counts[1]
    totalCounts[2] += counts[2]

csvFile.close()


#write sets to txt file
for name, data in zip(["train", "val", "test"], [trainSet, valSet, testSet]):
    f = open(f"data/{name}.txt", "w")
    f.write("\n".join(data))
    f.close()