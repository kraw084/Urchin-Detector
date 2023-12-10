import csv
import ast

"""Formats the SQ+ csv files"""

csvFilePath = "annotations3.csv"
csvFile = open(csvFilePath, "r")
reader = csv.DictReader(csvFile)

#used to store the rows for the new csv
imageDataDict = {}

boxCount = 0
polygonCount = 0

for row in reader:
    url = row["point.media.path_best"]
    label = row["label.name"]

    #ignore annotations that are not urchins or empty images
    if label not in ["", "Evechinus chloroticus", "Centrostephanus rodgersii"]: continue

    if url not in imageDataDict: #if this is the first time the image is encounted create a new entry
        name = url.split("/")[-1]
        campaignName = row["point.media.deployment.campaign.name"]
        deploymentName = row["point.media.deployment.name"]
        depth = row["point.pose.dep"]
        lat = row["point.pose.lat"]
        lon = row["point.pose.lon"]
        timestamp = row["point.pose.timestamp"]

        imageData = {"url": url, "name":name, "deployment": deploymentName, 
                     "campaign": campaignName, "latitude": lat, "longitude": lon, 
                     "depth": depth, "time": timestamp, "boxes":[]}
        
        imageDataDict[url] = imageData
    
    #if the label is an urchin and the point has a bounding polygon
    if label and row["point.data.polygon"]:
        confidence = float(row["likelihood"])
        x = float(row["point.x"])
        y = float(row["point.y"])

        #squidle stores polygon points relative to the center point and the image dimensions e.g. actual point is (x, y) + (w * polygonPointX, h * polygonPointY)
        points = ast.literal_eval(row["point.data.polygon"])

        if len(points) == 4: #bounding box
            #this gets the width and height of the box relative to the image dimensions 
            #point order is BL, TL, TR, BR
            boxWidth = points[3][0] * 2
            boxHeight = points[0][1] * 2
            box = (label, confidence, x, y, boxWidth, boxHeight)
            boxCount += 1
        else:
            xValues = [p[0] for p in points]
            yValues = [p[1] for p in points]
            boxWidth = max(xValues) + abs(min(xValues))
            boxHeight = max(yValues) + abs(min(yValues))
            box = (label, confidence, x, y, boxWidth, boxHeight)
            polygonCount += 1

        (imageDataDict[url])["boxes"].append(box)

newRows = []
for url in imageDataDict:
    newRows.append(imageDataDict[url])

csvFile.close()

#saving the new dataset
csvOutputName = "NSW_urchin_dataset.csv"
csvFileOut = open(csvOutputName, "w", newline="")
writer = csv.DictWriter(csvFileOut, newRows[0].keys())
writer.writeheader()
writer.writerows(newRows)
csvFileOut.close()

print("-------- FINISHED --------")
print(f"Entries created: {len(imageDataDict)}")
print(f"Boxes found: {boxCount}")
print(f"Polygons found: {polygonCount}")
print(f"Boxes is formatted dataset: {boxCount + polygonCount}")
print(f"Images with no urchins: {sum([1 for img in newRows if not img['boxes']])}")