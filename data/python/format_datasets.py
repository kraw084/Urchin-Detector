import csv
import ast

"""Formats the SQ+ csv files"""

def write_rows_to_csv(output_csv_name, rows):
    formated_csv_file = open(output_csv_name, "w", newline="")
    writer = csv.DictWriter(formated_csv_file, rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    formated_csv_file.close()

def format_csv(csv_file_path, source_name, formated_csv_name):
    csv_file = open(csv_file_path, "r")
    reader = csv.DictReader(csv_file)

    #used to store the rows for the new csv
    image_data_dict = {}

    box_count = 0
    polygon_count = 0

    i = 0
    for row in reader:
        url = row["point.media.path_best"]
        label = row["label.name"]

        #ignore annotations that are not urchins or empty images
        if label not in ["", "Evechinus chloroticus", "Centrostephanus rodgersii"]: continue

        if url not in image_data_dict: #if this is the first time the image is encounted create a new entry
            name = url.split("/")[-1]
            campaign_name = row["point.media.deployment.campaign.name"]
            deployment_name = row["point.media.deployment.name"]
            depth = row["point.pose.dep"]
            lat = row["point.pose.lat"]
            lon = row["point.pose.lon"]
            timestamp = row["point.pose.timestamp"]
 
            image_data = {"id":i, "url": url, "name":name, "source":source_name, "deployment": deployment_name, 
                        "campaign": campaign_name, "latitude": lat, "longitude": lon, 
                        "depth": depth, "time": timestamp, "boxes":[]}
            
            image_data_dict[url] = image_data
            i += 1
        
        #if the label is an urchin and the point has a bounding polygon
        if label and row["point.data.polygon"]:
            confidence = float(row["likelihood"])
            x = float(row["point.x"])
            y = float(row["point.y"])

            #squidle stores polygon points relative to the center point and the image dimensions e.g. actual point is (x, y) + (w * polygonPointX, h * polygonPointY)
            points = ast.literal_eval(row["point.data.polygon"])

            if len(points) == 4: #polygon is box
                #squidle stores bounding boxes as point offsets, relative to the center point
                #point order is BL, TL, TR, BR
                box_width = points[3][0] * 2
                box_height = points[0][1] * 2
                box = (label, confidence, x, y, box_width, box_height)
                box_count += 1
            else: #polygon is not a box
                xValues = [p[0] for p in points]
                yValues = [p[1] for p in points]
                box_width = max(xValues) + abs(min(xValues))
                box_height = max(yValues) + abs(min(yValues))
                box = (label, confidence, x, y, box_width, box_height)
                polygon_count += 1

            (image_data_dict[url])["boxes"].append(box)

    newRows = []
    for url in image_data_dict:
        newRows.append(image_data_dict[url])

    csv_file.close()

    #saving the new dataset
    write_rows_to_csv(formated_csv_name, newRows)
    print("-------- FINISHED --------")
    print(f"Entries created: {len(image_data_dict)}")
    print(f"Boxes found: {box_count}")
    print(f"Polygons found: {polygon_count}")
    print(f"Boxes is formatted dataset: {box_count + polygon_count}")
    print(f"Images with no urchins: {sum([1 for img in newRows if not img['boxes']])}")

    return newRows

def concat_formated_csvs(csv_paths, concat_csv_name):
    combined_rows = []
    for path in csv_paths:
        csv_file = open(path, "r")
        reader = csv.DictReader(csv_file)
        rows = [r for r in reader]
        csv_file.close()
        combined_rows += rows

    #reset ids
    for i, row in enumerate(combined_rows):
        row["id"] = i

    write_rows_to_csv(concat_csv_name, combined_rows)

def label_correction(csv_path):
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)
    rows = [row for row in reader]
    csv_file.close()

    for i, row in enumerate(rows):
        boxes = ast.literal_eval(row["boxes"])
        if len(boxes) != len(set(boxes)): print(f"Row {i}: duplicate found")
        for box in boxes:
            if box[2] < 0 or box[3] < 0 or box[4] < 0 or box[5] < 0: print(f"Row {i}: negative value found")


if __name__ == "__main__":
    label_correction("data/csvs/Complete_urchin_dataset_V2.csv")