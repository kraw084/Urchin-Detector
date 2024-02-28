import csv
import ast
import cv2

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

        #skip annotations that are labeled as urchins but have no boxes
        if label and "point.data.polygon" in row and not row["point.data.polygon"]: continue

        if url not in image_data_dict: #if this is the first time the image is encounted create a new entry
            id = row["point.media.id"]
            name = url.split("/")[-1]
            campaign_name = row["point.media.deployment.campaign.name"]
            deployment_name = row["point.media.deployment.name"]
            depth = row["point.pose.dep"]
            lat = row["point.pose.lat"]
            lon = row["point.pose.lon"]
            timestamp = row["point.pose.timestamp"]
            alt = "" if not "point.pose.alt" in row else row["point.pose.alt"]
 
            image_data = {"id":id, "url": url, "name":name, "width":0, "height":0, "source":source_name, "deployment": deployment_name, 
                        "campaign": campaign_name, "latitude": lat, "longitude": lon, 
                        "depth": depth, "altitude": alt, "time": timestamp, "flagged": False, 
                        "count":0, "Evechinus":False, "Centrostephanus":False, "boxes":[]}
            
            image_data_dict[url] = image_data
            i += 1

        #if the label is an urchin and the point has a bounding polygon
        if label and "point.data.polygon" in row and row["point.data.polygon"]:
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
                box = (label, confidence, max(x, 0) , max(y, 0), box_width, box_height, row["needs_review"] == "True", row["id"])
                box_count += 1
            else: #polygon is not a box
                xValues = [p[0] for p in points]
                yValues = [p[1] for p in points]
                box_width = max(xValues) + abs(min(xValues))
                box_height = max(yValues) + abs(min(yValues))
                box = (label, confidence, max(x, 0), max(y, 0), box_width, box_height, row["needs_review"] == "True", row["id"])
                polygon_count += 1

            if box not in (image_data_dict[url])["boxes"]: 
                (image_data_dict[url])["boxes"].append(box)
                (image_data_dict[url])["count"] += 1
                (image_data_dict[url])["Evechinus"] = (image_data_dict[url])["Evechinus"] or label == "Evechinus chloroticus"
                (image_data_dict[url])["Centrostephanus"] = (image_data_dict[url])["Centrostephanus"] or label == "Centrostephanus rodgersii"

            if row["needs_review"] == "True": (image_data_dict[url])["flagged"] = True

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
    ids_seen = []
    for path in csv_paths:
        csv_file = open(path, "r")
        reader = csv.DictReader(csv_file)
        rows = [r for r in reader]
        csv_file.close()
        for row in rows:
            id = row["id"]
            if id in ids_seen:
                print(f"Duplicate found in {path}: {id}")
                continue
            else:
                combined_rows.append(row)
                ids_seen.append(id)

    for row in combined_rows:
        if "altitude" in row: continue
        row["altitude"] = ""

    write_rows_to_csv(concat_csv_name, combined_rows)


def label_error_check(csv_path):
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)
    rows = [row for row in reader]
    csv_file.close()

    for i, row in enumerate(rows):
        boxes = ast.literal_eval(row["boxes"])
        if len(boxes) != len(set(boxes)): print(f"Row {i}: duplicate found")
        for box in boxes:
            if box[2] < 0 or box[3] < 0 or box[4] < 0 or box[5] < 0: print(f"Row {i}: negative value found")


def high_conf_csv(input_csv, output_csv_name, conf_cutoff = 0.7):
        csv_file = open(input_csv, "r")
        reader = csv.DictReader(csv_file)
        rows = [r for r in reader]

        #remove boxes that have low confidence or are flagged for review
        for row in rows:
            boxes = ast.literal_eval(row["boxes"])
            original_len = len(boxes)
            for i in range(len(boxes) - 1, -1, -1):
                if boxes[i][1] < conf_cutoff or boxes[i][6]: boxes.pop(i)
            updated_len = len(boxes)

            #some boxes have been removed
            if updated_len != original_len:
                row["boxes"] = boxes
                row["flagged"] = "FALSE"
                row["count"] = updated_len
                
                contains_kina = False
                contains_centro = False
                for box in boxes:
                    if box[0] == "Evechinus chloroticus": contains_kina = True
                    if box[1] == "Centrostephanus rodgersii": contains_centro = True

                row["Evechinus"] = "TRUE" if contains_kina else "FALSE"
                row["Centrostephanus"] = "TRUE" if contains_centro else "FALSE"

        write_rows_to_csv(output_csv_name, rows)


def add_wh_col(input_csv, output_csv_name, im_dir):
    csv_file = open(input_csv, "r")
    reader = csv.DictReader(csv_file)
    rows = [r for r in reader]

    for row in rows:
        id = row["id"]
        im = cv2.imread(f"{im_dir}/im{id}.JPG")
        h, w, _ = im.shape
        row["width"] = w
        row["height"] = h

    write_rows_to_csv(output_csv_name, rows)


def clip_boxes(input_csv, output_csv_name):
    csv_file = open(input_csv, "r")
    reader = csv.DictReader(csv_file)
    rows = [r for r in reader]

    #remove boxes that have low confidence or are flagged for review
    for row in rows:
        h, w = int(row["height"]), int(row["width"])
        if w == 0 or h == 0:
            print(f"WARNING: {row['id']} - height or width is 0")
            continue

        boxes = ast.literal_eval(row["boxes"])
        for i in range(len(boxes) - 1, -1, -1):
            box = boxes[i]

            xCenter = box[2] * w
            yCenter = box[3] * h
            boxWidth = box[4] * w
            boxHeight = box[5] * h

            xMin, yMin = xCenter - boxWidth/2, yCenter - boxHeight/2
            xMax, yMax = xCenter + boxWidth/2, yCenter + boxHeight/2

            if xMin < 0: xMin = 0
            if yMin < 0: yMin = 0
            if xMax > w: xMax = w
            if yMax > h: yMax = h

            xCenter = ((xMax + xMin)/2)/w
            yCenter = ((yMax + yMin)/2)/h
            boxWidth = (xMax - xMin)/w
            boxHeight = (yMax - yMin)/h

            boxes[i] = (box[0], box[1], xCenter, yCenter, boxWidth, boxHeight, box[6], box[7])

        row["boxes"] = boxes

    write_rows_to_csv(output_csv_name, rows)


def ims_removed(original_csv, new_csv):
    csv_file = open(original_csv, "r")
    reader = csv.DictReader(csv_file)
    og_rows = [r for r in reader]
    old_ids = [int(row["id"]) for row in og_rows]

    csv_file = open(new_csv, "r")
    reader = csv.DictReader(csv_file)
    new_rows = [r for r in reader]
    new_ids = [int(row["id"]) for row in new_rows]

    removed_ids = []
    for id in old_ids:
        if id not in new_ids:
            print(f"id {id} not found in {new_csv}")
            removed_ids.append(id)

    return removed_ids


def transfer_wh(original_csv, new_csv, output_csv_name):
    csv_file = open(original_csv, "r")
    reader = csv.DictReader(csv_file)
    original_rows = {int(row["id"]):row for row in reader}
    csv_file.close()

    csv_file = open(new_csv, "r")
    reader = csv.DictReader(csv_file)
    new_rows = list(reader)
    csv_file.close()

    for row in new_rows:
        id = int(row["id"])
        if id in original_rows:
            w = original_rows[id]["width"]
            h = original_rows[id]["height"]
            row["width"] = w
            row["height"] = h

    write_rows_to_csv(output_csv_name, new_rows) 

if __name__ == "__main__":
    format_csv("NSW_annot_V4.csv", "NSW DPI Urchins", "NSW_urchin_dataset_V4.csv")
    format_csv("UOA_annot_V4.csv", "UoA Sea Urchin", "UOA_urchin_dataset_V4.csv")
    format_csv("UOA_empty_V4.csv", "UoA Sea Urchin", "UOA_negative_dataset_V4.csv")
    format_csv("TAS_annot_V4.csv", "Urchins - Eastern Tasmania", "Tasmania_urchin_dataset_V4.csv")
    
    concat_formated_csvs(["UOA_urchin_dataset_V4.csv", 
                          "UOA_negative_dataset_V4.csv", 
                          "Tasmania_urchin_dataset_V4.csv",
                          "NSW_urchin_dataset_V4.csv"],
                          "Complete_urchin_dataset_V4.csv")

    #download images first
    #add_wh_col("data/csvs/Complete_urchin_dataset_V3.csv", "Complete_urchin_dataset_V3_updated.csv", "data/images_v3")

    #high_conf_csv("data/csvs/Complete_urchin_dataset_V3_FIX.csv", "High_conf_dataset_V3_FIX.csv", 0.7)

    #clip_boxes("High_conf_dataset_V3_FIX.csv", "High_conf_clipped_dataset_V3_FIX.csv")
    pass