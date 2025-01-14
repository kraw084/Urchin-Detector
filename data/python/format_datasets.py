import csv
import ast

import cv2

from dataset_utils import URCHIN_SPECIES, URCHIN_SPECIES_SHORT, write_rows_to_csv




def format_csv(csv_file_path, source_name, formated_csv_name):
    """Formats a squidle annotation csv file to make it easier to work with.
    Args:
        csv_file_path: path to the csv file of the dataset
        source_name: name of the dataset source (to be added as a column)
        formated_csv_name: name of the new csv file
    """
    #read squidle annotation csv
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
        if label not in [""] + URCHIN_SPECIES: continue

        #skip annotations that are labeled as urchins but have no boxes
        if label and "point.polygon" in row and not row["point.polygon"]: continue

        #if this is the first time the image is encounted create a new entry
        if url not in image_data_dict: 
            id = row["point.media.id"]
            name = url.split("/")[-1]
            campaign_name = row["point.media.deployment.campaign.name"]
            deployment_name = row["point.media.deployment.name"]
            depth = row["point.pose.dep"]
            lat = row["point.pose.lat"]
            lon = row["point.pose.lon"]
            timestamp = row["point.pose.timestamp"]
            alt = "" if not "point.pose.alt" in row else row["point.pose.alt"]
 
            #create new image entry
            image_data = {"id":id, "url": url, "name":name, "width":0, "height":0, "source":source_name, "deployment": deployment_name, 
                        "campaign": campaign_name, "latitude": lat, "longitude": lon, 
                        "depth": depth, "altitude": alt, "time": timestamp, "flagged": False, 
                        "count":0}
            
            #add urchin speces flags
            for name in URCHIN_SPECIES_SHORT:
                image_data[name] = False
                
            image_data["boxes"] = []
            
            image_data_dict[url] = image_data
            i += 1

        #if the label is an urchin and the point has a bounding polygon
        if label and "point.polygon" in row and ast.literal_eval(row["point.polygon"]):
            confidence = float(row["likelihood"])
            x = float(row["point.x"])
            y = float(row["point.y"])
            points = ast.literal_eval(row["point.polygon"])
            
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

            #add box to image entry (if it's not a duplicate)
            if box not in (image_data_dict[url])["boxes"]: 
                (image_data_dict[url])["boxes"].append(box)
                (image_data_dict[url])["count"] += 1
                
                #set species flags
                for i in range(len(URCHIN_SPECIES)):
                    (image_data_dict[url])[URCHIN_SPECIES_SHORT[i]] = (image_data_dict[url])[URCHIN_SPECIES_SHORT[i]] or label == URCHIN_SPECIES[i]
        
            if row["needs_review"] == "True": (image_data_dict[url])["flagged"] = True

    new_rows = []
    for url in image_data_dict:
        new_rows.append(image_data_dict[url])

    csv_file.close()

    #saving the new dataset
    write_rows_to_csv(formated_csv_name, new_rows)
    print("-------- FINISHED --------")
    print(f"Entries created: {len(image_data_dict)}")
    print(f"Boxes found: {box_count}")
    print(f"Polygons found: {polygon_count}")
    print(f"Boxes is formatted dataset: {box_count + polygon_count}")
    print(f"Images with no urchins: {sum([1 for img in new_rows if not img['boxes']])}")


def concat_formated_csvs(csv_paths, concat_csv_name):
    """Concatenates multiple formated csvs into one
    Args:
        csv_paths: list of paths to the formated csvs
        concat_csv_name: path to the output csv
    """
    combined_rows = []
    ids_seen = []
    for path in csv_paths:
        #read csv
        csv_file = open(path, "r")
        reader = csv.DictReader(csv_file)
        rows = [r for r in reader]
        csv_file.close()
        
        #add each row to combined_rows
        for row in rows:
            id = row["id"]
            if id in ids_seen:
                print(f"Duplicate found in {path}: {id}")
                continue
            else:
                combined_rows.append(row)
                ids_seen.append(id)

    #needed to combine older csvs that didn't have altitude or Heliocidaris columns (this should no longer be needed)
    for row in combined_rows:
        if not "altitude" in row: row["altitude"] = ""
        if not "Heliocidaris" in row: row["Heliocidaris"] = False

    #save csv
    write_rows_to_csv(concat_csv_name, combined_rows)


def high_conf_csv(input_csv, output_csv_name, conf_cutoff = 0.7):
    """Filters a formated csv to include only high confidence boxes and non-flagged boxes
    Args:
        input_csv: path to the formated csv to filter
        output_csv_name: path to the output csv
        conf_cutoff: minimum confidence threshold for a box to be included
    """
    #read csv
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
            row["flagged"] = False
            row["count"] = updated_len
            
            #adjust species flags
            species_flags = [False for i in range(len(URCHIN_SPECIES))]
            for box in boxes:
                species_id = URCHIN_SPECIES.index(box[0])
                species_flags[species_id] = True
                
            for i, flag in enumerate(species_flags):
                row[URCHIN_SPECIES_SHORT[i]] = flag

    #save csv
    write_rows_to_csv(output_csv_name, rows)


def set_wh_col(input_csv, output_csv_name, im_dir):
    """Sets the width and height columns of a formated csv
    Args:
        input_csv: path to the formated csv
        output_csv_name: path to the output csv
        im_dir: path to the images directory
    """
    #read csv
    csv_file = open(input_csv, "r")
    reader = csv.DictReader(csv_file)
    rows = [r for r in reader]

    for row in rows:
        id = row["id"]
        #if the width or height is not set
        if row["width"] == "0" or row["height"] == "0":
            print(f"Adding width/height to im{id}.JPG")
            #read image and set the width and height columns
            im = cv2.imread(f"{im_dir}/im{id}.JPG")
            h, w, _ = im.shape
            row["width"] = w
            row["height"] = h
        else:
            print(f"Skipping im{id}.JPG")

    #save csv
    write_rows_to_csv(output_csv_name, rows)


def clip_boxes(input_csv, output_csv_name):
    """Clips the bounding boxes in a formated csv to be within the images bounds
    Args:
        input_csv: path to the formated csv
        output_csv_name: path to the output csv
    """
    #read csv
    csv_file = open(input_csv, "r")
    reader = csv.DictReader(csv_file)
    rows = [r for r in reader]

    for row in rows:
        h, w = int(row["height"]), int(row["width"])
        if w == 0 or h == 0:
            #without a width or height the box cannot be clipped so it is skipped
            print(f"WARNING: {row['id']} - height or width is 0")
            continue

        boxes = ast.literal_eval(row["boxes"])
        for i in range(len(boxes) - 1, -1, -1):
            box = boxes[i]

            #calculate box coordinates in terms of pixels
            xCenter = box[2] * w
            yCenter = box[3] * h
            boxWidth = box[4] * w
            boxHeight = box[5] * h

            #calculate corner coordinates of the box
            xMin, yMin = xCenter - boxWidth/2, yCenter - boxHeight/2
            xMax, yMax = xCenter + boxWidth/2, yCenter + boxHeight/2

            #clip coordinates
            if xMin < 0: xMin = 0
            if yMin < 0: yMin = 0
            if xMax > w: xMax = w
            if yMax > h: yMax = h

            #translate back to image relative coordinates
            xCenter = ((xMax + xMin)/2)/w
            yCenter = ((yMax + yMin)/2)/h
            boxWidth = (xMax - xMin)/w
            boxHeight = (yMax - yMin)/h

            boxes[i] = (box[0], box[1], xCenter, yCenter, boxWidth, boxHeight, box[6], box[7])

        row["boxes"] = boxes

    #save csv
    write_rows_to_csv(output_csv_name, rows)

	
if __name__ == "__main__":
    format_csv("data/csvs/NSW_annot.csv", "NSW DPI Urchins", "data/csvs/NSW_urchin_dataset_V5.csv")
    format_csv("data/csvs/UOA_annot.csv", "UoA Sea Urchin", "data/csvs/UOA_urchin_dataset_V5.csv")
    format_csv("data/csvs/UOA_empty.csv", "UoA Sea Urchin", "data/csvs/UOA_negative_dataset_V5.csv")
    format_csv("data/csvs/TAS_annot.csv", "Urchins - Eastern Tasmania", "data/csvs/Tasmania_urchin_dataset_V5.csv")
    format_csv("data/csvs/Helio_annot.csv", "RLS- Heliocidaris PPB", "data/csvs/Helio_urchin_dataset.csv")
    format_csv("data/csvs/Helio_empty_annot.csv", "RLS- Heliocidaris PPB", "data/csvs/Helio_negative_dataset.csv")
    
    concat_formated_csvs(["data/csvs/UOA_urchin_dataset_V5.csv", 
                          "data/csvs/UOA_negative_dataset_V5.csv", 
                          "data/csvs/Tasmania_urchin_dataset_V5.csv",
                          "data/csvs/NSW_urchin_dataset_V5.csv",
                          "data/csvs/Helio_urchin_dataset.csv",
                          "data/csvs/Helio_negative_dataset.csv"],
                          "data/csvs/Complete_urchin_dataset_V5.csv")


    
  
    high_conf_csv("data/csvs/Complete_urchin_dataset_V5.csv", "High_conf_dataset_V5.csv", 0.7)
    
    #download images first
    set_wh_col("data/csvs/Complete_urchin_dataset_V5.csv", "data/csvs/Complete_urchin_dataset_V5.csv", "data/images")
    clip_boxes("High_conf_dataset_V5.csv", "High_conf_clipped_dataset_V5.csv")
    
    
