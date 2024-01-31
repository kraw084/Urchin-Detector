import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from datetime import datetime
import cv2
import numpy as np

random.seed(42)

csv_file = open("data/csvs/Complete_urchin_dataset_V3.csv", "r")
reader = csv.DictReader(csv_file)
rows = [row for row in reader]
csv_file.close()

def long_lat_plot():
    var = "campaign"
    all_colours = [*mcolors.CSS4_COLORS]
    random.shuffle(all_colours)
    Values = list({x[var] for x in rows})
    col_dict = {Values[i]:all_colours[i] for i in range(len(Values))}

    for key in col_dict:
        print(f"{key}: {col_dict[key]}")


    long = [float(row["longitude"]) for row in rows]
    lat = [float(row["latitude"]) for row in rows]
    col = [col_dict[row[var]] for row in rows]


    plt.scatter(long, lat, c = col)
    plt.ylabel("lat")
    plt.xlabel("long")
    plt.show()


def group_by_time(seconds_threshold = 5, display = True, txt_path = None):
    time_strings = [(x["time"], x["id"]) for x in rows]
    datetimes = [] 
    for value, id in time_strings:
        try:
            datetimes.append((datetime.strptime(value, "%Y-%m-%d %H:%M:%S"), id))
        except:
            datetimes.append((datetime.strptime(value, "%d/%m/%Y %H:%M"), id))

    datetimes.sort()
    groups = [[]]
    for t, id in datetimes:
        for other_t in groups[-1]:
            if abs((t - other_t[0]).total_seconds()) <= seconds_threshold:
                groups[-1].append((t, id))
                break

        if (t, id) not in groups[-1]:
            groups.append([(t, id)])

    groups.pop(0)
    print(len(groups))

    if txt_path:
        f = open("data/datasets/full_dataset_v3/val.txt", "r")
        val_paths = [line.strip("\n") for line in f.readlines()]
        f.close()

        paths = [f"data/images/im{x[1]}.JPG" for g in groups for x in g]
        paths = [p for p in paths if p in val_paths]
        to_write = "\n".join(paths)
        f = open(f"{txt_path}", "w")
        f.write(to_write)
        f.close()

    if display:
        group_durations = [(g[-1][0] - g[0][0]).total_seconds() for g in groups if len(g) > 1]
        plt.hist(group_durations, 50)
        plt.show()

        for group in groups:
            for i, (t, id) in enumerate(group):
                im = cv2.imread(f"data/images/im{id}.JPG")
                h, w, _ = im.shape
                im = cv2.resize(im, (w//4, h//4))
                cv2.imshow(f"{i + 1}/{len(group)} {t.strftime('%Y-%m-%d %H:%M:%S')}", im)
                cv2.waitKey(0)
            
    else:
        return groups


group_by_time(display=False, txt_path="data/time.txt")