import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import ast

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

def urchin_count_plot():
    count = []
    for row in rows:
        count.append(len(ast.literal_eval(row["boxes"])))
    print(sum(count))
    width = 1
    plt.hist(count, bins=list(range(0, max(count) + 1, width)), density=False, rwidth=0.9)
    plt.xlabel("Urchin count")
    plt.ylabel("Frequency")
    plt.grid(True, which="both")
    plt.title("Number of urchins in images")
    plt.show()

urchin_count_plot()