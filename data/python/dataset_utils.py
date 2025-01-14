import csv
import ast

from matplotlib import pyplot as plt

URCHIN_SPECIES = ["Evechinus chloroticus", "Centrostephanus rodgersii", "Heliocidaris erythrogramma"]
URCHIN_SPECIES_SHORT = [name.split(" ")[0] for name in URCHIN_SPECIES]

def write_rows_to_csv(output_csv_name, rows):
    """Writes a list of dicts to a csv file
    Args:
        output_csv_name: name of the csv to write to
        rows: list of dicts to write to the csv
    """
    formated_csv_file = open(output_csv_name, "w", newline="")
    writer = csv.DictWriter(formated_csv_file, rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    formated_csv_file.close()


def get_stats(csv_path):
    """Get various counts and stats on the data set
    Args:
        csv_path: path to the csv file of the dataset
    """
    #read csv
    csv_file = open(csv_path, "r")
    reader = list(csv.DictReader(csv_file))

    print(f"Number of images: {len(reader)}")

    image_counts = [0] * (len(URCHIN_SPECIES) + 1)
    box_counts = [0] * (len(URCHIN_SPECIES))
    prob_less_than_1 = 0
    
    for row in reader:
        boxes = ast.literal_eval(row["boxes"])
        contains = [False] * len(URCHIN_SPECIES)
        #check the species contained in each box
        if boxes:
            for box in boxes:
                for i, c in enumerate(URCHIN_SPECIES):
                    if box[1] < 1: prob_less_than_1 += 1

                    #adjust image level species flags and total box counts
                    if box[0] == c:
                        box_counts[i] += 1
                        contains[i] = True
                        break
                    
        #increment empty image count
        if not any(contains):
            image_counts[-1] += 1

        #increment image level species counts
        for i, contain in enumerate(contains):
            if contain:
                image_counts[i] += 1

    #print stats
    for i, c in enumerate(URCHIN_SPECIES):
        print(f"Number of images that contain {c}: {image_counts[i]}")
    print(f"Number of images that contain no urchins: {image_counts[-1]}")
    print(f"Number of bounding boxes: {sum(box_counts)}")

    for i, c in enumerate(URCHIN_SPECIES):
        print(f"Number of {c} boxes: {box_counts[i]}")
    print(f"Number of boxes with liklihood less than 1: {prob_less_than_1}")

    csv_file.close()


def label_error_check(csv_path):
    """Checks for duplicate boxes and negative values in csv
    Args:
        csv_path: path to the csv to check
    """
    #read csv
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)
    rows = [row for row in reader]
    csv_file.close()

    #check for duplicate boxes and negative values
    for i, row in enumerate(rows):
        boxes = ast.literal_eval(row["boxes"])
        if len(boxes) != len(set(boxes)): print(f"Row {i}: duplicate found")
        for box in boxes:
            if box[2] < 0 or box[3] < 0 or box[4] < 0 or box[5] < 0: print(f"Row {i}: negative value found")
            
            
def urchin_count_plot(csv_path):
    """Plots a histogram of the number of urchins in each image
    Args:
        csv_path: csv_path: path to the csv file of the dataset
    """
    #read csv
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)
    rows = [row for row in reader]
    csv_file.close()

    #get number of urchins in each image
    count = []
    for row in rows:
        count.append(len(ast.literal_eval(row["boxes"])))
    
    #display histogram
    width = 1
    plt.hist(count, bins=list(range(0, max(count) + 1, width)), density=False, rwidth=0.9)
    plt.xlabel("Urchin count")
    plt.ylabel("Frequency")
    plt.grid(True, which="both")
    plt.title("Number of urchins in images")
    plt.show()