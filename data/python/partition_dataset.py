import os
import random
import csv
import ast

def partition(image_dir, dest_dir, csv_path, train_size = 0.8, val_size = 0.1, test_size = 0.1):
    """Seperate the dataset into train, validation and test sets, stratified by class"""
    image_names = os.listdir(image_dir)

    csv_file = open(csv_path)
    reader = csv.DictReader(csv_file)
    csv_list = list(reader)

    #stratify by class
    kina_images = []
    centro_images = []
    empty_images = []

    for image in image_names:
        id = int(image.split(".")[0][2:])
        boxes = ast.literal_eval(csv_list[id]["boxes"])
        if boxes:
            first_label = boxes[0][0]
            if first_label == "Evechinus chloroticus":
                kina_images.append(image)
            else:
                centro_images.append(image)
        else:
            empty_images.append(image)

    #make negative images ~10% of final image count
    #random.shuffle(empty_images)
    #empty_images = empty_images[:380]

    #partition the dataset
    sets_to_Partition = [kina_images, centro_images, empty_images]
    for l in sets_to_Partition:
        random.shuffle(l)

    train_set = []
    val_set = []
    test_set = []

    if train_size + val_size + test_size != 1: raise ValueError("Set proportions dont add to 1")

    for data in sets_to_Partition:
        test_index_cutoff = int(len(data) * test_size) - 1
        val_index_cutoff = test_index_cutoff + int(len(data) * val_size) - 1

        indices = list(range(len(data)))
        random.shuffle(indices)

        for i, randIndex in enumerate(indices):
            if i <= test_index_cutoff:
                test_set.append(data[randIndex])
            elif i <= val_index_cutoff:
                val_set.append(data[randIndex])
            else:
                train_set.append(data[randIndex])

    print(f"train set size: {len(train_set)} - {round(len(train_set)/len(image_names), 4)}")
    print(f"val set size: {len(val_set)} - {round(len(val_set)/len(image_names), 4)}")
    print(f"test set size: {len(test_set)} - {round(len(test_set)/len(image_names), 4)}")

    #get stats on the partition
    kina_total = 0
    centro_total = 0
    empty_total = 0
    for name, data in zip(["train", "val", "test"], [train_set, val_set, test_set]):
        print(f"------- {name} -------")
        kina_count = 0
        centro_count = 0
        empty_count = 0
        for image in data:
            id = int(image.split(".")[0][2:])
            row = csv_list[id]
            boxes = ast.literal_eval(row["boxes"])
            if len(boxes) == 0:
                empty_count += 1
            else:
                contains_kina = False
                contains_centro = False
                for box in boxes:
                    class_label = box[0]
                    contains_kina = contains_kina or class_label == "Evechinus chloroticus"
                    contains_centro = contains_centro or class_label == "Centrostephanus rodgersii"

                kina_count += int(contains_kina)
                centro_count += int(contains_centro)


        print(f"Total classes: {kina_count + centro_count + empty_count}")
        print(f"Kina count: {kina_count} - {round(kina_count/(kina_count + centro_count + empty_count), 4)}")
        print(f"Centro count: {centro_count} - {round(centro_count/(kina_count + centro_count + empty_count), 4)}")
        print(f"Empty count: {empty_count} - {round(empty_count/(kina_count + centro_count + empty_count), 4)}")

        kina_total += kina_count
        centro_total += centro_count
        empty_total += empty_count

    print("--------------------------------")
    print(f"Total classes: {kina_total + centro_total + empty_total}")
    print(f"Total Kina count: {kina_total}")
    print(f"Total Centro count: {centro_total}")
    print(f"Total Empty count: {empty_total}")

    csv_file.close()

    #write sets to txt file
    for name, data in zip(["train", "val", "test"], [train_set, val_set, test_set]):
        data = [os.path.join("data/images/", image) for image in data]

        f = open(f"{dest_dir}/{name}.txt", "w")
        f.write("\n".join(data))
        f.close()


def filter_txt(txt, var, target_value):
    csv_file = open("data/csvs/Complete_urchin_dataset_V3.csv", "r")
    reader = csv.DictReader(csv_file)
    rows = [row for row in reader]
    csv_file.close()

    f = open(txt, "r")
    image_paths = [line.strip("\n") for line in f.readlines()]
    f.close()

    filtered_paths = [x for x in image_paths if rows[int(x.split("/")[-1].split(".")[0][2:])][var] == target_value]

    f = open(txt, "w")
    f.write("\n".join(filtered_paths))
    f.close()

if __name__ == "__main__":
    #partition("data/images", "data", "data/csvs/Complete_urchin_dataset_v3.csv")

    filter_txt("data/datasets/NSW_dataset/test.txt", "source", "NSW DPI Urchins")