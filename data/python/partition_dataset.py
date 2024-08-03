import os
import random
import csv

def id_from_im_name(im_name):
    if "\\" in im_name: im_name = im_name.split("\\")[-1].strip("\n")
    if "/" in im_name: im_name = im_name.split("/")[-1].strip("\n")
    return int(im_name.split(".")[0][2:])

def partition(csv_path, train_size = 0.8, val_size = 0.1, test_size = 0.1, ids=None, counter=None, classes=["Evechinus", "Centrostephanus"]):
    """Seperate the dataset into train, validation and test sets, stratified by class"""

    csv_file = open(csv_path)
    reader = csv.DictReader(csv_file)
    csv_list = list(reader)

    #stratify by class
    images_by_class = [[] for i in range(len(classes + 1))]

    for row in csv_list:
        if ids and not int(row["id"]) in ids:
            continue

        not_empty = False
        for i, c in enumerate(classes):
            if row[c].upper() == "TRUE":
                images_by_class[i].append(f"im{row['id']}.JPG")
                not_empty = True
                break
        
        if not not_empty:
            images_by_class[-1].append(f"im{row['id']}.JPG")

    for l in images_by_class:
        random.shuffle(l)

    train_set = []
    val_set = []
    test_set = []

    if train_size + val_size + test_size != 1: raise ValueError("Set proportions dont add to 1")

    for data in images_by_class:
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

    print(f"train set size: {len(train_set)} - {round(len(train_set)/len(csv_list), 4)}")
    print(f"val set size: {len(val_set)} - {round(len(val_set)/len(csv_list), 4)}")
    print(f"test set size: {len(test_set)} - {round(len(test_set)/len(csv_list), 4)}")

    #get stats on the partition
    total_class_counts = [0] * (len(classes) + 1)
    id_to_im_data = {int(row["id"]):row for row in csv_list}
    for name, data in zip(["train", "val", "test"], [train_set, val_set, test_set]):
        if len(data) == 0: continue
        print(f"------- {name} -------")
        set_class_counts = [0] * (len(classes) + 1)
        for im_name in data:
            id = int(im_name.split(".")[0][2:])
            row = id_to_im_data[id]
           
            not_empty = False
            for i, c in enumerate(classes):
                if row[c].upper() == "TRUE":
                    set_class_counts[i] += 1
                    not_empty = True
                    break
        
            if not not_empty:
                set_class_counts[-1] += 1

        print(f"Total classes: {sum(set_class_counts)}")
        for i, c in enumerate(classes):
            print(f"{c} count: {set_class_counts[i]} - {round(set_class_counts[i]/sum(set_class_counts) , 3)}")

    
    csv_file.close()

    #write sets to txt file
    for name, data in zip(["train", "val", "test"], [train_set, val_set, test_set]):
        data = [os.path.join("data/images/", image) for image in data]

        f = open(f"{name}{f'_{counter}' if counter else ''}.txt", "w")
        f.write("\n".join(data))
        f.close()

def combine_txts(txt1, txt2):
    with open(txt1, 'a') as file1:
        with open(txt2, 'r') as file2:
            file2_contents = file2.read()
            file1.write(file2_contents)

def create_cv_folds(k = 5):
    all_images = os.listdir("data/images")
    random.shuffle(all_images)
    folds = []

    csv_file = open("data/csvs/High_conf_clipped_dataset_V4.csv")
    reader = csv.DictReader(csv_file)
    csv_list = list(reader)
    ids = [int(row["id"]) for row in csv_list]

    all_images = [im for im in all_images if int((im.split("/")[-1].strip("\n")).split(".")[0][2:]) in ids]

    cutoff = (len(all_images) // k) + 1
    for i in range(k):
        folds.append(all_images[i * cutoff: (i+1) * cutoff])

    sum = 0
    for f in folds:
        print(len(f))
        sum += len(f)

    print("-----------")
    print(f"sum of fold lengths: {sum}")
    print(f"total images: {len(all_images)}")
    print("-----------")

    for i, fold in enumerate(folds):
        f = open(f"val{i}.txt", "w")
        data = [os.path.join("data/images/", image) for image in fold]
        f.write("\n".join(data))
        f.close()

        f = open(f"train{i}.txt", "w")
        data = []
        for j in range(k):
            if j != i:
                 data += [os.path.join("data/images/", image) for image in folds[j]]
        f.write("\n".join(data))
        f.close()

        
    

if __name__ == "__main__":
    #partition("data/csvs/Complete_urchin_dataset_V4.csv")
    create_cv_folds(5)