import os
import random
import csv

from dataset_utils import URCHIN_SPECIES_SHORT

def partition(csv_path, train_size = 0.8, val_size = 0.1, test_size = 0.1, classes=URCHIN_SPECIES_SHORT):
    """Seperate the dataset into train, validation and test sets, stratified by class
    Args:
        csv_path: path to the csv file of the dataset
        train_size: the proportion of the dataset that will be in the train set
        val_size: the proportion of the dataset that will be in the validation set
        test_size: the proportion of the dataset that will be in the test set
        classes: a list of the classes to be included in the dataset
    """

    if train_size + val_size + test_size != 1: raise ValueError("Set proportions dont add to 1")

    #read csv
    csv_file = open(csv_path)
    reader = csv.DictReader(csv_file)
    csv_list = list(reader)

    #Seperate images by class (for simplicity, class is set to be species of first urchin in an image or empty)
    images_by_class = [[] for i in range(len(classes) + 1)]
    for row in csv_list:
        not_empty = False
        for i, c in enumerate(classes):
            if row[c].upper() == "TRUE":
                images_by_class[i].append(f"im{row['id']}.JPG")
                not_empty = True
                break
        
        if not not_empty:
            images_by_class[-1].append(f"im{row['id']}.JPG")

    train_set = []
    val_set = []
    test_set = []

    #partition each set of images into the train, val and test sets
    for data in images_by_class:
        #calculate index cutoffs to determine which images will be placed in each set
        test_index_cutoff = int(len(data) * test_size) - 1
        val_index_cutoff = test_index_cutoff + int(len(data) * val_size) - 1

        #assign each image a random index
        indices = list(range(len(data)))
        random.shuffle(indices)

        #sort images into splits based on the random index and cutoffs
        for i, rand_index in enumerate(indices):
            if i <= test_index_cutoff:
                test_set.append(data[rand_index])
            elif i <= val_index_cutoff:
                val_set.append(data[rand_index])
            else:
                train_set.append(data[rand_index])

    #print set sizes
    print(f"train set size: {len(train_set)} - {round(len(train_set)/len(csv_list), 4)}")
    print(f"val set size: {len(val_set)} - {round(len(val_set)/len(csv_list), 4)}")
    print(f"test set size: {len(test_set)} - {round(len(test_set)/len(csv_list), 4)}")

    #get stats on the partition to esnure they are stratified properly    
    #create dict for looking up image data based on id
    id_to_im_data = {int(row["id"]):row for row in csv_list}
    for name, data in zip(["train", "val", "test"], [train_set, val_set, test_set]):
        if len(data) == 0: continue
        print(f"------- {name} -------")
        
        #get counts for each class
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

        #print out class counts and proportions
        print(f"Total classes: {sum(set_class_counts)}")
        for i, c in enumerate(classes):
            print(f"{c} count: {set_class_counts[i]} - {round(set_class_counts[i]/sum(set_class_counts) , 3)}")

    csv_file.close()

    #write sets to txt file
    for name, data in zip(["train", "val", "test"], [train_set, val_set, test_set]):
        data = [os.path.join("data/images/", image) for image in data]

        f = open(f"{name}.txt", "w")
        f.write("\n".join(data))
        f.close()


def combine_txts(txt1, txt2):
    """Used to combine txt files into a single txt file
    Args:
        txt1: path to the first txt file
        txt2: path to the second txt file
    """
    with open(txt1, 'a') as file1:
        with open(txt2, 'r') as file2:
            file2_contents = file2.read()
            file1.write("\n" + file2_contents)


def create_cv_folds(csv_path, k = 5):
    """Creates k folds of the dataset for cross validation
    Args:
        csv_path: path to the csv file of the dataset
        k: number of folds to create
    """
    #create a list of all image names in the dataset
    all_images = os.listdir("data/images")
    random.shuffle(all_images)
    folds = []

    #read csv and create a list of all ids
    csv_file = open(csv_path)
    reader = csv.DictReader(csv_file)
    csv_list = list(reader)
    ids = [int(row["id"]) for row in csv_list]

    #filter the list of all images to only include images that are in the csv
    all_images = [im for im in all_images if int((im.split("/")[-1].strip("\n")).split(".")[0][2:]) in ids]

    #seperate the images into k folds
    cutoff = (len(all_images) // k) + 1
    for i in range(k):
        folds.append(all_images[i * cutoff: (i+1) * cutoff])

    #write each fold to a txt file
    for i, fold in enumerate(folds):
        #create val set from one fold
        f = open(f"val_{i}.txt", "w")
        data = [os.path.join("data/images/", image) for image in fold]
        f.write("\n".join(data))
        f.close()

        #create train set from all other folds
        f = open(f"train_{i}.txt", "w")
        data = []
        for j in range(k):
            if j != i:
                 data += [os.path.join("data/images/", image) for image in folds[j]]
        f.write("\n".join(data))
        f.close()

        
if __name__ == "__main__":
    partition("data/csvs/High_conf_clipped_dataset_V5.csv")
