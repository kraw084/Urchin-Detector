import sys
import os
import csv
import urchin_utils

urchin_utils.project_sys_path()
import yolov5.val

def get_metrics_by_var(weights_path, csv_path, csv_var, dataset_yaml, task):
    csv_file = open(csv_path, "r")
    csv_reader = csv.DictReader(csv_file)

    #get ids of images in the task partion of the chosen dataset
    set_txt = open(f"data/{task}.txt", "r")
    set_txt_lines = set_txt.readlines()
    images_in_set = [name.split("\\")[-1].strip("\n") for name in set_txt_lines]
    ids_in_set = sorted([int(name.split(".")[0][2:]) for name in images_in_set])
    set_txt.close()
       
    #seperate images in the task set by csv_var
    id_by_var = {}
    for row in csv_reader:
        id = int(row["id"])
        if not (id in ids_in_set): continue

        var_value = row[csv_var]
        if var_value in id_by_var:
            id_by_var[var_value].append(id)
        else:
            id_by_var[var_value] = [id]

    csv_file.close()

    for var_value in id_by_var:
        #replace task.txt with the filtered image list
        set_txt = open(f"data/{task}.txt", "w")
        ids_with_value = id_by_var[var_value]
        abs_image_paths = [str(os.path.abspath(f"data/images/im{id}.JPG")) for id in ids_with_value]
        set_txt.write("\n".join(abs_image_paths))

        print(f"---------------- {csv_var}: {var_value} ----------------")
        metrics = yolov5.val.run(data = dataset_yaml, weights = weights_path, plots = False, task = task)
        print(metrics)

    #replace original contents of task.txt
    set_txt.close()
    set_txt = open(f"data/{task}.txt", "w")
    set_txt.writelines(set_txt_lines)
    set_txt.close()
    print("-------------- FINISHED --------------")


if __name__ == "__main__":
    model_name = "yolov5s-fullDataset"
    weights = os.path.abspath(f"models/{model_name}/weights/best.pt")
    #get_metrics_by_var(weights, "data/csvs/Complete_urchin_dataset.csv", "campaign", "data/dataset.yaml", "train")