from urchin_utils import project_sys_path, DATASET_YAML_PATH, WEIGHTS_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    #yolov5.train.run(img = 1280, 
    #                epochs = 150, 
    #                data = DATASET_YAML_PATH, 
    #                weights = "yolov5m.pt", 
    #                save_period = 40,
    #                cache = "ram",
    #                batch_size = 32)
    
    yolov5.val.run(data = DATASET_YAML_PATH, weights = "models/yolov5m/weights/best.pt", task = "train")
    yolov5.val.run(data = DATASET_YAML_PATH, weights = "models/yolov5m/weights/best.pt", task = "val")
    yolov5.val.run(data = DATASET_YAML_PATH, weights = "models/yolov5m/weights/last.pt", task = "val")