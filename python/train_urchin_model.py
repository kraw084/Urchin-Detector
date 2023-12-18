from urchin_utils import project_sys_path, DATASET_YAML_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    if False:
        yolov5.train.run(img = 640, 
                        epochs = 150, 
                        data = DATASET_YAML_PATH, 
                        weights = "yolov5s.pt", 
                        save_period = 40,
                        cache = "ram",
                        batch_size = 32)
    
    yolov5.val.run(data = DATASET_YAML_PATH, weights = "yolov5/runs/train/exp2/weights/best.pt", task = "val")