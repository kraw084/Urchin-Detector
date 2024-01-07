from urchin_utils import project_sys_path, DATASET_YAML_PATH, WEIGHTS_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    yolov5.train.run(imgsz = 640, 
                    epochs = 100, 
                    data = WEIGHTS_PATH, 
                    weights = "yolov5s.pt", 
                    save_period = 40,
                    cache = "ram",
                    batch_size = -1)
    


