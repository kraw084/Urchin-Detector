from urchin_utils import project_sys_path, DATASET_YAML_PATH, WEIGHTS_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    if True:
        yolov5.train.run(imgsz = 640, 
                        epochs = 100, 
                        data = DATASET_YAML_PATH, 
                        weights = "yolov5s.pt", 
                        save_period = 20,
                        batch_size = -1,
                        cache = "ram",
                        patience = 20
                        #hyp = "models/hyp.overfit.yaml"
                        )
    
    yolov5.val.run(DATASET_YAML_PATH, "models/yolov5s-reducedOverfitting/weights/last.pt")

