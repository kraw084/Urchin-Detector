from urchin_utils import project_sys_path, DATASET_YAML_PATH, WEIGHTS_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    if False:
        yolov5.train.run(imgsz = 640, 
                        epochs = 50, 
                        data = DATASET_YAML_PATH, 
                        weights = "yolov5s.pt", 
                        save_period = 20,
                        batch_size = -1,
                        cache = "ram",
                        patience = 50,
                        hyp = "models/hyp.overfit.yaml",
                        evolve = 40,
                        workers=4,
                        resume = True,
                        name = "exp2"
                        ) 

    yolov5.val.run(DATASET_YAML_PATH, "models/yolov5m-highres-ro-v4/weights/best.pt", task="test", imgsz=1280)