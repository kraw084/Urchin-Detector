from urchin_utils import project_sys_path, DATASET_YAML_PATH, WEIGHTS_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    if False:
        yolov5.train.run(imgsz = 1280, 
                        epochs = 200, 
                        data = DATASET_YAML_PATH, 
                        weights = "yolov5m.pt", 
                        save_period = 10,
                        batch_size = -1,
                        #cache = "ram",
                        patience = 100,
                        hyp = "models/hyp.overfit.yaml",
                        resume = "yolov5/runs/train/exp16/weights/epoch110.pt"
                        )
        x
    yolov5.val.run(DATASET_YAML_PATH, "yolov5/runs/train/exp16/weights/epoch190.pt", task="train", imgsz=1280)
