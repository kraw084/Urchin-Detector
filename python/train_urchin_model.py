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
                        save_period = 5,
                        batch_size = 6,
                        #cache = "ram",
                        patience = 50,
                        hyp = "models/hyp.overfit.yaml",
                        resume = "yolov5/runs/train/exp16/weights/epoch170.pt"
                        )

    yolov5.val.run(DATASET_YAML_PATH, "models/yolov5m-highres-ro-V4/weights/best.pt", task="val", imgsz=1280)
