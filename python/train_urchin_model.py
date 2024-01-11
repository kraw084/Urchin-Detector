from urchin_utils import project_sys_path, DATASET_YAML_PATH, WEIGHTS_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    yolov5.train.run(imgsz = 1280, 
                    epochs = 150, 
                    data = DATASET_YAML_PATH, 
                    weights = "yolov5m.pt", 
                    save_period = 40,
                    batch_size = -1,
                    hyp = "models/hyp.custom.yaml",
                    resume = "yolov5/runs/train/exp4/weights/epoch80.pt"
                    )
    


