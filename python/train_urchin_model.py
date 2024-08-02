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
                        patience = 50,
                        hyp = "models/hyp.overfit.yaml",
                        name = "clahe_full"
                        ) 

    model_path = r"models\yolov5m-highRes-ro-V4\weights\best.pt"
    #model_path = r"yolov5/runs/train/clahe_full/weights/best.pt"

    yolov5.val.run(DATASET_YAML_PATH, 
                    model_path, 
                    task="val", 
                    imgsz=1280)

