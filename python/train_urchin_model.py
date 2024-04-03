from urchin_utils import project_sys_path, DATASET_YAML_PATH, WEIGHTS_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    if False:
        yolov5.train.run(imgsz = 640, 
                        epochs = 100, 
                        data = DATASET_YAML_PATH, 
                        weights = "yolov5s.pt", 
                        #save_period = 20,
                        batch_size = 50,
                        cache = "ram",
                        patience = 40,
                        hyp = "models/hyp.overfit.yaml",
                        evolve=300,
                        workers=4
                        )

    #yolov5.val.run(DATASET_YAML_PATH, "models/yolov5m-highres-ro/weights/best.pt", task="val", imgsz=1280, miniou=0.5)
    #yolov5.val.run(DATASET_YAML_PATH, "models/yolov5m-highres-ro/weights/best.pt", task="val", imgsz=1280, miniou=0.3)
