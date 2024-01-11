from urchin_utils import project_sys_path, DATASET_YAML_PATH, WEIGHTS_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    #yolov5.train.run(imgsz = 640, 
    #                epochs = 80, 
    #                data = DATASET_YAML_PATH, 
     #               weights = "yolov5s.pt", 
    #                save_period = 40,
    #                batch_size = -1,
     #               cache = "ram",
    #                )
    
   DATASET_YAML_PATH = "data/datasets/full_dataset_v2/datasetV2.yaml"

   yolov5.val.run(DATASET_YAML_PATH, "models/yolov5s-fullDatasetV3/weights/best.pt", task="val")
