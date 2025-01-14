from python.model_utils import project_sys_path, DATASET_YAML_PATH, WEIGHTS_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    if False:
        yolov5.train.run(imgsz = 1280, 
                        epochs = 200, 
                        data = "data/datasets/full_dataset_v5/datasetV5.yaml", 
                        weights = "yolov5m.pt", 
                        save_period = 10,
                        batch_size = -1,
                        #cache = "ram",
                        patience = 50,
                        hyp = "models/hyp.overfit.yaml",
                        name = "helio_v1"
                        ) 

    #yolov5.val.run(DATASET_YAML_PATH, 
    #               r"models\yolov5m-highRes-ro-V4\weights\best.pt", 
    #               task="test", 
    #               imgsz=1280,
    #               conf_thres=0.45)
    
    yolov5.val.run("data/datasets/full_dataset_v5/datasetV5.yaml", 
                   r"models\yolov5m_helio\weights\best.pt", 
                   task="test", 
                   imgsz=1280,
                   conf_thres=0.45)

