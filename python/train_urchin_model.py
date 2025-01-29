from urchin_utils.model_utils import project_sys_path
from urchin_utils.data_utils import DATASET_YAML_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    yolov5.train.run(imgsz = 1280, 
                    epochs = 200, 
                    data = DATASET_YAML_PATH, 
                    weights = "yolov5m.pt", 
                    save_period = 10,
                    batch_size = -1,
                    #cache = "ram",
                    patience = 50,
                    hyp = "models/hyp.overfit.yaml",
                    name = "helio_v1"
                    ) 

