from urchin_utils.model_utils import project_sys_path, WEIGHTS_PATH
from urchin_utils.data_utils import DATASET_YAML_PATH

project_sys_path()
import yolov5.train
import yolov5.val

if __name__ == "__main__":
    yolov5.val.run(DATASET_YAML_PATH, 
                   WEIGHTS_PATH, 
                   task="test", 
                   imgsz=1280,
                   conf_thres=0.45)

