from urchin_utils.model_utils import project_sys_path, WEIGHTS_PATH
from urchin_utils.data_utils import DATASET_YAML_PATH

project_sys_path()
import yolov5.val

if __name__ == "__main__":
    #IMPORTANT: if fixed_conf_for_pr is None then the confidence will be optimised on the specifed set and the optimal p and r
    #will be printed. This should not be done on the test set; instead, set fixed_conf_for_pr to the desired confidence.

    yolov5.val.run(DATASET_YAML_PATH, 
                   WEIGHTS_PATH, 
                   task="test", 
                   imgsz=1280,
                   fixed_conf_for_pr=0.45)

