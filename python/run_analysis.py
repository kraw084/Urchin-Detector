import os

from urchin_utils.data_utils import dataset_by_id
from urchin_utils.model_utils import UrchinDetector_YoloV5
from urchin_utils.eval_utils import validiate
from urchin_utils.vis_utils import annotate_preds_on_folder
from analysis_tools import compare_to_gt, save_detections, metrics_by_var, calibration_curve, bin_by_count


if __name__ == "__main__":
    weight_path = "models/yolov5m_helio/weights/best.pt"
    val_txt = "data/datasets/full_dataset_v5/val.txt"
    test_txt = "data/datasets/full_dataset_v5/test.txt"
    d = dataset_by_id("data/csvs/High_conf_clipped_dataset_V5.csv")
    yolov5_model_helio = UrchinDetector_YoloV5(weight_path)
    
    compare_to_gt(yolov5_model_helio, val_txt, d, "all", display_correct=True)

    #calibration_curve(yolov5_model_helio, val_txt, d)

    #validiate(yolov5_model_helio, val_txt, d)

    #compare_to_gt(yolov5_model_helio, val_txt, d, "all", filter_var="source",
    #              filter_func=lambda x: x == "UoA Sea Urchin")
    #NSW DPI Urchins
    #UoA Sea Urchin
    #Urchins - Eastern Tasmania
    #RLS- Heliocidaris PPB

    #metrics_by_var(yolov5_model_helio, val_txt, d, "source")

    #bin_by_count(yolov5_model_helio, val_txt, d, 5, True)
    
    
