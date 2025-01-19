from urchin_utils.data_utils import dataset_by_id
from urchin_utils.model_utils import UrchinDetector_YoloV5
from urchin_utils.eval_utils import validiate, metrics_by_var
from analysis_tools import compare_to_gt


if __name__ == "__main__":
    weight_path = "models/yolov5m_helio/weights/best.pt"
    val_txt = "data/datasets/full_dataset_v5/val.txt"
    test_txt = "data/datasets/full_dataset_v5/test.txt"
    d = dataset_by_id("data/csvs/High_conf_clipped_dataset_V5.csv")
    yolov5_model_helio = UrchinDetector_YoloV5(weight_path)
    
    
    compare_to_gt(yolov5_model_helio, val_txt, d, "all", display_correct=True)

    #validiate(yolov5_model_helio, val_txt, d)

    #compare_to_gt(yolov5_model_helio, val_txt, d, "all", filter_var="source",
    #              filter_func=lambda x: x == "UoA Sea Urchin")
    #NSW DPI Urchins
    #UoA Sea Urchin
    #Urchins - Eastern Tasmania
    #RLS- Heliocidaris PPB

    #metrics_by_var(yolov5_model_helio, val_txt, d, "source")
