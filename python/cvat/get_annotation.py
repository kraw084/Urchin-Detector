import xmltodict

def get_annotation_dict (cvat_annotation_xml_path): 
    with open(cvat_annotation_xml_path) as f:
        dataset = xmltodict.parse(f.read())
    annotation_data_dict = dict()
    for index, frame_metadata in enumerate(dataset["annotations"]["image"]): # for each frame
        frame_name = frame_metadata["@name"]
        annotation_data_dict[frame_name] = {"boxes": [], "labels": [], "tags": []}
        if 'box' in frame_metadata.keys():
            boxes_metadata = frame_metadata["box"]
            if not type(boxes_metadata) is list:
                boxes_metadata = [boxes_metadata]
            for box_metadata in boxes_metadata:
                label = box_metadata["@label"]
                box = [float(box_metadata["@xtl"]),float(box_metadata["@ytl"]),float(box_metadata["@xbr"]),float(box_metadata["@ybr"])]
                tag = []
                if box_metadata["attribute"][0]["#text"]=="true":
                    tag.append("missed")
                if box_metadata["attribute"][1]["#text"]=="true":
                    tag.append("false-positive")
                if box_metadata["attribute"][2]["#text"]=="true":
                    tag.append("miss-classified")
                
                annotation_data_dict[frame_name]["boxes"].append(box)
                annotation_data_dict[frame_name]["labels"].append(label)
                annotation_data_dict[frame_name]["tags"].append(tag)
    return annotation_data_dict

# annotation_data_dict = get_annotation_dict ('annotations.xml')
