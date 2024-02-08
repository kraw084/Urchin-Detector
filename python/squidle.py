import torch

#loading the model
weights_path = "models/yolov5m-highRes-ro/weights/best.pt"
model = torch.hub.load("yolov5", "custom", path=weights_path, source="local")
model.cpu()

#Setting model parameters
model.conf = 0.45
img_size = 1280

images = [f"data/images/im{i}.JPG" for i in range(5)] + ["data/images/im30.JPG"]

predictions = model(images, size = img_size).xywhn
#list of tensors, 1 for each image
#each tensor has a row of the form [x, y, w, h, conf, label] for each urchin prediction (normalised by im w and h)

for image_pred in predictions:
    print(image_pred)
