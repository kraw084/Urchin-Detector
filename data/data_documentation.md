# Dataset Documentation
The data folder contains everything needed to create datasets for training and evaluating urchin detector models. 

The core sea urchin object detection dataset is stored in two parts: a csv and the images. The csv contains the bounding box information for each image and are constructed by formating downloaded squidle annotation file. The columns in the csv are as follows:

- id: squidle id that uniquely identifies the image
- url: url of the image, can be used to download it
- name: name of the image, these may not be unique across deployment
- width: width of the image in pixels
- height: height of the image in pixels
- source: the name of the data set the image was sourced from
- deployment: deployment the image was taken from
- campaign: campaign the image was take from
- latitude: latitude the image was taken at
- longitude: longitude the image was take at
- depth: depth the image was taken at
- altitude: height above sea floor the image was taken at
- time: the time the photo was taken 
- flagged: True or false, whether the image contains a box that is flagged for review (may not contain an urchin)
- count: number of urchins in this image
- Evechinus: True/False if the image contains at least 1 kina
- Centrostephanus: True/False if the image contains at least 1 centro
- Heliocidaris: True/False if the image contains at least 1 helio
- boxes: a list of bounding boxes (an empty list if there are none), each box is a tuple in the form (class, confidence, x, y, w, h, flagged)
    - Class: species of the urchin in the box ("Evechinus chloroticus" or "Centrostephanus rodgersii")
    - Confidence: value between 0 and 1 that represents the confidence that the object in the box is an urchin
    - x: x coordinate of the boxes center point, relative to the images width
    - y: y coordinate of the boxes center point, relative to the image height
    - w: width of the bounding box, relative to the images width
    - h: height of the bounding box, relative to the images height
    - flagged: True or false, whether the box is flagged for review
    

## Organization
The data folder is structured as follows:

1. **csvs**: contains the csv files with bounding box data for each image
2. **datasets**: contains the yolov5 dataset yaml files and train/val/test txt files
3. **python**: contains the python scripts used to create and format the dataset

Every function in the python scripts contains a docstring explaining what it does and the parameters it takes.

## Setting up an existing data
To create a dataset for training a yolov5 model using an existing formatted csv file (e.g. ```data/csvs/High_conf_clipped_dataset_V5.csv```), two things need to be done. First run the ```download_images.py```script to download the images from the urls in the csv file. After the images are downloaded run the ```exif_correction.py``` script. This reads each image and applies the exif rotation to them before stripping the data. This ensures that all applications and software read the images the same way and bounding box data correctly aligns with the images.

Next run the ```create_yolo_labels.py``` script to create the label files for each image. It is recommended to not change the directory these scripts save the images and labels to (```data/images``` and ```data/labels```) and other scripts may depend on them.

## Creating a new dataset
To create a new dataset from a squidle annotation file, several steps need to be done to format the csv, download the images, create the label files and setup the yolov5 dataset yaml and txt files. 

All the scripts that you will need to run to create a new dataset are preset to create the High_conf_clipped_dataset_V5 dataset. This means you should only need to change the input and output csv names and not need to call any additional functions that arent already called.

### 1. Format the csv
```data/python/format_csv.py``` contains several functions used to format squidle annotations. This is done to remove unecessary data and make the csv for usable for object detection. 

The squidle annotation file must have the following columns: point.media.path_best, label.name, point.polygon, point.media.id, point.media.deployment.campaign.name, point.media.deployment.name, point.pose.dep, point.pose.lat, point.pose.lon, point.pose.timestamp, point.pose.alt, likelihood, point.x, point.y, needs_review

1. first, run ```format_csv(csv_file_path, source_name, formated_csv_name)``` on each squidle csv annotation file. This does the bulk of the formatting work and creates a new csv file.
2. If the dataset you wish to create is comprised of multiple squidle annotation files, each one should be formatted seperately then combined using the ```concat_formated_csvs(csv_paths, concat_csv_name)```.
3. To remove urchin annotations that have low confidence or are flagged for review, run ```high_conf_csv(input_csv, output_csv_name, conf_cutoff)``` to created a new filtered csv file.

### 2. Download the images
Now that the formatted csv is setup the images can be downloaded as described in the "Setting up an existing data" section. After the images are downloaded and exif data has been removed run ```set_wh_col(input_csv, output_csv_name, im_dir)``` to set the width and height columns of the csv file. This has to be done seperately to the rest of the formatting as it requires the images to be downloaded. Lastly, run ```clip_boxes(input_csv, output_csv_name)``` to clip the bounding boxes to the image size. This is done as yolov5 models will only create preidctions within the bounds of the image. 

### 3. Create the label files
The label files as described in the "Setting up an existing data" section.

### 4. Setup the yolov5 dataset yaml and txt files
yolov5 models require a yaml file to specify the class labels (and there corresoponding ids) and the train/val/test txt paths. In ```data/datasets``` create a new folder with your datasets name. Inside this folder create a new yaml file. See the existing dataset yaml file (```data/datasets/full_dataset_v5/datasetv5.yaml```) for an example of what this should look like. 

Next, to create train/val/test splits run ```python/partition_dataset.py```. This will create a txt of image names for each of the dataset splits. Move these into the new folder you have created alongside the yaml file.

