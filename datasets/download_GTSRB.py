
import os
import json
import requests
import zipfile
import glob
from PIL import Image

dataset_name = "tanganke/gtsrb"
dataset_local_path = "GTSRB"

# Define local dataset path
dataset_local_path = "datasets/" + dataset_local_path
os.makedirs(f"{dataset_local_path}/", exist_ok=True)

# get class names
class_names = {
  "0": "20 Kph Limit",
  "1": "30 Kph Limit",
  "2": "50 Kph Limit",
  "3": "60 Kph Limit",
  "4": "70 Kph Limit",
  "5": "80 Kph Limit",
  "6": "End Of Restriction",
  "7": "100 Kph Limit",
  "8": "120 Kph Limit",
  "9": "No Cars Passing",
  "10": "No Trucks Passing",
  "11": "Right-Of-Way At Intersection",
  "12": "Priority Road",
  "13": "Yield",
  "14": "Stop",
  "15": "Empty Circle",
  "16": "No Truck Entry",
  "17": "No Entry",
  "18": "Exclamation Mark Warning",
  "19": "Left Curve Warning",
  "20": "Right Curve Warning",
  "21": "Double Curve Warning",
  "22": "Rough/Bumpy Road Warning",
  "23": "Slippery Road Warning",
  "24": "Merging/Narrow Lanes Warning",
  "25": "Construction/Road Work Warning",
  "26": "Traffic Light Warning",
  "27": "Pedestrian Warning",
  "28": "Child And Pedestrian Warning",
  "29": "Bicycle Warning",
  "30": "Ice/Snow Warning",
  "31": "Deer Warning",
  "32": "No Speed Limit",
  "33": "Right Arrow",
  "34": "Left Arrow",
  "35": "Forward Arrow",
  "36": "Forward-Right Arrow",
  "37": "Forward-Left Arrow",
  "38": "Keep-Right Arrow",
  "39": "Keep-Left Arrow",
  "40": "Circle Arrow",
  "41": "End Car Passing Ban",
  "42": "End Truck Passing Ban"
}


# Save class names
with open(f"{dataset_local_path}/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)
    
    
# download the dataset
path = "datasets/"
response = requests.get("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip")
with open(path+'GTSRB-Training_fixed.zip', "wb") as f:
    f.write(response.content)

with zipfile.ZipFile(path+'GTSRB-Training_fixed.zip', 'r') as zip_ref:
    zip_ref.extractall(path)
    
if os.path.exists(path+'GTSRB-Training_fixed.zip'):
    os.remove(path+'GTSRB-Training_fixed.zip')

# change to png
files = glob.glob("datasets/GTSRB/**/*.ppm", recursive=True)
for img_path in files:
    img = Image.open(img_path)
    img.save(img_path.split(".")[0]+".png", "PNG")
    if os.path.exists(img_path):
        os.remove(img_path)
        
print(f"\nImages and metadata of GTSRB dataset saved successfully!")