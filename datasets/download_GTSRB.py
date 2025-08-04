
import os
import json
from datasets import load_dataset
from tqdm import tqdm


dataset_name = "tanganke/gtsrb"
dataset_local_path = "GTSRB"

# Define local dataset path
dataset_local_path = "datasets/" + dataset_local_path
os.makedirs(f"{dataset_local_path}/", exist_ok=True)

# Load dataset splits
dataset_splits = ["train", "test"]

# https://huggingface.co/datasets/tanganke/gtsrb
dataset = {split: load_dataset(dataset_name, split=split) for split in dataset_splits}

# Extract class names
labels = dataset[dataset_splits[0]].features["label"]
class_names = {i: label for i, label in enumerate(labels.names)}


# Save class names
with open(f"{dataset_local_path}/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

# Save images and metadata
for split, data in dataset.items():
    metadata = []
    
    # Create output directories
    os.makedirs(f"{dataset_local_path}/{split}/images", exist_ok=True)

    for i, item in tqdm(enumerate(data), desc=f"Saving {split} split", total=len(data)):
        image = item["image"]  # Adjust this key based on your dataset structure
        label = item["label"]  # Adjust this key as needed

        # Define image filename
        img_filename = f"images/{i}.png"
        img_path = os.path.join(f"{dataset_local_path}/{split}", img_filename)

        # Save image
        image.save(img_path)

        # Store metadata
        metadata.append({"image": img_filename, "label": label})

        # Save metadata as JSON
        with open(f"{dataset_local_path}/{split}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

print(f"Images and metadata of GTSRB dataset saved successfully!")