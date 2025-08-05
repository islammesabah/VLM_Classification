import numpy as np
from collections import defaultdict
import json
import os
from PIL import Image
from typing import List, Tuple
import glob
from pathlib import Path
import random

from data_cleaning import CIFAR_load_images, GTSRB_load_images


def extract_random_samples_per_class(images, labels, class_names, samples_per_class, random_seed=42) -> Tuple[List, List, np.ndarray]:
    """Extract a specified number of random samples from each class."""
    np.random.seed(random_seed)
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        # print(f"Processing image {idx} with label {label}")
        class_indices[label].append(idx)
            
    # Sample indices for each class
    sampled_indices = []
    sampled_counts = {}
    
    num_classes = len(class_names) if isinstance(class_names, list) else len(class_names.keys())
    
    for class_label in range(num_classes):
        if isinstance(class_names, list):
            class_name = class_names[class_label]
        else:
            class_name = class_names.get(str(class_label), f"Class_{class_label}")
        
        available_indices = class_indices[str(class_label)]
        available_count = len(available_indices)
        
        if available_count == 0:
            print(f"Warning: No images found for class '{class_name}'")
            sampled_counts[class_name] = 0
            continue
        
        # Sample min(samples_per_class, available_count) images
        sample_count = min(samples_per_class, available_count)
        selected_indices = np.random.choice(available_indices, size=sample_count, replace=False)
        sampled_indices.extend(selected_indices)
        sampled_counts[class_name] = sample_count
        
        if sample_count < samples_per_class:
            print(f"Warning: Only {sample_count} images available for class '{class_name}', requested {samples_per_class}")
    
    # Extract sampled images and labels
    print(f"type images: {type(images)}, type labels: {type(labels)}, type sampled_indices: {type(sampled_indices)}")
    sampled_images = [images[idx] for idx in sampled_indices]
    print(f"\nSampled {len(sampled_images)} images total:")
    for class_name, count in sampled_counts.items():
        print(f"  {class_name}: {count} images")
    
    return np.array(sampled_indices)


# Extract random samples per class for CIFAR
all_images, all_labels, class_names, all_batch_info = CIFAR_load_images('datasets/CIFAR-10', 'images')
print(f"\nExtracting 100 random samples per class with seed 42...")

sampled_indices = extract_random_samples_per_class(
            all_images, all_labels, class_names, 100, 42
        )

sampled_batch_info = all_batch_info[sampled_indices]

# generate sample list
sampled_list = ''
for batch_idx, img_idx in sampled_batch_info:
    sampled_list += f"batch_{batch_idx}_image_{img_idx}\n"

with open("datasets/CIFAR-10/1000_samples_1.txt", "w") as file:
    file.write(sampled_list)

# Extract random samples per class for GTSRB
images, labels, class_names, images_paths = GTSRB_load_images('datasets/GTSRB', 'Training')
print(f"\nExtracting 20 random samples per class with seed 42...")

# extract random samples per class
sampled_indices = extract_random_samples_per_class(
            images, labels, class_names, 21, 42
)

images_paths = images_paths[sampled_indices]
random.shuffle(images_paths)

# generate sample list
sampled_list = ''
for path in images_paths[:901]:
    path = path.replace("datasets/GTSRB/Training/","")
    path = path.replace(".png","")
    sampled_list += f"{path}\n"

with open("datasets/GTSRB/901_samples_1.txt", "w") as file:
    file.write(sampled_list)