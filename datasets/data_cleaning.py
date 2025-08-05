import numpy as np
from collections import defaultdict
import json
import os
from PIL import Image
from typing import List, Tuple
import glob
from pathlib import Path



def get_class_names(dataset_path):
    # Load class names
    with open(f'{dataset_path}/class_names.json', 'r') as file:
        class_names = json.load(file)
    return class_names

def CIFAR_load_images(dataset_path: str, images_dir: str, samples_path: str=None) -> Tuple[List, List[int], dict[str, str], np.ndarray]:
    """Load images from the specified dataset and optionally extract random samples per class."""
    path = os.path.join(dataset_path, images_dir)

    
    # Initialize lists to store all data
    all_images = []
    all_labels = []
    all_batch_info = []  # Store (batch_id, image_index) for each image
    class_names = {}
    batch_and_index_to_image = {}  # Map (batch_id, image_index) to image data
    batch_and_index_to_label = {}  # Map (batch_id, image_index) to label
    # Load all 5 training batches

    images = glob.glob(path + "/**/*.png", recursive=True)
    for img_path in images:
        img =  Image.open(img_path) 

        path = Path(img_path)
        parts = path.name.split('.')[0].split('_')
        label = path.parent.name
        batch_id = int(parts[1])
        img_idx = int(parts[3])

        # Add batch info for each image
        all_batch_info.append((batch_id, img_idx))
        batch_and_index_to_image[batch_id, img_idx] = img
        batch_and_index_to_label[batch_id, img_idx] = label
        all_images.append(img)
        all_labels.append(label)
        
    # Store class names (they're the same across all batches)
    class_names = get_class_names(dataset_path)
    
    # Convert to numpy arrays for easier manipulation
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    all_batch_info = np.array(all_batch_info)
    
    print(f"\nTotal images loaded: {len(all_images)}")
    print(f"Image shape: {all_images[0].shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Class distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {class_names[str(label)]}: {count} images")
    
    # get the samples already saved in the samples_path (text file every row is e.g. 'batch_1_image_440')
    if samples_path and os.path.exists(samples_path):
        print(f"\nSamples path: {samples_path}, os.path.exists(samples_path): {os.path.exists(samples_path)}")
        print(f"Loading samples from {samples_path}...")
        sampled_images = []
        sampled_labels = []
        sampled_batch_info = []
        
        samples_ids = []
        with open(samples_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples_ids.append(line)
        
        for sample_id in samples_ids:
            batch_id = sample_id.split('_')[1]  # e.g. 'batch_1_image_440' -> '1'
            image_index = sample_id.split('_')[-1]  # e.g. 'batch_1_image_440' -> '440'
            batch_id = int(batch_id)
            image_index = int(image_index)
            if batch_id < 1 or batch_id > 5 or image_index < 0 or image_index >= len(all_images):
                print(f"Warning: Invalid sample ID '{sample_id}' in {samples_path}")
                continue
            if (batch_id, image_index) in batch_and_index_to_image:
                sample_image = batch_and_index_to_image[batch_id, image_index]
                sample_label = batch_and_index_to_label[batch_id, image_index]

                sampled_images.append(sample_image)
                sampled_labels.append(sample_label)
                sampled_batch_info.append((batch_id, image_index))
        
        return sampled_images, sampled_labels, class_names, np.array(sampled_batch_info)
    else:
        print("\nNo samples extracted, returning all images and labels.")
        return all_images, all_labels.tolist(), class_names, all_batch_info



def GTSRB_load_images(dataset_path: str, images_dir: str, samples_path: str=None) -> Tuple[List, List[int], dict[str, str], np.ndarray]:
    """Load images from the GTSRB dataset."""
    path = os.path.join(dataset_path, images_dir)
    class_names = get_class_names(dataset_path)

    images = []
    labels = []
    images_paths = []
    
    for label in os.listdir(path):
        folder_path = os.path.join(path, label)
        if not os.path.isdir(folder_path):
            continue
        
        for image in os.listdir(folder_path):
            if not image.endswith('.png'):
                continue
            
            full_image_path = os.path.join(folder_path, image)
            img = Image.open(full_image_path)
            images.append(img)
            labels.append(str(int(label)))
            full_image_path = full_image_path.replace('\\', '/')
            images_paths.append(full_image_path)
    
    # get the samples already saved in the samples_path (text file every row is e.g. '00007/00008_00009')
    if samples_path and os.path.exists(samples_path):
        print(f"\nSamples path: {samples_path}, os.path.exists(samples_path): {os.path.exists(samples_path)}")
        print(f"Loading samples from {samples_path}...")        
        
        sampled_images = []
        sampled_labels = []
        sampled_images_paths = []
        with open(samples_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    indx = [i for i, s in enumerate(images_paths) if line in s][0]
                    sampled_images.append(images[indx])
                    sampled_labels.append(labels[indx])
                    sampled_images_paths.append(images_paths[indx])
        
        return sampled_images, sampled_labels, class_names, sampled_images_paths
                                          
    else:
        return images, labels, class_names, images_paths
