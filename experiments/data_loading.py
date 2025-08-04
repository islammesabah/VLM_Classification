from Data import read_cifar_images
import numpy as np
from collections import defaultdict
import json
import os
from PIL import Image
from typing import List, Tuple
def get_class_names(dataset_path):
    # Load class names
    with open(f'{dataset_path}/class_names.json', 'r') as file:
        class_names = json.load(file)
    return class_names

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
        
        available_indices = class_indices[class_label]
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
    # sampled_indices = np.array(sampled_indices)
    print(f"type images: {type(images)}, type labels: {type(labels)}, type sampled_indices: {type(sampled_indices)}")
    # list, np.ndarray, list
    labels = labels.tolist()
    sampled_images = [images[idx] for idx in sampled_indices]
    sampled_labels = [labels[idx] for idx in sampled_indices]    
    print(f"\nSampled {len(sampled_images)} images total:")
    print(f"Sampled indices: {sampled_indices}")
    print(f"Sampled labels: {sampled_labels}")
    for class_name, count in sampled_counts.items():
        print(f"  {class_name}: {count} images")
    
    return sampled_images, sampled_labels, np.array(sampled_indices)


def CIFAR_load_images(dataset_path: str, images_dir: str, samples_path: str, samples_per_class=None, random_seed: int=42) -> Tuple[List, List[int], dict[str, str], np.ndarray]:
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
    for batch_id in range(1, 6):
        print(f"Loading batch {batch_id}...")
        batch_images, batch_labels, batch_class_names = read_cifar_images.read_images(path, batch_id)
        
        # Add batch info for each image
        for img_idx in range(len(batch_images)):
            all_batch_info.append((batch_id, img_idx))
            batch_and_index_to_image[batch_id, img_idx] = batch_images[img_idx]
            batch_and_index_to_label[batch_id, img_idx] = batch_labels[img_idx]


        all_images.extend(batch_images)
        all_labels.extend(batch_labels)
        
        # Store class names (they're the same across all batches)
        class_names = batch_class_names
    
    # Convert to numpy arrays for easier manipulation
    # all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    all_batch_info = np.array(all_batch_info)
    
    print(f"Total images loaded: {len(all_images)}")
    print(f"Image shape: {all_images[0].shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Class distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {class_names[str(label)]}: {count} images")
    
    # get the samples already saved in the samples_path (text file every row is e.g. 'batch_1_image_440')
    print(f"Samples path: {samples_path}, os.path.exists(samples_path): {os.path.exists(samples_path)}")
    if samples_path and os.path.exists(samples_path):
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
    # Extract random samples per class
    elif samples_per_class is not None:
        print(f"Extracting {samples_per_class} random samples per class with seed {random_seed}...")
        sampled_images, sampled_labels, sampled_indices = extract_random_samples_per_class(
            all_images, all_labels, class_names, samples_per_class, random_seed
        )
        sampled_batch_info = all_batch_info[sampled_indices]
        return sampled_images, sampled_labels, class_names, sampled_batch_info
    else:
        print("No samples extracted, returning all images and labels.")
        return all_images, all_labels.tolist(), class_names, all_batch_info



def GTSRB_load_images(dataset_path: str, images_dir: str) -> Tuple[List, List[int], dict[str, str], np.ndarray]:
    """Load images from the GTSRB dataset."""
    path = os.path.join(dataset_path, images_dir)
    images = []
    labels = []
    images_paths = []
    class_names = get_class_names(dataset_path)
    for label in os.listdir(path):
        folder_path = os.path.join(path, label)
        for image in os.listdir(folder_path):
            if not image.endswith('.jpeg'):
                continue
            full_image_path = os.path.join(folder_path, image)
            img = Image.open(full_image_path)
            images.append(img)
            labels.append(label)
            full_image_path = full_image_path.replace('\\', '/')
            images_paths.append(full_image_path)
    images_paths = np.array(images_paths)
    # extract random samples per class
    # if samples_per_class is not None:
    #     images, labels, sampled_indices = extract_random_samples_per_class(
    #         images, labels, class_names, samples_per_class, random_seed
    #     )
    #     print(f"sampled_indices: {sampled_indices}")
    #     print(f"images_paths: {images_paths}")
    #     sampled_indices = [int(idx) for idx in sampled_indices]
    #     images_paths = images_paths[sampled_indices]
    
    # labels = [int(label) for label in labels] 
    return images, labels, class_names, images_paths
