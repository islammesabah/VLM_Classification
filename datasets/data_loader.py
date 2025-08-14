from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
import os
from PIL import Image

from tree_generation.gtsrb_tree import build_traffic_sign_tree
from tree_generation.cifar_tree import build_cifar_tree

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


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load_images(self, config) -> Tuple[List, List[int], dict[str, str], Any]:
        """Load images from dataset."""
        pass
    
    @abstractmethod
    def get_class_names(self) -> Dict[str, str]:
        """Get class names mapping."""
        pass
    
    @abstractmethod
    def build_tree(self) -> Any:
        """Build decision tree for the dataset."""
        pass
    
    @abstractmethod
    def get_zero_shot_prompt(self, config, RunID) -> str:
        """Get the zero-shot inference prompt."""
        pass
    
    @abstractmethod
    def extract_image_info(self, image_idx: int, additional_batch_info: Any) -> Tuple[str, str]:
        """Extract image path and sequence information."""
        pass


class CIFARLoader(DatasetLoader):
    """CIFAR-10 dataset loader."""
    
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def read_images(self, path: str, batch_id: int) -> tuple[np.ndarray, list, dict[str, str]]:
        """
        Reads images from a specified batch file in the CIFAR-10 dataset.

        Args:
            path (str): The path to the CIFAR-10 batch file.
            batch_id (int): The ID of the batch to read.

        Returns:
            tuple: A tuple containing:
                - images (numpy.ndarray): An array of shape (10000, 32, 32, 3) containing the images.
                - labels (list): A list of labels corresponding to the images.
                - class_names (list): A list of class names.
        """
        batch_file = f"{path}/data_batch_{batch_id}"
        data_dict = self.unpickle(batch_file)
        
        # Reshape the data correctly: the data is in format [R, G, B] channels flattened
        raw_data = np.array(data_dict[b'data'], dtype=np.uint8)
        
        # Reshape to (10000, 3, 32, 32) then transpose to (10000, 32, 32, 3)
        images = raw_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = data_dict[b'labels']
        
        # Read meta data
        meta_file = f"{path}/batches.meta"
        meta_dict = self.unpickle(meta_file)
        # class names is dict "0": "airplane", "1": "automobile", ..., "9": "truck"
        class_names = meta_dict[b'label_names']
        class_names = {str(i): name.decode('utf-8') for i, name in enumerate(class_names)}
        return images, labels, class_names

    def CIFAR_load_images(self, dataset_path: str, images_dir: str, samples_path: str, samples_per_class=None, random_seed: int=42) -> Tuple[List, List[int], dict[str, str], np.ndarray]:
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
            batch_images, batch_labels, batch_class_names = self.read_images(path, batch_id)

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


    def load_images(self, config) -> Tuple[List, List[int], dict[str, str], Any]:
        return self.CIFAR_load_images(
            config.dataset_path, 
            config.images_dir, 
            config.samples_path
        )
    
    def get_class_names(self) -> Dict[str, str]:
        # This should be returned from load_images, but keeping for interface consistency
        return {}
    
    def build_tree(self):
        return build_cifar_tree()
    
    def get_zero_shot_prompt(self, config, RunID) -> str:
        return config["zero_shot_prompts"][RunID]
    
    def extract_image_info(self, image_idx: int, additional_batch_info: Any) -> Tuple[str, str]:
        batch_id, orig_img_idx = additional_batch_info[image_idx]
        image_path = f"batch_{batch_id}_image_{orig_img_idx}"
        sequence = str(batch_id)
        return image_path, sequence


class GTSRBLoader(DatasetLoader):
    """GTSRB dataset loader."""
    
    def GTSRB_load_images(self, dataset_path: str, images_dir: str) -> Tuple[List, List[int], dict[str, str], np.ndarray]:
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
        return images, labels, class_names, images_paths
    
    def load_images(self, config) -> Tuple[List, List[int], Dict[str, str], Any]:
        return self.GTSRB_load_images(
            config.dataset_path, 
            config.images_dir)
    
    def get_class_names(self) -> Dict[str, str]:
        # This should be returned from load_images, but keeping for interface consistency
        return {}
    
    def build_tree(self):
        return build_traffic_sign_tree()
    
    def get_zero_shot_prompt(self, config, RunID) -> str:
        return config["zero_shot_prompts"][RunID]
    
    def extract_image_info(self, image_idx: int, additional_batch_info: Any) -> Tuple[str, str]:
        # For GTSRB, we assume additional_batch_info is a list of image paths
        image_path = additional_batch_info[image_idx]
        image_name = Path(image_path).name
        sequence = image_name.split('_')[0]
        return image_path, sequence
