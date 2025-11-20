"""
MNIST Dataset Extraction and Transformation Script

This script extracts images from the MNIST dataset (IDX format),
samples them equally and randomly from each class, and saves them
as individual image files with proper metadata.

The output structure matches the MSCOCO extraction pattern:
- A folder with sampled images as PNG files
- A text file with image IDs
- Metadata JSON with image-to-class mappings
"""

import json
import os
import random
import struct
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from PIL import Image


class MNISTExtractor:
    """Extract and transform MNIST dataset from IDX format."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the MNIST extractor.
        
        Args:
            dataset_path: Path to MNIST dataset directory containing IDX files
        """
        self.dataset_path = dataset_path
        self.images = None
        self.labels = None
        
    def read_idx_images(self, filename: str) -> np.ndarray:
        """
        Read images from MNIST IDX file format.
        
        Args:
            filename: Path to IDX image file
            
        Returns:
            numpy array of images with shape (n_images, height, width)
        """
        with open(filename, 'rb') as f:
            # Read magic number and dimensions
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2051:
                raise ValueError(f'Invalid magic number {magic} in MNIST image file')
            
            n_images = struct.unpack('>I', f.read(4))[0]
            n_rows = struct.unpack('>I', f.read(4))[0]
            n_cols = struct.unpack('>I', f.read(4))[0]
            
            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(n_images, n_rows, n_cols)
            
        return images
    
    def read_idx_labels(self, filename: str) -> np.ndarray:
        """
        Read labels from MNIST IDX file format.
        
        Args:
            filename: Path to IDX label file
            
        Returns:
            numpy array of labels
        """
        with open(filename, 'rb') as f:
            # Read magic number and dimensions
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2049:
                raise ValueError(f'Invalid magic number {magic} in MNIST label file')
            
            n_labels = struct.unpack('>I', f.read(4))[0]
            
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        return labels
    
    def load_mnist_data(self, use_train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MNIST images and labels from IDX files.
        
        Args:
            use_train: If True, load training set, otherwise load test set
            
        Returns:
            Tuple of (images, labels)
        """
        if use_train:
            images_file = os.path.join(self.dataset_path, 'train-images.idx3-ubyte')
            labels_file = os.path.join(self.dataset_path, 'train-labels.idx1-ubyte')
        else:
            images_file = os.path.join(self.dataset_path, 't10k-images.idx3-ubyte')
            labels_file = os.path.join(self.dataset_path, 't10k-labels.idx1-ubyte')
        
        print(f"Loading images from: {images_file}")
        print(f"Loading labels from: {labels_file}")
        
        images = self.read_idx_images(images_file)
        labels = self.read_idx_labels(labels_file)
        
        print(f"Loaded {len(images)} images with shape {images.shape}")
        print(f"Loaded {len(labels)} labels")
        
        self.images = images
        self.labels = labels
        
        return images, labels
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of samples across classes.
        
        Returns:
            Dictionary mapping class labels to counts
        """
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
        return distribution
    
    def sample_balanced(self, samples_per_class: int, random_seed: int = 42) -> Tuple[List[int], Dict[int, int]]:
        """
        Sample images equally from each class.
        
        Args:
            samples_per_class: Number of samples to take from each class
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (sampled_indices, samples_per_class_actual)
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Group indices by class
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_to_indices[int(label)].append(idx)
        
        sampled_indices = []
        actual_samples = {}
        
        print(f"\nSampling {samples_per_class} images per class...")
        for class_label in sorted(class_to_indices.keys()):
            available_indices = class_to_indices[class_label]
            available_count = len(available_indices)
            
            # Sample min(samples_per_class, available_count)
            sample_count = min(samples_per_class, available_count)
            selected_indices = np.random.choice(available_indices, size=sample_count, replace=False).tolist()
            sampled_indices.extend(selected_indices)
            actual_samples[class_label] = sample_count
            
            print(f"  Class {class_label}: sampled {sample_count}/{available_count} images")
        
        return sampled_indices, actual_samples
    
    def save_images_to_folder(self, indices: List[int], output_dir: str, 
                             prefix: str = "mnist") -> List[Dict]:
        """
        Save selected images as PNG files.
        
        Args:
            indices: List of image indices to save
            output_dir: Directory to save images to
            prefix: Prefix for image filenames
            
        Returns:
            List of image metadata dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_metadata = []
        
        print(f"\nSaving {len(indices)} images to: {output_dir}")
        
        for i, idx in enumerate(indices):
            # Get image and label
            image_array = self.images[idx]
            label = int(self.labels[idx])
            
            # Create filename with zero-padded index
            filename = f"{prefix}_{idx:06d}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Convert to PIL Image and save
            img = Image.fromarray(image_array, mode='L')  # 'L' mode for grayscale
            img.save(filepath)
            
            # Store metadata with zero-padded image_id
            image_metadata.append({
                'image_id': f"{idx:06d}",
                'file_name': filename,
                'label': label,
                'width': 28,
                'height': 28
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Saved {i + 1}/{len(indices)} images...")
        
        print(f"Successfully saved {len(indices)} images")
        
        return image_metadata
    
    def save_metadata(self, image_metadata: List[Dict], output_file: str, 
                     samples_per_class: Dict[int, int]):
        """
        Save image metadata to JSON file.
        
        Args:
            image_metadata: List of image metadata dictionaries
            output_file: Path to output JSON file
            samples_per_class: Dictionary of actual samples per class
        """
        # Load class names
        class_names_file = os.path.join(self.dataset_path, 'class_names.json')
        with open(class_names_file, 'r') as f:
            class_names = json.load(f)
        
        output_data = {
            'dataset': 'MNIST',
            'total_sampled': len(image_metadata),
            'samples_per_class': samples_per_class,
            'class_names': class_names,
            'image_shape': {
                'height': 28,
                'width': 28,
                'channels': 1
            },
            'images': image_metadata
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nMetadata saved to: {output_file}")
    
    def save_image_ids(self, image_metadata: List[Dict], output_file: str):
        """
        Save image IDs to a text file (one per line).
        
        Args:
            image_metadata: List of image metadata dictionaries
            output_file: Path to output text file
        """
        with open(output_file, 'w') as f:
            for img in image_metadata:
                f.write(f"{img['image_id']}\n")
        
        print(f"Image IDs saved to: {output_file}")
    
    def create_image_to_class_mapping(self, image_metadata: List[Dict], 
                                     output_file: str):
        """
        Create a JSON file mapping image IDs to class labels.
        
        Args:
            image_metadata: List of image metadata dictionaries
            output_file: Path to output JSON file
        """
        mapping = {
            img['image_id']: img['label']  # image_id is already zero-padded string
            for img in image_metadata
        }
        
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Image-to-class mapping saved to: {output_file}")
    
    def print_statistics(self, image_metadata: List[Dict], 
                        samples_per_class: Dict[int, int]):
        """Print extraction statistics."""
        # Load class names
        class_names_file = os.path.join(self.dataset_path, 'class_names.json')
        with open(class_names_file, 'r') as f:
            class_names = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"MNIST Image Extraction Results")
        print(f"{'='*60}")
        print(f"Total images in dataset: {len(self.images):,}")
        print(f"Total sampled images: {len(image_metadata):,}")
        print(f"\nSamples per class:")
        for class_label in sorted(samples_per_class.keys()):
            class_name = class_names[str(class_label)]
            count = samples_per_class[class_label]
            print(f"  Class {class_label} ({class_name}): {count} images")
        print(f"{'='*60}")


def main():
    """Main execution function."""
    
    # Configuration
    DATASET_PATH = './'
    SAMPLES_PER_CLASS = 100  # Number of samples per class (10 classes = 1000 total)
    USE_TRAIN = True  # Use training set (True) or test set (False)
    OUTPUT_DIR = os.path.join(DATASET_PATH, 'sampled_images')
    OUTPUT_METADATA = os.path.join(DATASET_PATH, 'sampled_images_metadata.json')
    OUTPUT_IMAGE_IDS = os.path.join(DATASET_PATH, 'sampled_image_ids.txt')
    OUTPUT_MAPPING = os.path.join(DATASET_PATH, 'image_to_class_mapping.json')
    RANDOM_SEED = 42
    
    # Initialize extractor
    print(f"Initializing MNIST extractor...")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Using {'training' if USE_TRAIN else 'test'} set")
    print(f"Random seed: {RANDOM_SEED}\n")
    
    extractor = MNISTExtractor(DATASET_PATH)
    
    # Load MNIST data
    extractor.load_mnist_data(use_train=USE_TRAIN)
    
    # Print class distribution
    distribution = extractor.get_class_distribution()
    print("\nOriginal class distribution:")
    for label, count in sorted(distribution.items()):
        print(f"  Class {label}: {count} images")
    
    # Sample balanced data
    sampled_indices, actual_samples = extractor.sample_balanced(
        SAMPLES_PER_CLASS, 
        RANDOM_SEED
    )
    
    # Save images to folder
    image_metadata = extractor.save_images_to_folder(
        sampled_indices,
        OUTPUT_DIR,
        prefix="mnist"
    )
    
    # Save metadata
    extractor.save_metadata(
        image_metadata,
        OUTPUT_METADATA,
        actual_samples
    )
    
    # Save image IDs to text file
    extractor.save_image_ids(image_metadata, OUTPUT_IMAGE_IDS)
    
    # Create image-to-class mapping
    extractor.create_image_to_class_mapping(image_metadata, OUTPUT_MAPPING)
    
    # Print statistics
    extractor.print_statistics(image_metadata, actual_samples)
    
    print("\n✓ MNIST extraction and transformation completed successfully!")


if __name__ == "__main__":
    main()
