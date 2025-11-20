#!/usr/bin/env python3
"""
MSCOCO Dataset Extraction and Transformation Script

This script filters MSCOCO images by the number of unique categories they contain,
samples a balanced subset, and creates various output files including:
- Filtered image information (JSON)
- Image IDs list (TXT)
- Category ID to name mapping (JSON)
- Simplified class names (JSON)

Usage:
    python extract_transform_mscoco.py <DATASET_IMAGES_PATH> [--DATASET_ANNOTATIONS_PATH PATH]
"""

import json
import os
import random
import argparse
import shutil
from collections import defaultdict
from typing import List, Dict, Set, Tuple


class COCOImageFilter:
    """Filter COCO images by number of unique categories."""
    
    def __init__(self, annotation_file_path: str):
        """
        Initialize the filter with COCO annotations.
        
        Args:
            annotation_file_path: Path to COCO annotations JSON file
        """
        self.annotation_file_path = annotation_file_path
        self.coco_data = self._load_annotations()
        self.image_categories = self._build_image_category_mapping()
    
    def _load_annotations(self) -> Dict:
        """Load COCO annotations from JSON file."""
        with open(self.annotation_file_path, 'r') as f:
            return json.load(f)
    
    def _build_image_category_mapping(self) -> Dict[int, Set[int]]:
        """Build mapping of image_id to set of category_ids."""
        image_categories = defaultdict(set)
        
        for annotation in self.coco_data['annotations']:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            image_categories[image_id].add(category_id)
        
        return dict(image_categories)

    def filter_by_category_count(self, num_categories: int) -> Tuple[List[int], int]:
        """
        Filter images by exact number of unique categories.
        
        Args:
            num_categories: Exact number of unique categories required
            
        Returns:
            Tuple of (list of image IDs matching the criteria, number of unique categories found)
        """
        filtered_images = [
            image_id for image_id, categories in self.image_categories.items()
            if len(categories) == num_categories
        ]
        unique_categories_found = set()

        # Print how many images in each category
        category_count = defaultdict(int)
        for image_id in filtered_images:
            for category_id in self.image_categories[image_id]:
                category_count[category_id] += 1
                unique_categories_found.add(category_id)

        print("\nNumber of images per category in filtered results:")
        for category_id, count in sorted(category_count.items()):
            print(f"  Category ID {category_id}: {count} images")

        print(f"\nTotal unique categories found in filtered images: {len(unique_categories_found)}")

        return filtered_images, len(unique_categories_found)
    
    def get_image_info(self, image_ids: List[int]) -> List[Dict]:
        """
        Get detailed information for given image IDs.
        
        Args:
            image_ids: List of image IDs
            
        Returns:
            List of dictionaries containing image information
        """
        image_info_list = []
        image_id_set = set(image_ids)
        
        for image in self.coco_data['images']:
            if image['id'] in image_id_set:
                categories = list(self.image_categories[image['id']])
                image_info_list.append({
                    'image_id': image['id'],
                    'file_name': image['file_name'],
                    'width': image.get('width'),
                    'height': image.get('height'),
                    'category_ids': categories,
                    'num_categories': len(categories)
                })
        
        return image_info_list
    
    def sample_images(self, image_ids: List[int], max_samples: int, unique_categories_found: int,
                     random_seed: int = None) -> List[int]:
        """
        Sample a maximum number of images from the filtered list.
        Sampling balanced across categories where each category is equally represented.
        
        Args:
            image_ids: List of image IDs to sample from
            max_samples: Maximum number of samples to return
            unique_categories_found: Number of unique categories found in filtered images
            random_seed: Random seed for reproducibility (optional)
            
        Returns:
            Sampled list of image IDs
        """
        if len(image_ids) <= max_samples:
            return image_ids
        if random_seed is not None:
            random.seed(random_seed)
        
        images_per_category = max_samples // unique_categories_found
        category_to_images = defaultdict(list)
        for image_id in image_ids:
            for category_id in self.image_categories[image_id]:
                category_to_images[category_id].append(image_id)

        sampled_images = []
        for category_id, images in category_to_images.items():
            if len(images) > images_per_category:
                sampled_images.extend(random.sample(images, images_per_category))
            else:
                sampled_images.extend(images)

        print(f"\nTotal balanced sampled images: {len(sampled_images)}")

        return sampled_images
    
    def save_results(self, image_info: List[Dict], output_file: str,
                    total_filtered: int, num_categories: int):
        """
        Save filtered results to JSON file.
        
        Args:
            image_info: List of image information dictionaries
            output_file: Path to output JSON file
            total_filtered: Total number of images before sampling
            num_categories: Number of categories used for filtering
        """
        output_data = {
            'filter_criteria': {
                'num_categories': num_categories,
                'total_matched': total_filtered,
                'sampled': len(image_info)
            },
            'dataset_stats': {
                'total_images': len(self.coco_data['images']),
                'total_annotations': len(self.coco_data['annotations'])
            },
            'images': image_info
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def copy_images_to_folder(self, image_info: List[Dict], 
                             source_dir: str, dest_dir: str) -> int:
        """
        Copy filtered images to a destination folder.
        
        Args:
            image_info: List of image information dictionaries
            source_dir: Source directory containing the images
            dest_dir: Destination directory to copy images to
            
        Returns:
            Number of images successfully copied
        """
        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        
        copied_count = 0
        failed_count = 0
        
        print(f"\nCopying images to: {dest_dir}")
        
        for img in image_info:
            source_path = os.path.join(source_dir, img['file_name'])
            dest_path = os.path.join(dest_dir, img['file_name'])
            
            try:
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                    if copied_count % 100 == 0:
                        print(f"  Copied {copied_count}/{len(image_info)} images...")
                else:
                    print(f"  Warning: Source file not found: {source_path}")
                    failed_count += 1
            except Exception as e:
                print(f"  Error copying {img['file_name']}: {e}")
                failed_count += 1
        
        print(f"Successfully copied {copied_count} images")
        if failed_count > 0:
            print(f"Failed to copy {failed_count} images")
        
        return copied_count
    
    def print_statistics(self, filtered_count: int, sampled_count: int, 
                        num_categories: int):
        """Print filtering statistics."""
        total_images = len(self.coco_data['images'])
        percentage = (filtered_count / total_images * 100) if total_images > 0 else 0
        
        print(f"{'='*60}")
        print(f"COCO Image Filtering Results")
        print(f"{'='*60}")
        print(f"Total images in dataset: {total_images:,}")
        print(f"Images with exactly {num_categories} unique category(ies): {filtered_count:,}")
        print(f"Percentage: {percentage:.2f}%")
        print(f"Sampled images: {sampled_count:,}")
        print(f"{'='*60}")


def save_category_mapping(coco_data: Dict, output_dir: str):
    """
    Save category ID to name mapping files.
    
    Args:
        coco_data: COCO dataset JSON data
        output_dir: Output directory for the mapping files
    """
    # Build dictionary mapping category IDs to names, supercategories
    category_id_to_name = {
        category['id']: {
            'name': category['name'],
            'supercategory': category['supercategory']
        }
        for category in coco_data['categories']
    }
    
    # Save the full category_id_to_name mapping
    category_mapping_file = os.path.join(output_dir, 'category_id_to_name.json')
    with open(category_mapping_file, 'w') as f:
        json.dump(category_id_to_name, f, indent=2)
    print(f"Category mapping saved to: {category_mapping_file}")
    
    # Save simplified class names (ID -> name only)
    simple_class_names = {int(k): v['name'] for k, v in category_id_to_name.items()}
    class_names_file = os.path.join(output_dir, 'class_names.json')
    with open(class_names_file, 'w') as f:
        json.dump(simple_class_names, f, indent=2)
    print(f"Class names saved to: {class_names_file}")


def save_image_ids(image_info: List[Dict], output_file: str):
    """
    Save image IDs to a text file.
    
    Args:
        image_info: List of image information dictionaries
        output_file: Path to output text file
    """
    image_ids = [img['image_id'] for img in image_info]
    with open(output_file, 'w') as f:
        for image_id in image_ids:
            f.write(f"{image_id}\n")
    print(f"Image IDs saved to: {output_file}")


def print_unique_category_statistics(image_info: List[Dict]):
    """
    Print statistics about unique categories in the filtered images.
    
    Args:
        image_info: List of image information dictionaries
    """
    unique_categories = set()
    for img in image_info:
        for cat_id in img['category_ids']:
            unique_categories.add(cat_id)
    
    print(f"\nNumber of unique categories in final filtered images: {len(unique_categories)}")


def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract and transform MSCOCO dataset by filtering images based on category count.'
    )
    parser.add_argument(
        'DATASET_IMAGES_PATH', 
        type=str, 
        default='train2014/', 
        help='Path to the MSCOCO dataset directory which contains the images'
    )
    parser.add_argument(
        '--DATASET_ANNOTATIONS_PATH', 
        type=str, 
        default='annotations/', 
        help='Path to the MSCOCO dataset directory which contains the annotations'
    )
    parser.add_argument(
        '--num_categories',
        type=int,
        default=1,
        help='Number of unique categories to filter for (default: 1)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=1000,
        help='Maximum number of images to sample (default: 1000)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--copy_images',
        action='store_true',
        help='Copy images to output directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for generated files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    IMAGES_SOURCE_DIR = args.DATASET_IMAGES_PATH
    ANNOTATIONS_DIR = args.DATASET_ANNOTATIONS_PATH

    
    # Configuration
    ANNOTATION_FILE_PATH = os.path.join(ANNOTATIONS_DIR, 'instances_train2014.json')
    NUM_CATEGORIES = args.num_categories
    MAX_SAMPLES = args.max_samples
    RANDOM_SEED = args.random_seed
    COPY_IMAGES = args.copy_images
    OUTPUT_DIR = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load annotations and save category mapping
    print(f"Loading annotations from: {ANNOTATION_FILE_PATH}")
    with open(ANNOTATION_FILE_PATH, 'r') as f:
        coco_data = json.load(f)
    
    print("\nSaving category mappings...")
    save_category_mapping(coco_data, OUTPUT_DIR)
    
    # Initialize filter
    filter_obj = COCOImageFilter(ANNOTATION_FILE_PATH)

    # Filter images
    print(f"\nFiltering images with exactly {NUM_CATEGORIES} unique category(ies)...")
    filtered_image_ids, unique_categories_found = filter_obj.filter_by_category_count(NUM_CATEGORIES)

    # Output files
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'filtered_{NUM_CATEGORIES}_category_images.json')
    OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, f'filtered_{NUM_CATEGORIES}_category_images/')
    IMAGE_IDS_FILE = os.path.join(OUTPUT_DIR, f'{len(filtered_image_ids)}_filtered_{NUM_CATEGORIES}_category_image_ids.txt') # {numberofsamples}_filtered_{NUM_CATEGORIES}_category_image_ids.txt
    
    # Sample images if needed
    if MAX_SAMPLES is not None and len(filtered_image_ids) > MAX_SAMPLES:
        print(f"\nSampling {MAX_SAMPLES} images from {len(filtered_image_ids)} filtered images...")
        sampled_image_ids = filter_obj.sample_images(
            filtered_image_ids, 
            MAX_SAMPLES,
            unique_categories_found,
            RANDOM_SEED
        )
    else:
        sampled_image_ids = filtered_image_ids
    
    # Get detailed image information
    image_info = filter_obj.get_image_info(sampled_image_ids)
    
    # Save results
    filter_obj.save_results(
        image_info, 
        OUTPUT_FILE,
        len(filtered_image_ids),
        NUM_CATEGORIES
    )
    
    # Print statistics
    filter_obj.print_statistics(
        len(filtered_image_ids),
        len(sampled_image_ids),
        NUM_CATEGORIES
    )
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    
    # Save image IDs to text file
    save_image_ids(image_info, IMAGE_IDS_FILE)
    
    # Print unique category statistics
    print_unique_category_statistics(image_info)
    
    # Copy images to folder
    if COPY_IMAGES:
        copied = filter_obj.copy_images_to_folder(
            image_info,
            IMAGES_SOURCE_DIR,
            OUTPUT_IMAGES_DIR
        )
        print(f"\nImages copied to: {OUTPUT_IMAGES_DIR}")
    
    # Show first 10 examples
    if image_info:
        print(f"\nFirst {min(10, len(image_info))} examples:")
        for i, img in enumerate(image_info[:10], 1):
            print(f"{i}. {img['file_name']}")
            print(f"   Image ID: {img['image_id']}, Category IDs: {img['category_ids']}")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
