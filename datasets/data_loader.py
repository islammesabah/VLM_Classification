from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from pathlib import Path

from tree_generation.gtsrb_tree import build_traffic_sign_tree
from tree_generation.cifar_tree import build_cifar_tree
from .data_cleaning import CIFAR_load_images, GTSRB_load_images

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
    
    def load_images(self, config) -> Tuple[List, List[int], dict[str, str], Any]:
        return CIFAR_load_images(
            config.dataset_path, 
            config.images_dir, 
            config.samples_path,
            config.samples_per_class, 
            config.random_seed
            
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
    
    def load_images(self, config) -> Tuple[List, List[int], Dict[str, str], Any]:
        return GTSRB_load_images(
            config.dataset_path, 
            config.images_dir,)
    
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
