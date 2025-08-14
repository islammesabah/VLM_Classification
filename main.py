"""
Implementation of image classification inference with VLM.
Supports multiple datasets and models with configurable parameters.
"""

import os
import json
import sqlite3
import argparse
import base64
import re
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv
from pprint import pprint
from results.initiate_db import create_table


# Import your existing modules
from models.vlm_model import ModelClient
from datasets.data_loader import DatasetLoader, CIFARLoader, GTSRBLoader

# program interface
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run VLM inference on traffic sign classification datasets"
    )
    
    # experiment config
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='GTSRB',
        choices=['GTSRB', 'CIFAR-10'],
        help='Dataset to use for inference'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['gpt-4o', 'qwen-vl-max', 'meta-llama/llama-3.2-11b-vision-instruct'],
        help='Model to use for inference'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file'
    )

    # run config
    parser.add_argument(
        '--disable_tree',
        action='store_true',
        help='Disable tree-based inference'
    )
    
    parser.add_argument(
        '--include_memory',
        action='store_true',
        help='Include history chat (questions and answers) in the LLM input of the tree inference'
    )

    parser.add_argument(
        '--include_description',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Include description of the Class and Image in the LLM input'
    )
    
    parser.add_argument(
        '--include_zero_shot_label',
        action='store_true',
        help='Include zero-shot label in the LLM input for the tree inference'
    )

    parser.add_argument(
        '--RunId',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help='Run ID to choose between prompts for zero-shot inference'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for sampling'
    )

    return parser

 
@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""
    dataset_name: str
    model: str
    dataset_path: str
    images_dir: str
    database_name: str
    samples_path: str
    samples_per_class: int = 100
    random_seed: int = 42
    max_tokens: int = 100
    zero_shot_prompts: List[str] = field(default_factory=list)
    sleep_time: float = 4.0
    description_path: Optional[str] = None
    temperature: float = 0.7
    
    @classmethod
    def from_file(cls, config_path: str, dataset_name: str, model: str, temperature: float = 0.7) -> 'InferenceConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            dataset_config = config[dataset_name]
            # retrieve the list of prompts for the dataset
            with open(f'prompts/{dataset_name}/zs_prompts.json', 'r') as f:
                dataset_config['zero_shot_prompts'] = json.load(f)
                dataset_config['zero_shot_prompts'] = dataset_config['zero_shot_prompts']['prompts']

            return cls(
                dataset_name=dataset_name,
                model=model,
                dataset_path=dataset_config['Dataset_path'],
                images_dir=dataset_config['images_dir'],
                database_name=dataset_config['database_name'],
                samples_per_class=dataset_config.get('samples_per_class', 100),
                random_seed=dataset_config.get('random_seed', 42),
                max_tokens=dataset_config.get('max_tokens', 100),
                zero_shot_prompts=dataset_config.get('zero_shot_prompts', []),
                sleep_time=dataset_config.get('sleep_time', 4.0),
                samples_path=dataset_config.get('samples_path', None),
                description_path=dataset_config.get('description_path', None),
                temperature=temperature
                )
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Error loading configuration: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'dataset_name': self.dataset_name,
            'model': self.model,
            'dataset_path': self.dataset_path,
            'images_dir': self.images_dir,
            'database_name': self.database_name,
            'samples_per_class': self.samples_per_class,
            'random_seed': self.random_seed,
            'max_tokens': self.max_tokens,
            'zero_shot_prompts': self.zero_shot_prompts,
            'sleep_time': self.sleep_time,
            'samples_path': self.samples_path,
            'description_path': self.description_path,
            'temperature': self.temperature
        }

@dataclass
class InferenceResult:
    """Result of an inference operation."""
    response: str
    predicted_class: int
    confidence: Optional[float] = None
    tree_path: Optional[List[Dict[str, str]]] = None


class DatabaseManager:
    """Handles database operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def image_exists(self, image_path: str, model_name: str, include_memory: bool, include_description: int, include_zero_shot_label: bool, temperature: float, RunId: int) -> bool:
        """Check if image result already exists in database."""
        image_path = image_path.replace("\\", "/")
        if model_name.__contains__('gpt'):
            model_name = 'gpt-4o'
        self.cursor.execute(
            'SELECT * FROM answers WHERE Image_path = ? AND LLM_name = ? AND Include_memory = ? AND Include_description = ? AND Include_zero_shot_label = ? AND temperature = ? AND RunId = ?',
            (image_path, model_name, include_memory, include_description, include_zero_shot_label, temperature, RunId)
        )
        return self.cursor.fetchone() is not None
    
    def insert_result(self, class_label: int, sequence: str, image_path: str, 
                     model_name: str, tree_result: Optional[str], 
                     zero_shot_result: str, tree_class: int, zero_shot_class: int, include_memory: bool, include_description: int, include_zero_shot_label: bool, temperature: float = 0.7, RunId: int = 0):
        """Insert inference result into database."""
        image_path = image_path.replace("\\", "/")
        if model_name.__contains__('gpt'):
            model_name = 'gpt-4o'

        sql = '''
        INSERT INTO answers (Class, Sequence, Image_path, LLM_name, 
                           LLM_output_with_tree, LLM_output_without_tree, 
                           LLM_output_with_tree_class, LLM_output_without_tree_class,
                           Include_memory, Include_description, Include_zero_shot_label, temperature, RunId)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(sql, (
            class_label, sequence, image_path, model_name,
            tree_result, zero_shot_result, tree_class, zero_shot_class,
            include_memory, include_description, include_zero_shot_label, temperature, RunId
        ))
        self.conn.commit()
    
    def update_row_tree(self, image_path: str, model_name: str, tree_result: Optional[str], 
                     tree_class: int, include_memory: bool, include_description: int, include_zero_shot_label: bool, temperature: float = 0.7):
        image_path = image_path.replace("\\", "/")
        if model_name.__contains__('gpt'):
            model_name = 'gpt-4o'

        sql = f'''
        UPDATE answers
        SET LLM_output_with_tree = ?
        , LLM_output_with_tree_class = ?
        WHERE Image_path = ? AND LLM_name = ? AND Include_memory = ? AND Include_description = ? AND Include_zero_shot_label = ? AND temperature = ?
        '''
        self.cursor.execute(sql, (
            tree_result, tree_class, image_path, model_name, include_memory, include_description, include_zero_shot_label, temperature
        ))
        self.conn.commit()
        
    def close(self):
        """Close database connection."""
        self.conn.close()


# read the list from openrouter_providers.json
with open('models/openrouter_providers.json', 'r') as f:
    openrouter_providers = json.load(f)

class VLMInference:
    """Main inference engine for Vision Language Models."""
    
    def __init__(self, config: InferenceConfig, include_memory: bool = False,
                 include_description: int = 0, include_zero_shot_label: bool = False):
        self.config = config
        self.model_client = ModelClient(config.model)
        self.dataset_loader = self._create_dataset_loader()
        self.db_manager = DatabaseManager(config.database_name)
        self.include_memory = include_memory
        self.include_description = include_description
        self.include_zero_shot_label = include_zero_shot_label
        
    
    def _create_dataset_loader(self) -> DatasetLoader:
        """Create appropriate dataset loader."""
        loaders = {
            'CIFAR-10': CIFARLoader(),
            'GTSRB': GTSRBLoader()
        }
        
        if self.config.dataset_name not in loaders:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
        return loaders[self.config.dataset_name]
    
    def _encode_image(self, image: Union[np.ndarray, Image.Image]) -> str:
        """Encode image to base64."""
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
        
        # Convert to RGB if necessary
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def _extract_class_from_response(self, response: Union[str, None], class_names: Dict[str, str]) -> int:
        """Extract class number from model response."""
        if not response:
            print("Empty response received from model.")
            return -1
        match = re.search(r'^\D*(\d+)', response)
        if match:
            class_id = int(match.group(1))
            return class_id if str(class_id) in class_names else -1
        return -1
    
    def zero_shot_inference(self, base64_image: str, class_names: Dict[str, str], config: dict[str, Any], include_description: int, short_description: str="", RunId: int=0) -> InferenceResult:
        """Perform zero-shot inference."""
        class_names_str = ", ".join([f"{i}: {class_names[str(i)]}" for i in range(len(class_names))])
        messages = [
                {
                    "role": "user",
                    "content": []
                }
        ]
        
        if include_description == 2:
            with open("prompts/describe_image.txt", "r") as file:
                prompt_description = file.read()

            messages_descr = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_description},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
            payload = {
                "model": self.model_client.model_name,
                "messages": messages_descr,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            if self.model_client.model_name.startswith('meta-llama/'):
                payload['extra_body'] = {
                    "provider": {
                        "order": openrouter_providers
                    },
                    "allow_fallbacks": False
                }
            completion = self.model_client.client.chat.completions.create(
                **payload
            )
            image_description = completion.choices[0].message.content

            
            # Include the image description in the messages
            messages[0]["content"].append({
                "type": "text",
                "text": f"Image description:{image_description}"
            })

            # add the short description of the classes
            messages[0]["content"].append({
                "type": "text",
                "text": f"Description of the classes: {short_description}"
            })

        prompt = self.dataset_loader.get_zero_shot_prompt(config, RunId).replace("{class_names_str}", class_names_str)
        print(f"\n >> Zero-shot inference prompt:")
        print('-'*50)
        print(prompt)
        print('-'*50)

        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })
        # print(f"Zero-shot inference message: {messages[0]['content']}")
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

        # print(f"Messages for zero-shot inference: {messages}")
        payload = {
            "model": self.model_client.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        if self.model_client.model_name.startswith('meta-llama/'):
            payload['extra_body'] = {
                "provider": {
                    "order": openrouter_providers
                },
                "allow_fallbacks": False
            }
        completion = self.model_client.client.chat.completions.create(
            **payload
        )
        # print(f"completion: {completion}")
        response = completion.choices[0].message.content
        predicted_class = self._extract_class_from_response(response, class_names)
        
        print(f"Zero-shot inference response: {response}, Class: {predicted_class}")
        if not response:
            response = ""
        # time.sleep(random.uniform(2, 4))
        return InferenceResult(response=response, predicted_class=predicted_class)

    def tree_inference(self, base64_image: str, include_memory: bool, include_description: int, class_names: Dict[str, str], long_description: str="", short_description: str="") -> InferenceResult:
        """Perform tree-based inference."""
        tree = self.dataset_loader.build_tree()
        tree_path = []

        # If include description for the tree of the classes and the image.
        if include_description == 1:
            with open("prompts/describe_image.txt", "r") as file:
                prompt = file.read()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
            payload = {
                "model": self.model_client.model_name,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            if self.model_client.model_name.startswith('meta-llama/'):
                payload['extra_body'] = {
                    "provider": {
                        "order": openrouter_providers
                    },
                    "allow_fallbacks": False
                }
            completion = self.model_client.client.chat.completions.create(
                **payload
            )
            image_description = completion.choices[0].message.content
            # print(f"Image description: {image_description}")
        while tree.question:
            print(f"Message: {tree.message}")
            prompt = f"{tree.question} Choose one of these answers: {tree.answers}."
            print(f"Prompt: {prompt}")
            
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            if include_memory:
                # Include previous questions and answers in the messages first
                messages[0]["content"].append({"type": "text", "text": "Previous questions and answers:"})
                for item in tree_path:
                    messages[0]["content"].append(
                        {"type": "text", "text": f"Q: {item['question']} A: {item['answer']}"}
                    )
            
            if include_description == 1:
                # Include the image description in the messages
                messages[0]["content"].append({
                    "type": "text",
                    "text": f"Image description: {image_description}"
                })

                # add the short description of the classes
                messages[0]["content"].append({
                    "type": "text",
                    "text": f"Description of the classes: {short_description}"
                })
            
            # Then add the current question
            messages[0]["content"].append({
                "type": "text", "text": prompt
            })
            
            # print(f"Messages for tree inference: {messages}")
            # Add the image
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

            # Sort answers by length (longest first) to avoid substring matching issues
            sorted_answers = sorted(tree.answers, key=len, reverse=True)
            regex_pattern = "(?i)(" + "|".join([f"\\b{ans}\\b" for ans in sorted_answers]) + ")"
            
            payload = {
                "model": self.model_client.model_name,
                "messages": messages,
                "max_tokens": self.config.max_tokens
            }
            if self.model_client.model_name.startswith('meta-llama/'):
                payload['extra_body'] = {
                    "provider": {
                        "order": openrouter_providers
                    },
                    "allow_fallbacks": False
                }
            completion = self.model_client.client.chat.completions.create(
                **payload
            )
            # print(f"Messages sent to the model: {messages}")
            response = completion.choices[0].message.content
            
            
            print("=" * 50)
            print(f"Response: {response}")
            
            if not response:
                print("Empty response received from model.")
                tree_path.append({
                    "question": tree.question,
                    "answer": response
                })
                return InferenceResult(response=json.dumps(tree_path), predicted_class=-1, tree_path=tree_path)
            
            match = re.search(regex_pattern.lower(), response.lower())
            if match:
                answer = match.group(1)
                tree_path.append({
                    "question": tree.question,
                    "answer": answer
                })
                print(f"{tree.question} : {answer}")
                tree = tree.children[answer]
                
            else:
                print("Invalid answer - breaking tree traversal")
                tree_path.append({
                    "question": tree.question,
                    "answer": response.strip()
                })
                break
            
            print("=" * 50)
            time.sleep(1)
        
        if not tree.question:
            print(f"Final: {tree.message} ({tree.possible_classes})")
            predicted_class = tree.possible_classes[0]
        else:
            print("Could not complete tree traversal")
            predicted_class = -1
        
        return InferenceResult(
            response=json.dumps(tree_path),
            predicted_class=predicted_class,
            tree_path=tree_path
        )

    def get_descriptions(self) -> Tuple[str, str]:
        """Get class descriptions from the dataset."""
        if self.config.description_path:
            with open(self.config.description_path, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)
            str_descriptions = json.dumps(descriptions.get('long_description', ""))
            str_short_descriptions = json.dumps(descriptions.get('short_description', ""))
            return str_descriptions, str_short_descriptions
        else:
            print("No description path provided in configuration.")
            return "", ""
    
    def run_inference(self, enable_tree: bool = False, include_memory: bool = False, include_description: int = 0, include_zero_shot_label: bool = False, run_id: int = 0):
        """Run inference on all images."""
        
        # Load images
        images, labels, class_names, additional_info = self.dataset_loader.load_images(self.config)
        print(f"\nLoaded {len(images)} images from dataset '{self.config.dataset_name}'")
        
        if include_description != 0:
            # Get class descriptions
            long_description, short_description = self.get_descriptions()
        else:
            long_description, short_description = "", ""
        
        # Process each image
        for image_idx in tqdm(range(len(images[:1])), desc="\nProcessing images"):
            # Extract image information
            image_path, sequence = self.dataset_loader.extract_image_info(image_idx, additional_info)
            
            # Check if already processed
            if self.db_manager.image_exists(image_path, self.config.model, include_memory, include_description, include_zero_shot_label, self.config.temperature, run_id):
                print(f"\nImage {image_path} already processed for {self.config.model}. Skipping.")
                continue
            
            # Encode image
            base64_image = self._encode_image(images[image_idx])
            
            # Zero-shot inference
            zero_shot_result = self.zero_shot_inference(base64_image, class_names, self.config.to_dict(), include_description, short_description, run_id)
            
            # Tree inference (if enabled)
            tree_result = None
            tree_class = -1
            if enable_tree:
                tree_result = self.tree_inference(base64_image, include_memory, include_description, class_names, long_description, short_description)
                tree_class = tree_result.predicted_class


            # Save results
            self.db_manager.insert_result(
                class_label=labels[image_idx],
                sequence=sequence,
                image_path=image_path,
                model_name=self.config.model,
                tree_result=tree_result.response if tree_result else None,
                zero_shot_result=zero_shot_result.response,
                tree_class=tree_class,
                zero_shot_class=zero_shot_result.predicted_class,
                include_memory=include_memory,
                include_description=include_description,
                include_zero_shot_label=include_zero_shot_label,
                temperature=self.config.temperature,
                RunId=run_id
            )
            
            # Rate limiting
            time.sleep(2)
        
        print("Inference completed!")
    
    def cleanup(self):
        """Clean up resources."""
        self.db_manager.close()


def main():
    """Main execution function."""

    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = InferenceConfig.from_file(args.config, args.dataset_name, args.model, args.temperature)
        
        # Create database if it doesn't exist
        if not os.path.exists(config.database_name):
            create_table(config.database_name)
            print(f"Database '{config.database_name}' created successfully.")
        else:
            print(f"Database '{config.database_name}' already exists. Skipping creation.")

        # Create inference engine
        inference_engine = VLMInference(config)
        
        print("\nLoaded configuration: ")
        pprint(config.to_dict())
        print("\nArguments received: ")
        pprint(vars(args))

        # Run inference
        inference_engine.run_inference(enable_tree=not args.disable_tree, include_memory=args.include_memory, include_description=args.include_description, include_zero_shot_label=args.include_zero_shot_label, run_id=args.RunId)

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

    finally:
        # Clean up
        if 'inference_engine' in locals():
            inference_engine.cleanup()


if __name__ == "__main__":
    main()
