import os
import openai
import json
import random
from typing import List, Dict, Any
import time
import dotenv

class PromptGenerator:
    def __init__(self, api_key: str, config_dir: str = "prompts"):
        """
        Initialize the prompt generator with OpenAI API key
        
        Args:
            api_key: OpenAI API key
            config_dir: Directory containing dataset configuration JSON files
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
            )
        self.config_dir = config_dir
        
        # Load dataset configurations from files
        self.datasets = self._load_dataset_configs()

    def _load_dataset_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load dataset configurations from JSON files in config directory"""
        datasets = {}
        
        # Load all JSON files from config directory
        # Path: prompts/{DATASET_NAME}/classes_description.json

        # list all the directories in config_dir
        for dataset_name in os.listdir(self.config_dir):
            dataset_path = os.path.join(self.config_dir, dataset_name)
            if os.path.isdir(dataset_path):
                config_file = os.path.join(dataset_path, "classes_description.json")
                print(config_file)
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                datasets[dataset_name] = {
                    "description": "",
                    "num_classes": len(config),
                    "initial_prompt": config['initial_prompt'],
                    "class_descriptions": config['short_description']
                }
        return datasets

    def generate_prompts(self, dataset_name: str, num_prompts: int = 10, variation_types: List[str] = None) -> List[str]:
        """
        Generate X prompts for a given dataset using GPT-4
        
        Args:
            dataset_name: Dataset name (e.g., "GTSRB", "CIFAR-10", "MNIST", "MSCOCO")
            num_prompts: Number of prompts to generate
            variation_types: Types of variations to generate (optional)
            
        Returns:
            List of generated prompts
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(self.datasets.keys())}")
        
        dataset = self.datasets[dataset_name]
        
        if variation_types is None:
            variation_types = [
                "formal_academic",
                "conversational",
                "detailed_descriptive",
                "concise_technical",
                "step_by_step",
                "context_aware",
                "confidence_based",
                "multi_perspective"
            ]
        
        generated_prompts = []
        
        # Create class names string for the template
        class_names = list(range(dataset["num_classes"]))
        class_names_str = ", ".join([f"{i}" for i in class_names])
        
        # Fill in the initial prompt template
        initial_prompt = dataset["initial_prompt"].format(class_names_str=class_names_str)
        
        # Generate prompts using GPT-5
        for i in range(num_prompts):
            variation_type = variation_types[i % len(variation_types)]
            
            system_prompt = f"""You are an expert in creating prompts for image classification tasks. 
            You need to generate a variation of the given initial prompt for the {dataset_name} dataset.
            
            Dataset: {dataset["description"]}
            Number of classes: {dataset["num_classes"]}
            
            Class descriptions:
            {json.dumps(dataset["class_descriptions"], indent=2)}
            
            Variation type: {variation_type}
            
            Requirements:
            1. The prompt must ask for classification into one of the {dataset["num_classes"]} classes
            2. The output should be only the class ID number
            3. Make the prompt variation match the specified variation type
            4. Keep the core classification task intact
            5. Use {{class_names_str}} as placeholder for class names list
            
            Generate only the prompt text, nothing else."""
            
            user_prompt = f"""Initial prompt: "{initial_prompt}"
            
            Create a {variation_type} variation of this prompt that maintains the same classification objective but changes the style, tone, or approach."""
            
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-4o-2024-11-20",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7
                )
                
                generated_prompt = response.choices[0].message.content.strip()
                generated_prompts.append(generated_prompt)
                print("generated prompt:", generated_prompt)
                print(response)
                print(f"Generated prompt {i+1}/{num_prompts} ({variation_type})")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating prompt {i+1}: {str(e)}")
                # Fallback to a simple variation
                fallback_prompt = self._create_fallback_prompt(initial_prompt, variation_type)
                generated_prompts.append(fallback_prompt)
        
        return generated_prompts

    def _create_fallback_prompt(self, initial_prompt: str, variation_type: str) -> str:
        """Create a fallback prompt if API call fails"""
        variations = {
            "formal_academic": f"In this image classification task, {initial_prompt.lower()}",
            "conversational": f"Hey! Can you look at this image and {initial_prompt.lower()}",
            "detailed_descriptive": f"Analyze the visual features in this image carefully and {initial_prompt.lower()}",
            "concise_technical": f"Classify: {initial_prompt}",
            "step_by_step": f"Step 1: Examine the image. Step 2: {initial_prompt}",
            "context_aware": f"Given the context of this classification system, {initial_prompt.lower()}",
            "confidence_based": f"With high confidence, {initial_prompt.lower()}",
            "multi_perspective": f"From multiple visual angles, {initial_prompt.lower()}"
        }
        
        return variations.get(variation_type, initial_prompt)

    def save_prompts(self, prompts: List[str], dataset_name: str, filename: str = None):
        """Save generated prompts to a JSON file"""
        if filename is None:
            filename = f"{dataset_name.lower()}_prompts.json"
        
        data = {
            "dataset": dataset_name,
            "num_prompts": len(prompts),
            "prompts": prompts,
            "class_descriptions": self.datasets[dataset_name]["class_descriptions"]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(prompts)} prompts to {filename}")

    def print_prompts(self, prompts: List[str], dataset_name: str):
        """Print generated prompts in a formatted way"""
        print(f"\n=== Generated Prompts for {dataset_name} ===\n")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"Prompt {i}:")
            print(f"{prompt}")
            print("-" * 80)


# Example usage
def main():
    dotenv.load_dotenv()
    
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Initialize with API key and config directory
    generator = PromptGenerator(API_KEY, config_dir="./")
    
    # Generate prompts for all datasets
    datasets = ["MSCOCO", "CIFAR-10", "GTSRB", "MNIST"]
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Generating prompts for {dataset_name} dataset...")
        print(f"{'='*80}")
        
        try:
            prompts = generator.generate_prompts(dataset_name, num_prompts=10)
            generator.save_prompts(prompts, dataset_name)
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")

if __name__ == "__main__":
    main()