import os
import openai
import json
import random
from typing import List, Dict, Any
import time
import dotenv

class PromptGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the prompt generator with OpenAI API key
        
        Args:
            api_key: OpenAI API key
        """
        self.client = openai.OpenAI(api_key=api_key)
        
        # Dataset configurations
        self.datasets = {
            "GTSRB": {
                "description": "German Traffic Sign Recognition Benchmark with 43 classes of traffic signs",
                "num_classes": 43,
                "class_descriptions": {
                    "0": "Red-bordered circle with '20': Enforces 20 km/h speed limit in sensitive areas",
                    "1": "Circular '30' sign: Urban 30 km/h speed restriction near pedestrians",
                    "2": "Red-circle '50': Standard 50 km/h limit in German city zones",
                    "3": "White circle '60': 60 km/h transitional limit on urban outskirts",
                    "4": "Bold '70' sign: 70 km/h rural road speed allowance",
                    "5": "Black '80' in circle: 80 km/h maximum on non-divided highways",
                    "6": "White circle with diagonal stripe: Ends all previous speed/overtaking restrictions",
                    "7": "Prominent '100' circle: 100 km/h limit on Bundesstraßen highways",
                    "8": "Large '120' sign: Autobahn recommended maximum speed",
                    "9": "Two-car icon: Prohibits all vehicle overtaking",
                    "10": "Truck symbols: Bans trucks (>3.5t) from passing other vehicles",
                    "11": "Upside-down red triangle: Grants right-of-way at next intersection",
                    "12": "Yellow diamond: Marks permanent priority road status",
                    "13": "Red-bordered ▽: Mandatory yield to crossing traffic",
                    "14": "Red octagon 'STOP': Requires full stop before proceeding",
                    "15": "Empty black-bordered circle: Placeholder/deactivated sign function",
                    "16": "Truck in red circle: Prohibits truck entry (>3.5t)",
                    "17": "Solid red circle: Universal no-entry for all vehicles",
                    "18": "❗ in triangle: General hazard warning",
                    "19": "Left-curve arrow: Warns of left-hand bend ahead",
                    "20": "Right-curve arrow: Alerts to upcoming right-hand curve",
                    "21": "S-shaped arrows: Double curve (left-right/right-left) warning",
                    "22": "Zigzag road line: Rough/bumpy surface hazard",
                    "23": "Skidding car icon: Slippery road conditions ahead",
                    "24": "Converging arrows: Lane narrowing/merging zone",
                    "25": "Worker shoveling: Road construction area warning",
                    "26": "Traffic light symbol: Unexpected or low-visibility signals ahead",
                    "27": "Walking figure: Pedestrian crossing/congestion area",
                    "28": "Adult+child silhouettes: School/playground zone warning",
                    "29": "Bicycle icon: Cyclist crossing/high-activity area",
                    "30": "Snowflake over car: Icy/snowy road conditions",
                    "31": "Leaping deer: Wildlife crossing zone (especially deer)",
                    "32": "5-diagonal stripes: Autobahn no-speed-limit section",
                    "33": "Blue ▷: Mandatory right turn",
                    "34": "Blue ◁: Compulsory left turn",
                    "35": "Blue ↑: Straight-ahead only movement",
                    "36": "Y-shaped arrow ↗: Straight or right turn requirement",
                    "37": "Y-shaped arrow ↖: Straight or left turn obligation",
                    "38": "Curving right arrow: Keep right of obstacle/divider",
                    "39": "Curving left arrow: Keep left of obstruction",
                    "40": "Circular arrow: Roundabout approach warning",
                    "41": "Car icon with diagonal: Ends car overtaking ban",
                    "42": "Truck icon with stripe: Terminates truck passing restriction"
                },
                "initial_prompt": "Please classify the object in the given image. It should be only one of these classes: {class_names_str}. Please generate only the class ID number"
            },
            "CIFAR10": {
                "description": "CIFAR-10 dataset with 10 classes of common objects",
                "num_classes": 10,
                "class_descriptions": {
                    "0": "Commercial/military airplanes in flight against sky backgrounds with visible wings, engines, and tail structures.",
                    "1": "Passenger vehicles (sedans/hatchbacks) showing full body, wheels, and windshields in urban/road settings.",
                    "2": "Small perching birds and waterfowl with visible beaks, wings, and legs against natural backgrounds.",
                    "3": "Domestic cats in various poses with triangular ears, whiskers, and fur patterns, often indoors.",
                    "4": "Deer in woodland settings with slender legs, antlers (males), and alert posture near trees/meadows.",
                    "5": "Diverse dog breeds showing snouts, floppy ears, and fur textures in yards/parks with collars.",
                    "6": "Frogs near water with bulbous eyes, crouching poses, and moist green/brown skin on lily pads.",
                    "7": "Horses with muscular builds, flowing manes, and tack (saddles/bridles) in pastures or motion.",
                    "8": "Water vessels (sailboats/cargo ships) on open sea with hulls, masts, and wake patterns.",
                    "9": "Commercial trucks with separate cabs/trailers, multiple axles, and logos on highways/construction sites."
                },
                "initial_prompt": "Please classify the object in the given image. It should be only one of these classes: {class_names_str}. Please generate only the class ID number."
            }
        }

    def generate_prompts(self, dataset_name: str, num_prompts: int = 10, variation_types: List[str] = None) -> List[str]:
        """
        Generate X prompts for a given dataset using GPT-4.1
        
        Args:
            dataset_name: Either "GTSRB" or "CIFAR10"
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
        
        # Generate prompts using GPT-4.1
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
                    model="gpt-4-turbo-preview",  # Using GPT-4 Turbo as GPT-4.1 might not be available
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                
                generated_prompt = response.choices[0].message.content.strip()
                generated_prompts.append(generated_prompt)
                
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
    # Initialize with API key
    
    API_KEY = os.getenv("OPENAI_API_KEY_3")

    generator = PromptGenerator(API_KEY)
    
    # Generate prompts for GTSRB
    print("Generating prompts for GTSRB dataset...")
    gtsrb_prompts = generator.generate_prompts("GTSRB", num_prompts=10)
    # generator.print_prompts(gtsrb_prompts, "GTSRB")
    generator.save_prompts(gtsrb_prompts, "GTSRB")
    
    # Generate prompts for CIFAR10
    print("\nGenerating prompts for CIFAR10 dataset...")
    cifar10_prompts = generator.generate_prompts("CIFAR10", num_prompts=10)
    # generator.print_prompts(cifar10_prompts, "CIFAR10")
    generator.save_prompts(cifar10_prompts, "CIFAR10")
    
    # Custom variation types example
    custom_variations = ["formal_academic", "conversational", "detailed_descriptive", "concise_technical", "step_by_step", "context_aware", "confidence_based", "multi_perspective"]
    custom_prompts = generator.generate_prompts("CIFAR10", num_prompts=9, variation_types=custom_variations)
    generator.print_prompts(custom_prompts, "CIFAR10 (Custom Variations)")


if __name__ == "__main__":
    main()