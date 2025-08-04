VLM Classification using a Decision Tree
This project explores a novel approach to image classification using Vision-Language Models (VLMs). Instead of traditional zero-shot classification, it employs a dynamically generated decision tree to guide the VLM's reasoning process, improving accuracy and providing interpretability.

1. Setup and Installation
Follow these steps to set up your local environment.

Prerequisites
Python 3.8 or higher

Installation Steps
Clone the Repository (if you haven't already):

git clone <repository-url>
cd VLM_Classification

Create a Virtual Environment (recommended):

python3 -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate

Install Dependencies:
Install all required packages using the requirements.txt file.

pip install -r requirements.txt

2. Configuration
Before running the experiments, you must provide your OpenRouter API key.

Open the config.json file.

Replace "YOUR_API_KEY" with your actual key.

{
    "OPENROUTER_API_KEY": "YOUR_API_KEY"
}

3. Running Experiments
To reproduce the results, run the main experiment script from your terminal:

python3 reproduce_results.py

This script will automatically iterate through all the predefined models, datasets, and configurations, saving the output of each run.

Customizing Experiments
You can easily customize the experiments by editing the configuration lists at the top of the reproduce_results.py script.

MODELS: A list of model identifiers from OpenRouter.

DATASETS: The datasets to test (CIFAR-10, GTSRB).

NUM_SAMPLES: The number of images to process per run.

TEMPERATURES: A list of temperature values for inference.

USE_TREE_OPTIONS: A list containing True and/or False to test with and without the decision tree.

4. Output Structure
All results are saved in the Results/ directory. For each experiment, two files are generated:

Raw Results: results_{model}_{dataset}_{samples}_{temp}_{strategy}.json

Contains the detailed inference output for each image, including the model's responses and the decision path taken.

Evaluation Summary: evaluation_{model}_{dataset}_{samples}_{temp}_{strategy}.json

Contains the aggregated metrics, including overall accuracy and class-wise performance.

The strategy in the filename will be either tree or no_tree depending on the method used.