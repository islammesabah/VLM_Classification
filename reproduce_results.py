import subprocess
import os
import itertools

# --- Configuration ---
# This section defines the parameters for the experiments.
# You can customize these lists to run a different set of tests.

# List of models to test. These must be available through the OpenRouter API.
MODELS = [
    "openai/gpt-4o",
    "google/gemini-pro-vision",
    "anthropic/claude-3-sonnet-20240229",
    "qwen/qwen-vl-max"
]

# List of datasets to use.
DATASETS = [
    "CIFAR-10",
    "GTSRB"
]

# Number of samples to evaluate from each dataset.
NUM_SAMPLES = 100

# Temperature settings for model inference.
TEMPERATURES = [
    0.0,
    0.7
]

# Inference strategies:
# True: Use the decision tree-based classification.
# False: Use standard zero-shot classification.
USE_TREE_OPTIONS = [
    True,
    False
]

def run_experiments():
    """
    Executes the VLM classification experiments based on the configuration above.
    """
    # Ensure the output directory exists.
    os.makedirs("Results", exist_ok=True)

    # Create all possible combinations of the parameters.
    experiment_configs = list(itertools.product(MODELS, DATASETS, TEMPERATURES, USE_TREE_OPTIONS))
    total_experiments = len(experiment_configs)

    print("Starting VLM classification experiment reproduction...")
    print(f"Total experiments to run: {total_experiments}")
    print("--------------------------------------------------")

    # Iterate through each configuration and run the inference script.
    for i, (model, dataset, temp, use_tree) in enumerate(experiment_configs, 1):
        print(f"Running experiment {i} of {total_experiments}:")
        print(f"  - Model:    {model}")
        print(f"  - Dataset:  {dataset}")
        print(f"  - Temp:     {temp}")
        print(f"  - Use Tree: {use_tree}")
        print("")

        # Construct the command to call the main inference script.
        command = [
            "python3",
            "src/clean_inference_code.py",
            "--model_name", model,
            "--dataset_name", dataset,
            "--num_samples", str(NUM_SAMPLES),
            "--temperature", str(temp),
        ]

        # Append the appropriate flag for using the decision tree.
        if use_tree:
            command.append("--use_tree")
        else:
            command.append("--no-use_tree")

        try:
            # Execute the command. The `check=True` flag will raise an error
            # if the script returns a non-zero exit code.
            subprocess.run(command, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during experiment {i}: {e}")
            print("Continuing with the next experiment...")
        except FileNotFoundError:
            print("Error: 'python3' command not found.")
            print("Please ensure Python 3 is installed and in your PATH.")
            return

        print("--------------------------------------------------")

    print("All experiments completed successfully.")
    print("Results have been saved in the 'Results' directory.")

if __name__ == "__main__":
    run_experiments()
