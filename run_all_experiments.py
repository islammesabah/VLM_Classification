import subprocess
import sys
import itertools
from concurrent.futures import as_completed, ProcessPoolExecutor
import multiprocessing as mp

# Redirect stdout to a log file
sys.stdout = open('run_all_experiments.log', 'w')

def run_command(command: str):
    print(f"Starting: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True)
        print(f"Completed: {command} (return code: {result.returncode})")
        if result.returncode != 0:
            print(f"Error in {command}: {result.stderr}")
        return command, result.returncode
    except Exception as e:
        print(f"Exception in {command}: {e}")
        return command, -1

def return_all_variations():
    """Generate all commands to be executed."""
    # Run all the experiments by changing one argument and fixing all the others with the default, so for example when 
    memory_var = [False, True]
    
    # if using description 2 then --disable_tree also (we only need the inference of the zeroshot)
    description_var = [0, 1, 2]  # 0: no description, 1: short description, 2: description for zeroshot
    
    models = ['gpt-4o', 'qwen-vl-max', 'meta-llama/llama-3.2-11b-vision-instruct']
    datasets = ['GTSRB', 'CIFAR-10']
    temperature_var = [0.7, 0]

    all_commands = []
    # Generate memory option variations
    for model, dataset, temperature, memory in itertools.product(models, datasets, temperature_var, memory_var):
        command =  f"python main.py --dataset_name \"{dataset}\" --model \"{model}\" --temperature {temperature}"
        if memory:
            command += " --include_memory"
        all_commands.append(command)

    # Generate description option variations
    for model, dataset, temperature, description in itertools.product(models, datasets, temperature_var, description_var):
        command = f"python main.py --dataset_name \"{dataset}\" --model \"{model}\" --temperature {temperature}"
        if description == 1:
            command += f" --include_description 1"
        elif description == 2:
            command += f" --include_description 2 --disable_tree"
        
        all_commands.append(command)
    # There are 10 prompts variations for the zeroshot inference, (run_id 1-9, as 0 is already covered above)
    for model, dataset, temperature, RunId in itertools.product(models, datasets, temperature_var, range(1,10)):
        command = f"python main.py --dataset_name \"{dataset}\" --model \"{model}\" --temperature {temperature} --RunId {RunId}"
        all_commands.append(command)
    
    return all_commands

def run_all_variations_parallel(max_workers: int = None):

    if max_workers == None or max_workers > mp.cpu_count():
        max_workers = mp.cpu_count()

    commands = return_all_variations()

    print(f"Generated {len(commands)} commands to execute")
    print(f"Running with {max_workers} parallel processes")

    completed_successfully = 0
    failed_commands = []

    with ProcessPoolExecutor(max_workers) as executor:
        # submit the commands
        future_to_command = {executor.submit(run_command, cmd): cmd for cmd in commands}

        for future in as_completed(future_to_command):
            command, return_code = future.result()
            if return_code == 0:
                completed_successfully += 1
            else:
                failed_commands.append(command)
    
    # Print summary
    print(f"\n=== Execution Summary ===")
    print(f"Total commands: {len(commands)}")
    print(f"Completed successfully: {completed_successfully}")
    print(f"Failed: {len(failed_commands)}")
    
    if failed_commands:
        print(f"\nFailed commands:")
        for cmd in failed_commands:
            print(f"  - {cmd}")
    
if __name__ == "__main__":
    run_all_variations_parallel()
