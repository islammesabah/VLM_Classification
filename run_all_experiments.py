import subprocess
import sys
import itertools
from concurrent.futures import as_completed, ProcessPoolExecutor
import multiprocessing as mp
import os
import argparse
import tempfile
from typing import Optional
import time

# sys.stdout = open('run_all_experiments.log', 'w', buffering=1)

def generate_slurm_script(command: str, slurm_config: dict) -> str:
    """
    Generate a SLURM batch script.
    
    Args:
        command: The Python command to execute
        slurm_config: Dictionary with SLURM parameters (nodes, ntasks, time, mem, job_name)
    
    Returns:
        SLURM script as a string
    """
    script = "#!/bin/bash\n"
    script += f"#SBATCH --output={slurm_config.get('output', 'slurm-%j.out')}\n"
    script += f"#SBATCH --error={slurm_config.get('error', 'slurm-%j.err')}\n"
    script += f"#SBATCH --nodes={slurm_config.get('nodes', 1)}\n"
    script += f"#SBATCH --ntasks={slurm_config.get('ntasks', 1)}\n"
    script += f"#SBATCH --time={slurm_config.get('time', '0-08:00:00')}\n"
    script += f"#SBATCH --job-name={slurm_config.get('job_name', 'inference')}\n"
    script += f"#SBATCH --mem={slurm_config.get('mem', 8000)}\n"
    
    # Add any additional SLURM parameters
    if 'partition' in slurm_config:
        script += f"#SBATCH --partition={slurm_config['partition']}\n"
    if 'cpus_per_task' in slurm_config:
        script += f"#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}\n"
    if 'gres' in slurm_config:
        script += f"#SBATCH --gres={slurm_config['gres']}\n"
    
    script += "\n"
    script += "# Activate conda environment if needed\n"
    script += "# source activate treeinference\n"
    script += "\n"
    script += f"{command}\n"
    
    return script


def submit_slurm_job(command: str, slurm_config: dict) -> tuple:
    """
    Submit a job to SLURM cluster.
    
    Args:
        command: The Python command to execute
        slurm_config: Dictionary with SLURM parameters
    
    Returns:
        Tuple of (command, return_code, job_id)
    """
    try:
        # Generate SLURM script
        script_content = generate_slurm_script(command, slurm_config)
        
        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Submit job using sbatch
        result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
        
        # Clean up temporary script
        os.remove(script_path)
        
        if result.returncode == 0:
            # Extract job ID from output (format: "Submitted batch job 123456")
            job_id = result.stdout.strip().split()[-1]
            print(f"Submitted SLURM job: {job_id} for command: {command}")
            return command, result.returncode, job_id
        else:
            print(f"Failed to submit SLURM job for command: {command}")
            print(f"Error: {result.stderr}")
            return command, result.returncode, None
    
    except Exception as e:
        print(f"Exception submitting SLURM job for {command}: {e}")
        return command, -1, None


def run_command(command: str):
    print(f"Starting: {command}")
    try:
        # Run without capturing output so we can see it in real-time
        result = subprocess.run(command, shell=True, text=True)
        print(f"Completed: {command} (return code: {result.returncode})")
        return command, result.returncode
    except Exception as e:
        print(f"Exception in {command}: {e}")
        return command, -1

def return_all_variations():
    """Generate all commands to be executed."""
    # Run all the experiments by changing one argument and fixing all the others with the default, so for example when 
    memory_var = [False]
    
    # if using description 2 then --disable_tree also (we only need the inference of the zeroshot)
    description_var = [0, 1, 2]  # 0: no description, 1: short description, 2: description for zeroshot
    
    models = ['gpt-4o', 'qwen-vl-max', 'meta-llama/llama-3.2-11b-vision-instruct']
    datasets = ['MNIST', 'MSCOCO']
    temperature_var = [0.7]

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
    # for model, dataset, temperature, RunId in itertools.product(models, datasets, temperature_var, range(1,10)):
    #     command = f"python main.py --dataset_name \"{dataset}\" --model \"{model}\" --temperature {temperature} --RunId {RunId} --disable_tree"
    #     all_commands.append(command)
    
    return all_commands

def run_all_variations_parallel(max_workers: int = None, use_slurm: bool = False, slurm_config: Optional[dict] = None):

    if use_slurm:
        # When using SLURM, submit all jobs and return
        commands = return_all_variations()
        

        print(f"Generated {len(commands)} commands to submit to SLURM")
        
        if slurm_config is None:
            slurm_config = {}
        
        submitted_jobs = []
        failed_submissions = []
        
        for i, cmd in enumerate(commands):
            command, return_code, job_id = submit_slurm_job(cmd, slurm_config)
            if return_code == 0 and job_id:
                submitted_jobs.append((job_id, cmd))
            else:
                failed_submissions.append(cmd)
            
            # Add 5-second delay between submissions (except after the last one)
            if i < len(commands) - 1:
                time.sleep(5)
        
        # Print summary
        print(f"\n=== SLURM Submission Summary ===")
        print(f"Total commands: {len(commands)}")
        print(f"Successfully submitted: {len(submitted_jobs)}")
        print(f"Failed submissions: {len(failed_submissions)}")
        
        if submitted_jobs:
            print(f"\nSubmitted job IDs:")
            for job_id, cmd in submitted_jobs:
                print(f"  Job {job_id}: {cmd}")
        
        if failed_submissions:
            print(f"\nFailed submissions:")
            for cmd in failed_submissions:
                print(f"  - {cmd}")
        
        return

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
    parser = argparse.ArgumentParser(description="Run all VLM classification experiments")
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help='Maximum number of parallel processes (default: CPU count)'
    )
    parser.add_argument(
        '--use_slurm',
        action='store_true',
        default=False,
        help='Submit jobs to SLURM cluster instead of running locally'
    )
    parser.add_argument(
        '--slurm_nodes',
        type=int,
        default=1,
        help='Number of SLURM nodes'
    )
    parser.add_argument(
        '--slurm_ntasks',
        type=int,
        default=1,
        help='Number of SLURM tasks'
    )
    parser.add_argument(
        '--slurm_time',
        type=str,
        default='0-08:00:00',
        help='SLURM time limit (format: D-HH:MM:SS)'
    )
    parser.add_argument(
        '--slurm_mem',
        type=int,
        default=8000,
        help='SLURM memory in MB'
    )
    parser.add_argument(
        '--slurm_job_name',
        type=str,
        default='vlm_inference',
        help='SLURM job name'
    )
    parser.add_argument(
        '--slurm_partition',
        type=str,
        default=None,
        help='SLURM partition/queue name'
    )
    parser.add_argument(
        '--slurm_cpus_per_task',
        type=int,
        default=None,
        help='SLURM CPUs per task'
    )
    parser.add_argument(
        '--slurm_gres',
        type=str,
        default=None,
        help='SLURM generic resources (e.g., gpu:1)'
    )
    
    args = parser.parse_args()
    
    # Build SLURM config if using SLURM
    slurm_config = None
    if args.use_slurm:
        slurm_config = {
            'nodes': args.slurm_nodes,
            'ntasks': args.slurm_ntasks,
            'time': args.slurm_time,
            'mem': args.slurm_mem,
            'job_name': args.slurm_job_name,
            'output': f'{args.slurm_job_name}-%j.out',
            'error': f'{args.slurm_job_name}-%j.err',
        }
        if args.slurm_partition:
            slurm_config['partition'] = args.slurm_partition
        if args.slurm_cpus_per_task:
            slurm_config['cpus_per_task'] = args.slurm_cpus_per_task
        if args.slurm_gres:
            slurm_config['gres'] = args.slurm_gres
    
    run_all_variations_parallel(max_workers=args.max_workers, use_slurm=args.use_slurm, slurm_config=slurm_config)
