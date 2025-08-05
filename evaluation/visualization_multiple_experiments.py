import sqlite3
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import sys
import json
import os
pd.options.mode.chained_assignment = None

os.makedirs("results/logs/", exist_ok=True)
sys.stdout = open(f'results/logs/visualization_log_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w')

def load_db(dataset):
    dataset_to_db = {
        'CIFAR10': 'results/cifar10.db',
        'GTSRB': 'results/gtsrb.db'
    }

    # load the db
    conn = sqlite3.connect(dataset_to_db[dataset])
    cursor = conn.cursor()

    return cursor, conn
def preprocess_dataframe(cursor, model):
    # 1. Filter the dataframe by model
    # 2. Change the numerical columns to int
    # 3. Add a column tree_is_correct (int(Class) == int(LLM_output_with_tree_class))
    # 4. Add a column without_tree_is_correct (int(Class) == int(LLM_output_without_tree_class))

    cursor.execute("SELECT * FROM answers WHERE LLM_name = ?", (model,))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[col[0] for col in cursor.description])
    
    # change the Class to int
    df['Class'] = df['Class'].astype(int)
    df['LLM_output_with_tree_class'] = df['LLM_output_with_tree_class'].astype(int)
    df['LLM_output_without_tree_class'] = df['LLM_output_without_tree_class'].astype(int)
    df['Include_memory'] = df['Include_memory'].astype(int)
    df['Include_description'] = df['Include_description'].astype(int)
    df['Include_zero_shot_label'] = df['Include_zero_shot_label'].astype(int)
    # add column tree_is_correct (int(Class) == int(LLM_output_with_tree_class))
    df['tree_is_correct'] = (df['Class'] == df['LLM_output_with_tree_class']).astype(int)
    df['without_tree_is_correct'] = (df['Class'] == df['LLM_output_without_tree_class']).astype(int)
    return df
def return_without_tree_accuracy_perclass(df):
    df['without_tree_is_correct'] = df['without_tree_is_correct'].astype(int)

    # Group by class and calculate mean accuracy
    without_tree_is_correct = df.groupby('Class')['without_tree_is_correct'].mean()

    # Sort by class index
    without_tree_is_correct = without_tree_is_correct.sort_index()

    return without_tree_is_correct
def ret_class_names(dataset):
    # load the class names from the json file
    if dataset == 'CIFAR10':
        dataset = 'cifar-10-python'
    with open(f'Data/{dataset}/class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # convert the class names to a dictionary
    class_names = {int(k): v for k, v in class_names.items()}
    
    return class_names
def create_results_dir(dirs):
    import os
    # create the results directory if it does not exist
    if not os.path.exists('Results'):
        os.makedirs('Results')

    for dataset in dirs['datasets']:
        os.makedirs(f'Results/{dataset}', exist_ok=True)
        # create directories for each model
        for model in dirs['models']:
            os.makedirs(f'Results/{dataset}/{dirs["models"][model]}', exist_ok=True)
            for RunId in range(10):
                os.makedirs(f'Results/{dataset}/{dirs["models"][model]}/RunId_{RunId}', exist_ok=True)

def main():
    models = ['meta-llama/llama-3.2-11b-vision-instruct', 'qwen-vl-max', 'gpt-4o']
    datasets = ['CIFAR10', 'GTSRB']
    number_of_runs = 10
    model_to_name = {
        'meta-llama/llama-3.2-11b-vision-instruct': 'LLAMA',
        'qwen-vl-max': 'Qwen VL Max',
        'gpt-4o': 'GPT-4o'
    }
    
    config = {
    'datasets': {
        'CIFAR10': 'Results/CIFAR10',
        'GTSRB': 'Results/GTSRB'
    },
    'models': {
        'meta-llama/llama-3.2-11b-vision-instruct': 'LLAMA',
        'qwen-vl-max': 'Qwen VL Max',
        'gpt-4o': 'GPT-4o'
    }
    }

    mean_accuracy = {}
    mean_accuracy_classes = {}
    # {"GTSRB": {"LLAMA": {"memory_True": {"temperature_0": {"tree": 20, "without_tree": 10, "Same": 5}
                                                                 
    outperformance = {}
    for dataset in datasets:
        # load the db
        cursor, conn= load_db(dataset)
        mean_accuracy[dataset] = {}
        outperformance[dataset] = {}
        mean_accuracy_classes[dataset] = {}
        for model in models:
            mean_accuracy[dataset][model_to_name[model]] = {}
            mean_accuracy_classes[dataset][model_to_name[model]] = {}
            outperformance[dataset][model_to_name[model]] = {}
            print(f"Processing dataset: {dataset}, model: {model_to_name[model]}")
            # preprocess the dataframe
            df = preprocess_dataframe(cursor, model)
            # fix the Include_memory: 0 and Include_description: 0 and temperature: 0.7 
            df = df[(df['Include_memory'] == 0) & (df['Include_description'] == 0) & (df['temperature'] == 0.7)]
            print(f"Number of rows in the dataframe: {len(df)}")
            if df.empty:
                print(f"No data found for model {model} on dataset {dataset}. Skipping...")
                continue
            
            for RunId in range(number_of_runs):
                print(f"Processing RunId: {RunId}")
                
                df_run = df[df['RunId'] == RunId]
                if df_run.empty:
                    print(f"No data found for RunId {RunId}. Skipping...")
                    continue
                print(f"Number of rows in the dataframe for RunId {RunId}: {len(df_run)}")
                # calculate the accuracy for the model without tree
                without_tree_is_correct = return_without_tree_accuracy_perclass(df_run)
                mean_accuracy[dataset][model_to_name[model]][f'RunId_{RunId}'] = without_tree_is_correct.mean()
                mean_accuracy_classes[dataset][model_to_name[model]][f'RunId_{RunId}'] = without_tree_is_correct.to_dict()
            

    create_results_dir(config)
    # save the results to a file
    with open('Results/mean_accuracy_among_experiments.json', 'w') as f:
        json.dump(mean_accuracy, f, indent=4)

    # save the results to a file
    with open('Results/mean_accuracy_classes_among_experiments.json', 'w') as f:
        json.dump(mean_accuracy_classes, f, indent=4)

if __name__ == "__main__":
    main()