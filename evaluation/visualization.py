import sqlite3
import os

import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import sys
import json
pd.options.mode.chained_assignment = None
# write stdout to a logging file with timestamp

os.makedirs("results/logs/", exist_ok=True)
sys.stdout = open(f'results/logs/visualization_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w')

def load_db(dataset):
    dataset_to_db = {
        'CIFAR10': 'results/cifar10.db',
        'GTSRB': 'results/gtsrb.db'
    }

    # load the db
    conn = sqlite3.connect(dataset_to_db[dataset])
    cursor = conn.cursor()

    return cursor, conn

def preprocess_dataframe(cursor, model, RunId):
    # 1. Filter the dataframe by model
    # 2. Change the numerical columns to int
    # 3. Add a column tree_is_correct (int(Class) == int(LLM_output_with_tree_class))
    # 4. Add a column without_tree_is_correct (int(Class) == int(LLM_output_without_tree_class))

    cursor.execute("SELECT * FROM answers WHERE LLM_name = ? AND RunId = ?", (model,RunId))
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

def plot_comparison_accuracy(df, dir):
    df['tree_is_correct'] = df['tree_is_correct'].astype(int)
    df['without_tree_is_correct'] = df['without_tree_is_correct'].astype(int)

    # Group by class and calculate mean accuracy
    tree_is_correct = df.groupby('Class')['tree_is_correct'].mean()
    without_tree_is_correct = df.groupby('Class')['without_tree_is_correct'].mean()

    # Sort by class index
    tree_is_correct = tree_is_correct.sort_index()
    without_tree_is_correct = without_tree_is_correct.sort_index()

    # Bar chart
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.4
    opacity = 0.8

    # Increase space between columns by adjusting the index
    index = np.arange(len(tree_is_correct)) * 1.5  # Multiply by a factor to increase spacing

    # Plot bars
    rects1 = plt.bar(index, tree_is_correct, bar_width, alpha=opacity, color='b', label='With tree')
    rects2 = plt.bar(index + bar_width, without_tree_is_correct, bar_width, alpha=opacity, color='r', label='Without tree')

    # Customize the plot
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')
    plt.xticks(index + bar_width / 2, tree_is_correct.index, rotation=90)  # Center ticks between bars
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.show()

    # save the figure
    fig.savefig(f'{dir}/comparison_tree_without_tree.png', bbox_inches='tight')

    return tree_is_correct, without_tree_is_correct

def plot_comparison_accuracy_memory(df_1, df_2, label_1, label_2, class_names: dict, dir: str):
    # compare between the including memory and not including memory

    # Group by class and calculate mean accuracy
    tree_is_correct_1 = df_1.groupby('Class')['tree_is_correct'].mean()
    tree_is_correct_2 = df_2.groupby('Class')['tree_is_correct'].mean()

    # Sort by class index
    tree_is_correct_1 = tree_is_correct_1.sort_index()
    tree_is_correct_2 = tree_is_correct_2.sort_index()

    # add class names to the series (column 'Class_name' is already added in the DataFrame)
    tree_is_correct_1.index = tree_is_correct_1.index.map(lambda x: class_names.get(x))
    tree_is_correct_2.index = tree_is_correct_2.index.map(lambda x: class_names.get(x))

    # Bar chart
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.4
    opacity = 0.8

    # Increase space between columns by adjusting the index
    index = np.arange(len(tree_is_correct_1)) * 1.5  # Multiply by a factor to increase spacing

    # Plot bars
    print(f"size of tree_is_correct_1: {len(tree_is_correct_1)}")
    print(f"size of tree_is_correct_2: {len(tree_is_correct_2)}")

    rects1 = plt.bar(index, tree_is_correct_1, bar_width, alpha=opacity, color='b', label=label_1)
    rects2 = plt.bar(index + bar_width, tree_is_correct_2, bar_width, alpha=opacity, color='r', label=label_2)

    # Customize the plot
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')
    plt.xticks(index + bar_width / 2, tree_is_correct_1.index, rotation=90)  # Center ticks between bars
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.show()

    # save the figure
    fig.savefig(f'{dir}/accuracy_comparison_memory_and_without_memory.png', bbox_inches='tight')

    # print the 
    return tree_is_correct_1, tree_is_correct_2

# plot the difference in accuracy
def plot_accuracy_difference(tree_is_correct_with_memory, tree_is_correct_without_memory, dir):
    # Calculate the difference in accuracy
    accuracy_diff = tree_is_correct_with_memory - tree_is_correct_without_memory

    # Sort by class index
    accuracy_diff = accuracy_diff.sort_index()

    # Create a bar chart for the difference
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.4
    opacity = 0.8

    index = np.arange(len(accuracy_diff)) * 1.5  # Increase spacing between bars

    rects = plt.bar(index, accuracy_diff, bar_width, alpha=opacity, color='g', label='Accuracy Difference')

    plt.xlabel('Class')
    plt.ylabel('Accuracy Difference')
    plt.title('Accuracy Difference by Class (With Memory - Without Memory)')
    plt.xticks(index + bar_width / 2, accuracy_diff.index, rotation=90)  # Center ticks between bars
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.show()
    
    # save the plt
    fig.savefig(f'{dir}/accuracy_difference_memory_and_without_memory.png', bbox_inches='tight')

def accuracy_comparison_grouped(tree_is_correct, without_tree_is_correct):
    # get the classes where the difference in accuracy is greater than or equal 0.4
    tree_diff = tree_is_correct - without_tree_is_correct
    tree_diff = tree_diff[tree_diff >= 0.4]
    # sort by value
    tree_diff = tree_diff.sort_values(ascending=False)
    for i in tree_diff.index:
        if tree_diff[i] < 0.2:
            continue
        print(f'Class {i} with tree outperform without tree by {tree_diff[i]*100}%')

    print('----------------------')
    # get the classes where the difference in accuracy is greater than or equal 0.4
    without_tree_diff = without_tree_is_correct - tree_is_correct
    without_tree_diff = without_tree_diff[without_tree_diff >= 0.4]
    # sort by value
    without_tree_diff = without_tree_diff.sort_values(ascending=False)
    # mean accuracy of each model
    print("Mean accuracy of model with tree: ", tree_is_correct.mean())
    print("Mean accuracy of model without tree: ", without_tree_is_correct.mean())
    print('----------------------')
    for i in without_tree_diff.index:
        if without_tree_diff[i] < 0.2:
            continue
        print(f'Class {i} without tree outperform with tree model by {without_tree_diff[i]*100}%')
    
def compare_classes(tree_is_correct, without_tree_is_correct):
    # store the classes outperformance of tree model over without tree model and vice versa
    tree_outperformance = 0
    without_tree_outperformance = 0
    same_performance = 0
    for i in range(len(tree_is_correct)):
        if tree_is_correct[i] > without_tree_is_correct[i]:
            tree_outperformance += 1
        elif tree_is_correct[i] < without_tree_is_correct[i]:
            without_tree_outperformance += 1
        else:
            same_performance += 1
    return tree_outperformance, without_tree_outperformance, same_performance


def plot_venn_comparison(df, dir):
    # rows where both models are correct
    both_correct = df[(df['tree_is_correct'] == 1) & (df['without_tree_is_correct'] == 1)]

    # rows where both models are incorrect
    both_incorrect = df[(df['tree_is_correct'] == 0) & (df['without_tree_is_correct'] == 0)]

    # rows where tree model is correct and the other is incorrect
    tree_is_correct_only = df[((df['tree_is_correct'] == 1) & (df['without_tree_is_correct'] == 0))]

    # rows where without tree model is correct and the other is incorrect
    without_tree_is_correct_only = df[((df['tree_is_correct'] == 0) & (df['without_tree_is_correct'] == 1))]

    # Create the Venn diagram
    venn2(subsets=(len(tree_is_correct_only), len(without_tree_is_correct_only), len(both_correct)),
        set_labels=('With Tree', 'Without Tree'))

    # save the figure
    plt.title('Venn Diagram of Tree vs Without Tree Model Correctness')
    plt.savefig(f'{dir}/venn_diagram_tree_wo_tree.png', bbox_inches='tight')

    print("Number of samples where both models are correct: ", len(both_correct))
    print("Number of samples where both models are incorrect: ", len(both_incorrect))
    print("Number of samples where tree model is correct and the other is incorrect: ", len(tree_is_correct_only))
    print("Number of samples where without tree model is correct and the other is incorrect: ", len(without_tree_is_correct_only))

def create_results_dir(dirs):
    # create the results directory if it does not exist
    if not os.path.exists('Results'):
        os.makedirs('Results')

    for dataset in dirs['datasets']:
        os.makedirs(f'Results/{dataset}', exist_ok=True)
        # create directories for each model
        for model in dirs['models']:
            os.makedirs(f'Results/{dataset}/{dirs["models"][model]}', exist_ok=True)
            # create directories for each temperature setting
            for temperature in dirs['temperature']:
                os.makedirs(f'Results/{dataset}/{dirs["models"][model]}/{dirs["temperature"][temperature]}', exist_ok=True)
            # create directories for each memory setting
            for memory in dirs['memory']:
                os.makedirs(f'Results/{dataset}/{dirs["models"][model]}/{dirs["memory"][memory]}', exist_ok=True)
                for temperature in dirs['temperature']:
                    os.makedirs(f'Results/{dataset}/{dirs["models"][model]}/{dirs["memory"][memory]}/{dirs["temperature"][temperature]}', exist_ok=True)
            
            # create directories for each description setting
            for description in dirs['description']:
                os.makedirs(f'Results/{dataset}/{dirs["models"][model]}/{dirs["description"][description]}', exist_ok=True)
                for temperature in dirs['temperature']:
                    os.makedirs(f'Results/{dataset}/{dirs["models"][model]}/{dirs["description"][description]}/{dirs["temperature"][temperature]}', exist_ok=True)
                
            os.makedirs(f'Results/{dataset}/{dirs["models"][model]}/description_zero_shot', exist_ok=True)

def ret_class_names(dataset):
    # load the class names from the json file
    if dataset == 'CIFAR10':
        dataset = 'cifar-10-python'
    with open(f'Data/{dataset}/class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # convert the class names to a dictionary
    class_names = {int(k): v for k, v in class_names.items()}
    
    return class_names

def visualization_violin_plot(df, model_name, dataset, dir):
    # Create a violin plot for the mean accuracy of the model across all classes
    classes_accuracies = df.groupby('Class')['tree_is_correct'].mean()
    classes_accuracies_without_tree = df.groupby('Class')['without_tree_is_correct'].mean()
    class_names = ret_class_names(dataset)

    fig, ax = plt.subplots(figsize=(15, 8))

    # Create a violin plot for the tree model
    ax.violinplot(classes_accuracies, positions=np.arange(len(classes_accuracies)), widths=0.8, showmeans=True, showmedians=True)
    # Create a violin plot for the without tree model
    ax.violinplot(classes_accuracies_without_tree, positions=np.arange(len(classes_accuracies_without_tree)) + 0.5, widths=0.8, showmeans=True, showmedians=True)
    # Set x-ticks to class names
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels([class_names[i] for i in range(len(class_names))], rotation=45)
    ax.set_title(f"Violin Plot of {model_name} on {dataset}")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"{dir}/violin_plot.png")
    plt.close()
    return classes_accuracies, classes_accuracies_without_tree

def visualization_box_plot(df, model_name, dataset, dir):
    # Create a box plot for the mean accuracy of the model across all classes

    # Group by class and calculate mean accuracy
    mean_accuracy = df.groupby('Class')['tree_is_correct'].mean()
    mean_accuracy_without_tree = df.groupby('Class')['without_tree_is_correct'].mean()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # # Box plot for with tree model
    # ax1.boxplot(mean_accuracy, vert=False)

def main():
    models = ['meta-llama/llama-3.2-11b-vision-instruct', 'qwen-vl-max', 'gpt-4o']
    include_memory = [True, False]
    include_description = [True, False]
    include_zero_shot_answer = [True, False]
    datasets = ['CIFAR10', 'GTSRB']
    RunId = 0

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
    },

    'memory': {
        True: 'with_memory',
        False: 'without_memory'
    },
    'description': {
        True: 'with_description',
        False: 'without_description'
    },
    'zero_shot_answer': {
        True: 'with_zero_shot_answer',
        False: 'without_zero_shot_answer'
    },
    "temperature": {
        0: 'temperature_0',
        0.7: 'temperature_0.7'
    }
    }
    # create the results directory if it does not exist
    create_results_dir(config)


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
            df = preprocess_dataframe(cursor, model, RunId)
            
            print(f"Number of rows in the dataframe: {len(df)}")
            if df.empty:
                print(f"No data found for model {model} on dataset {dataset}. Skipping...")
                continue
            
            dir = f"{config['datasets'][dataset]}/{model_to_name[model]}/"
            
            # mean of the accuracy for the model without tree
            for temperature in config['temperature']:
                df_temp = df[df['temperature'] == temperature]
                df_temp = df_temp[df_temp['Include_description'] != 2]

                print(f"Processing dataset: {dataset}, model: {model_to_name[model]}, temperature: {temperature} with {len(df_temp)} rows")

                # create the directory for the model
                local_dir = dir + config['temperature'][temperature] + '/'
                os.makedirs(local_dir, exist_ok=True)
                # plot the comparison accuracy between tree and without tree model accuracies
                tree_is_correct, without_tree_is_correct = plot_comparison_accuracy(df_temp, local_dir)
                # save the mean accuracy for the model
                mean_accuracy[dataset][model_to_name[model]][f'temperature_{temperature}'] = without_tree_is_correct.mean()

            for memory in include_memory:
                mean_accuracy[dataset][model_to_name[model]][f'memory_{memory}'] = {}
                mean_accuracy_classes[dataset][model_to_name[model]][f'memory_{memory}'] = {}
                outperformance[dataset][model_to_name[model]][f'memory_{memory}'] = {}
                # Fix description
                df_mem = df[df['Include_memory'] == memory]
                df_mem = df_mem[df_mem['Include_description'] == 0]
                # df_mem = df_mem[df_mem['temperature'] == 0.7]

                for temperature in config['temperature']:
                    mean_accuracy[dataset][model_to_name[model]][f'memory_{memory}'][f'temperature_{temperature}'] = {}
                    mean_accuracy_classes[dataset][model_to_name[model]][f'memory_{memory}'][f'temperature_{temperature}'] = {}
                    outperformance[dataset][model_to_name[model]][f'memory_{memory}'][f'temperature_{temperature}'] = {}
                    df_temp = df_mem[df_mem['temperature'] == temperature]
                    print(f"Processing dataset: {dataset}, model: {model_to_name[model]}, memory: {memory}, temperature: {temperature} with {len(df_temp)} rows")
                    
                    # First we do compare between the tree and without tree model
                    # construct the directory path
                    local_dir = dir + config['memory'][memory] + '/' + config['temperature'][temperature] + '/'

                    # plot the comparison accuracy between tree and without tree model accuracies
                    tree_is_correct, without_tree_is_correct = plot_comparison_accuracy(df_temp, local_dir)

                    # save the mean accuracy for the model
                    mean_accuracy[dataset][model_to_name[model]][f'memory_{memory}']['temperature_' + str(temperature)] = {
                        'tree': tree_is_correct.mean(),
                        'without_tree': without_tree_is_correct.mean()
                    }
                    outperformance_output = compare_classes(tree_is_correct, without_tree_is_correct)
                    outperformance[dataset][model_to_name[model]][f'memory_{memory}']['temperature_' + str(temperature)] = {
                        'tree_outperformance': outperformance_output[0],
                        'without_tree_outperformance': outperformance_output[1],
                        'same_performance': outperformance_output[2]
                    }

                    accuracy_comparison_grouped(tree_is_correct, without_tree_is_correct)

                    # plot the venn diagram comparison
                    plot_venn_comparison(df_temp, local_dir)

                    # plot the violin plot for the accuracy of the model
                    # classes_accuracies, classes_accuracies_without_tree = visualization_violin_plot(df_temp, model_to_name[model], dataset, local_dir)
                    # save the classes accuracies
                    # mean_accuracy_classes[dataset][model_to_name[model]][f'memory_{memory}'][f'temperature_{temperature}'] = {
                    #     'tree': classes_accuracies.to_list(),
                    #     'without_tree': classes_accuracies_without_tree.to_list()
                    # }

            for description in include_description:
                mean_accuracy[dataset][model_to_name[model]][f'description_{description}'] = {}
                mean_accuracy_classes[dataset][model_to_name[model]][f'description_{description}'] = {}
                outperformance[dataset][model_to_name[model]][f'description_{description}'] = {}
                # Fix memory
                df_desc = df[df['Include_description'] == description]
                df_desc = df_desc[df_desc['Include_memory'] == 0]

                for temperature in config['temperature']:
                    mean_accuracy[dataset][model_to_name[model]][f'description_{description}']['temperature_' + str(temperature)] = {}
                    mean_accuracy_classes[dataset][model_to_name[model]][f'description_{description}']['temperature_' + str(temperature)] = {}
                    df_temp = df_desc[df_desc['temperature'] == temperature]
                    print(f"Processing dataset: {dataset}, model: {model_to_name[model]}, description: {description}, temperature: {temperature} with {len(df_temp)} rows")
                    # construct the directory path
                    local_dir = dir + config['description'][description] + '/' + config['temperature'][temperature] + '/'

                    # plot the comparison accuracy between tree and without tree model accuracies
                    tree_is_correct, without_tree_is_correct = plot_comparison_accuracy(df_temp, local_dir)
                    mean_accuracy[dataset][model_to_name[model]][f'description_{description}']['temperature_' + str(temperature)] = {
                        'tree': tree_is_correct.mean(),
                        'without_tree': without_tree_is_correct.mean()
                    }
                    outperformance_output = compare_classes(tree_is_correct, without_tree_is_correct)
                    outperformance[dataset][model_to_name[model]][f'description_{description}']['temperature_' + str(temperature)] = {
                        'tree_outperformance': outperformance_output[0],
                        'without_tree_outperformance': outperformance_output[1],
                        'same_performance': outperformance_output[2]
                    }
                    
                    accuracy_comparison_grouped(tree_is_correct, without_tree_is_correct)
                    # plot the violin plot for the accuracy of the model
                    # classes_accuracies, classes_accuracies_without_tree = visualization_violin_plot(df_temp, model_to_name[model], dataset, local_dir)

                    # # save the classes accuracies
                    # mean_accuracy_classes[dataset][model_to_name[model]][f'description_{description}']['temperature_' + str(temperature)] = {
                    #     'tree': classes_accuracies.to_list(),
                    #     'without_tree': classes_accuracies_without_tree.to_list()
                    # }

            # write the accuracy of the zero shot with description with temperature 0, 0.7
            mean_accuracy[dataset][model_to_name[model]]['description_zero_shot'] = {}
            for temperature in config['temperature']:
                df_zero_shot_desc = df[df['Include_description'] == 2]
                df_zero_shot_desc = df[df['temperature'] == temperature]
                tree_is_correct_zero_shot, without_tree_is_correct_zero_shot = plot_comparison_accuracy(df_zero_shot_desc, dir + 'description_zero_shot')
                mean_accuracy[dataset][model_to_name[model]]['description_zero_shot'][f'temperature_{temperature}'] = without_tree_is_correct_zero_shot.mean()

            
            # comparison between the memory and without memory model
            # df_memory = df[df['Include_memory'] == True]
            # df_without_memory = df[df['Include_memory'] == False]
            # class_names = ret_class_names(dataset)
            # tree_is_correct_with_memory, tree_is_correct_without_memory = plot_comparison_accuracy_memory(df_memory, df_without_memory, 'Include Memory', 'Without Memory', class_names, dir)
            # plot_accuracy_difference(tree_is_correct_with_memory, tree_is_correct_without_memory, dir)

            # plot the box plot for the mean accuracy of the model across all classes
            # visualization_box_plot(df, model_to_name[model], dataset, dir)


            # get the zero temperature accuracy
            # df_zero_temp = df[df['temperature'] == 0]
            # df_non_zero_temp = df[df['temperature'] == 0.7]
            # print(f"Processing dataset: {dataset}, model: {model_to_name[model]}, temperature: 0 with {len(df_zero_temp)} rows")
            # tree_is_correct_zero_temp, without_tree_is_correct_zero_temp = plot_comparison_accuracy(df_zero_temp, dir + 'temperature_0')
            # accuracy_comparison_grouped(tree_is_correct_zero_temp, without_tree_is_correct_zero_temp)

            # print(f"Processing dataset: {dataset}, model: {model_to_name[model]}, temperature: 0.7 with {len(df_non_zero_temp)} rows")
            # tree_is_correct_non_zero_temp, without_tree_is_correct_non_zero_temp = plot_comparison_accuracy(df_non_zero_temp, dir + 'temperature_0.7')

            # accuracy_comparison_grouped(tree_is_correct_non_zero_temp, without_tree_is_correct_non_zero_temp)
            # print('-----------------------------------------------')
            # print()
    # save the mean accuracy for the model
    with open('Results/mean_accuracy.json', 'w') as f:
        json.dump(mean_accuracy, f, indent=4)
    
    # save the outperformance for the model
    with open('Results/outperformance.json', 'w') as f:
        json.dump(outperformance, f, indent=4)

    # save the mean accuracy classes for the model
    # with open('Results/mean_accuracy_classes.json', 'w') as f:
    #     json.dump(mean_accuracy_classes, f, indent=4)
if __name__ == "__main__":
    main()
    print("Done")