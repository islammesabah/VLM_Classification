import json

def load_data_from_file(filename):
    """Load JSON data from file"""
    with open(filename, 'r') as f:
        return json.load(f)

def extract_best_scores(data, temperature):
    """Extract the best scores for each model and dataset configuration"""
    results = {}
    
    for dataset, models in data.items():
        results[dataset] = {}
        
        for model, configs in models.items():
            model_results = {}
            
            # Extract tree-based scores (memory_False, description_False)
            base_config = configs.get('memory_False', {}).get(temperature, {})
            model_results['tree_based'] = base_config.get('tree', 0)
            # Assuming zero-shot result of the memory_False and description_False configuration
            # As the zeroshot is done 6 times (memory_False(true, false), description_False(true, false), temperature(0, 0.7))
            model_results['zero_shot'] = base_config.get('without_tree', 0)
            
            # Extract tree-based with memory (memory_True, description_False)
            memory_config = configs.get('memory_True', {}).get(temperature, {})
            model_results['tree_memory'] = memory_config.get('tree', 0)
            
            # Extract tree-based with description (memory_False, description_True)
            desc_config = configs.get('description_True', {}).get(temperature, {})
            model_results['tree_description'] = desc_config.get('tree', 0)
            
            results[dataset][model] = model_results
    
    return results

def format_percentage(value):
    """Format value as percentage with 2 decimal places"""
    return f"{value * 100:.2f}\\%"

def find_best_score(scores):
    """Find the maximum score and return formatted string with bold if it's the best"""
    max_score = max(scores)
    formatted_scores = []
    
    for score in scores:
        formatted = format_percentage(score)
        if score == max_score:
            formatted = f"\\textbf{{{formatted}}}"
        formatted_scores.append(formatted)
    
    return formatted_scores

def generate_latex_table(results):
    """Generate LaTeX table from results"""
    
    # Model name mapping for cleaner display
    model_mapping = {
        'Llama 3.2 11B Vision Instruct': 'LLAMA',
        'Qwen VL Max': 'Qwen-VL',
        'GPT-4o': 'GPT-4o'
    }
    
    # Start building the LaTeX table
    latex_lines = [
        "\\begin{table*}[ht]",
        "\\centering",
        "\\caption{Mean Accuracy Comparison across different hyperparameter settings}",
        "\\begin{tabular}{llcccc}",
        "\\hline",
        "\\textbf{Dataset} & \\textbf{Model} & \\textbf{Tree-based} & \\textbf{Tree-based with Memory} & \\textbf{Tree-based with Description} & \\textbf{Zero-shot} \\\\ \\hline"
    ]
    
    # Process each dataset
    for i, (dataset, models) in enumerate(results.items()):
        if i > 0:  # Add hline between datasets
            latex_lines.append("\\hline")
        
        # Process each model for this dataset
        for j, (model, scores) in enumerate(models.items()):
            model_name = model_mapping.get(model, model)
            
            # Get all scores for this model
            score_values = [
                scores['tree_based'],
                scores['tree_memory'], 
                scores['tree_description'],
                scores['zero_shot']
            ]
            
            # Format scores with bold for the best one
            formatted_scores = find_best_score(score_values)
            
            # Create the row
            if j == 0:  # First model for this dataset
                row = f"{dataset} & {model_name} & {formatted_scores[0]} & {formatted_scores[1]} & {formatted_scores[2]} & {formatted_scores[3]} \\\\"
            else:  # Subsequent models
                row = f"      & {model_name} & {formatted_scores[0]} & {formatted_scores[1]} & {formatted_scores[2]} & {formatted_scores[3]} \\\\"
            
            latex_lines.append(row)
    
    # Close the table
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\label{tab:accuracy_comparison_hyperparams}",
        "\\end{table*}"
    ])
    
    return "\n".join(latex_lines)

def main():
    temperature = 'temperature_0'
    # Load data from the JSON file
    with open('mean_accuracy.json', 'r') as f:
        data = json.load(f)
    
    # Extract the best scores
    results = extract_best_scores(data, temperature)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    
    # Optionally save to file
    with open('generated_table_temp0.tex', 'w') as f:
        f.write(latex_table)

    print("\n\nTable saved to 'generated_table_temp0.tex'.")

if __name__ == "__main__":
    main()