#!/usr/bin/env python3
"""
Script to convert JSON performance data to LaTeX table format.
Compares tree-based methods against baseline (without-tree) method.
"""

import json
import sys
from pathlib import Path

def load_json_data(filepath):
    """Load JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{filepath}'.")
        sys.exit(1)

def get_method_data(model_data, method_key, temperature):
    """Extract data for a specific method and temperature."""
    if method_key in model_data and f"temperature_{temperature}" in model_data[method_key]:
        data = model_data[method_key][f"temperature_{temperature}"]
        return data['tree_outperformance'], data['without_tree_outperformance'], data['same_performance']
    return None, None, None

def format_comparison(tree_win, baseline_win, ties):
    """Format comparison showing tree wins vs baseline wins."""
    if tree_win is None:
        return "N/A"
    
    total = tree_win + baseline_win + ties
    if tree_win > baseline_win:
        return f"\\textbf{{{tree_win}}}/{baseline_win}"
    elif baseline_win > tree_win:
        return f"{tree_win}/\\textbf{{{baseline_win}}}"
    else:
        return f"{tree_win}/{baseline_win}"

def generate_latex_table(data, temperature):
    """Generate LaTeX table from JSON data."""
    
    latex_output = []
    latex_output.append("\\begin{table*}[ht]")
    latex_output.append("\\centering")
    latex_output.append(f"\\caption{{Performance Comparison: Tree-based Methods vs Baseline (Temperature = {temperature})}}")
    latex_output.append("\\begin{tabular}{ll|cc|cc|cc}")
    latex_output.append("\\hline")
    latex_output.append("\\multirow{2}{*}{\\textbf{Dataset}} & \\multirow{2}{*}{\\textbf{Model}}")
    latex_output.append("& \\multicolumn{2}{c|}{\\textbf{Tree-based}}")
    latex_output.append("& \\multicolumn{2}{c|}{\\textbf{Tree + History}}")
    latex_output.append("& \\multicolumn{2}{c}{\\textbf{Tree + Description}} \\\\")
    latex_output.append("\\cline{3-8}")
    latex_output.append("& & \\textbf{Win/Loss} & \\textbf{Ties} & \\textbf{Win/Loss} & \\textbf{Ties} & \\textbf{Win/Loss} & \\textbf{Ties} \\\\")
    latex_output.append("\\hline")
    
    # Model name mapping for cleaner display
    model_mapping = {
        "Llama 3.2 11B Vision Instruct": "LLAMA",
        "Qwen VL Max": "Qwen-VL",
        "GPT-4o": "GPT-4o"
    }
    
    # Process each dataset
    for dataset_idx, (dataset_name, dataset_data) in enumerate(data.items()):
        first_row = True
        
        for model_name, model_data in dataset_data.items():
            # Get data for each method
            tree_basic_win, tree_basic_loss, tree_basic_ties = get_method_data(model_data, "memory_False", temperature)
            tree_history_win, tree_history_loss, tree_history_ties = get_method_data(model_data, "memory_True", temperature)
            tree_desc_win, tree_desc_loss, tree_desc_ties = get_method_data(model_data, "description_True", temperature)
            
            # Format comparisons
            basic_comp = format_comparison(tree_basic_win, tree_basic_loss, tree_basic_ties)
            history_comp = format_comparison(tree_history_win, tree_history_loss, tree_history_ties)
            desc_comp = format_comparison(tree_desc_win, tree_desc_loss, tree_desc_ties)
            
            # Format ties
            basic_ties = str(tree_basic_ties) if tree_basic_ties is not None else "N/A"
            history_ties = str(tree_history_ties) if tree_history_ties is not None else "N/A"
            desc_ties = str(tree_desc_ties) if tree_desc_ties is not None else "N/A"
            
            dataset_cell = f"\\multirow{{3}}{{*}}{{{dataset_name}}}" if first_row else ""
            model_display = model_mapping.get(model_name, model_name)
            
            latex_output.append(f"{dataset_cell} & {model_display} & {basic_comp} & {basic_ties} & {history_comp} & {history_ties} & {desc_comp} & {desc_ties} \\\\")
            
            first_row = False
        
        # Add separator between datasets
        if dataset_idx < len(data) - 1:
            latex_output.append("\\hline")
    
    latex_output.append("\\hline")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\label{tab:performance_comparison}")
    latex_output.append("\\end{table*}")
    
    return '\n'.join(latex_output)

def main():
    """Main function."""
    
    temperature = sys.argv[1]
    input_file = sys.argv[2] if len(sys.argv) > 2 else "paste.txt"
    
    # Validate temperature
    if temperature not in ["0", "0.7"]:
        print("Error: Temperature must be either '0' or '0.7'")
        sys.exit(1)
    
    # Load and process data
    print(f"Loading data from: {input_file}")
    print(f"Using temperature: {temperature}")
    data = load_json_data(input_file)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(data, temperature)
    
    # Output to file
    output_file = f"comparison_table_temp_{temperature}.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to: {output_file}")
    print("\n" + "="*60)
    print("Generated LaTeX Table:")
    print("="*60)
    print(latex_table)
    print("\n" + "="*60)
    print("Legend:")
    print("Win/Loss format: TreeWins/BaselineWins (bold indicates higher value)")
    print("Ties: Number of classes where performance was equal")

if __name__ == "__main__":
    main()