'''This script puts the actual and predicted node to be able to create confusion matrix'''
import json
import os

with open('Results/ground_truth_branches.json', 'r') as file:
    node_paths = json.load(file)

classes = node_paths.keys()

# Actual class - Predicted class - layer_question - Actual Q answer - Predicted Q answer
# Actual "0": {
#     Predicted "1": [layer_question, Actual Q answer, Predicted Q answer]
# }
res = {}
for actual_class in classes:
    for predicted_class in classes:
        if actual_class == predicted_class:
            continue
        i = 0
        actual_path = node_paths[actual_class]
        predicted_path = node_paths[predicted_class]
        while i < len(actual_path):
            question = list(actual_path[i].keys())[0]
            actual_answer = list(actual_path[i].values())[0]
            predicted_answer = list(predicted_path[i].values())[0]
            print(f"Question: {question}")
            print(f"Actual Answer: {actual_answer}")
            print(f"Predicted Answer: {predicted_answer}")
            if actual_answer != predicted_answer:
                print(f"Actual {actual_class}: {predicted_class}: {question}: {actual_answer}: {predicted_answer}")
                if actual_class not in res:
                    res[actual_class] = {predicted_class: [question, actual_answer, predicted_answer]}
                else:
                    res[actual_class][predicted_class] = [question, actual_answer, predicted_answer]
                break
            i += 1

print(res)
with open('node_comparison.json', 'w') as file:
    json.dump(res, file)
