'''The script is used to count the visited nodes in the tree structure and the correctly classified nodes in the tree structure. The results are saved in the nodes_accuracy.json file.'''
import json
import re
import sqlite3
# Accuracy = How many times you correctly 
# answered the question in node / How many times you answered the question in node
# for example : How many times you tried to answer "[L0] Q: What's the sign's primary shape?
# / How many times it was circle and you answered circle
# Actual class - Predicted class [count]
MODEL = "llama-3.2-11b-vision-preview"
conn = sqlite3.connect('Results/db/answers.db')
cursor = conn.cursor()

with open('Results/ground_truth_branches.json', 'r') as file:
    branches = json.load(file)

# get all the predictions in the db
cursor.execute(f'SELECT * FROM answers WHERE LLM_name = "{MODEL}"')
rows = cursor.fetchall()

counter = {}
counter_correctly_classified = {}
for row in rows:
    predicted_class_tree = int(row[6])
    actual_class = int(row[0])
    # get the nodes that are visited from the model
    if predicted_class_tree == -1:
        predicted_node = row[4]
        predicted_node = json.loads(predicted_node)
    else:
        predicted_node = branches[str(predicted_class_tree)]
        
    actual_nodes = branches[str(actual_class)]
    print(f"Predicted Node: {predicted_node}, Length: {len(predicted_node)}")
    print(f"Actual Node: {actual_nodes}, Length: {len(actual_nodes)}")
    for i in range(len(predicted_node)):
        
        predicted_question = list(predicted_node[i].keys())[0]
        predicted_answer = list(predicted_node[i].values())[0]

        actual_question = list(actual_nodes[i].keys())[0]
        actual_answer = list(actual_nodes[i].values())[0]

        if predicted_class_tree == -1:
            predicted_question = predicted_node[i]['question']
            predicted_answer = predicted_node[i]['answer']

        if predicted_question.lower() not in actual_question.lower():
            continue
        
        if actual_question not in counter:
            counter[actual_question] = 1
        else:
            counter[actual_question] += 1

        print(f"Actual Question: {actual_question}")
        print(f"Actual Answer: {actual_answer}")
        print(f"Predicted Question: {predicted_question}")
        print(f"Predicted Answer: {predicted_answer}")
        print("---------------------------------------")

        # if the answer is correct
        if re.search(rf'\b{re.escape(actual_answer)}\b', predicted_answer, flags=re.IGNORECASE):
            if actual_question not in counter_correctly_classified:
                counter_correctly_classified[actual_question] = 1
            else:
                counter_correctly_classified[actual_question] += 1
        else:
            break
    
print(counter)
print(counter_correctly_classified)
# divide correctly classified node by total node count, only save 2 decimal points
for question in counter_correctly_classified:
    counter_correctly_classified[question] = round(counter_correctly_classified[question] / counter[question], 2)

sorted_counter_correctly_classified = sorted(counter_correctly_classified.items(), key=lambda x: x[1], reverse=True)
with open(f"Results/nodes_accuracy_{MODEL}.json", "w") as file:
    json.dump(sorted_counter_correctly_classified, file)

