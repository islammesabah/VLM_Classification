'''This script is used to generate the answers for the traffic sign classification task(901 images covers all the classes and sequences) using the VLM with and without the tree structure. The answers are saved in the SQLite database.'''
from openai import OpenAI
from tqdm import tqdm
import json
import re
import time
from dotenv import load_dotenv
import os
import sqlite3
import argparse
from pydantic import BaseModel, Field

prompt = """You are a traffic sign classification expert. Your are given label name / class name of a German traffic sign, and your task is to answer questions about that traffic sign.
The label name is: {label_name}.
Past questions and answers are:
{past_questions_and_answers}
You will be asked a question about the visual description of a traffic sign, and you should answer it based on the label name and the past questions and answers, and explain your justification for the answer.
The question is: {question}
You SHOULD answer the question from the provided answers ONLY:
The answers are: {answers}
Your answer should be in the following format:
label_name: {label_name}
"""
# The script basically takes the true branch questions for each class, and asks the LLM to answer the question based only on the label name(class) and the past questions and answers that it gave.
# but we don't go on in the tree based on the answer, we continue with the right branch questions for the class.
# predefined:{"(Question,Class)": "Truelabel"}
# LLM:{"(Question,Class,LLM_name)": "LLM_answer"}
# For each class try the same question 5 times, vote for the most common answer. / (maybe we can take the error rate)
# evaluation: Question: accuracy(number of correct answers / total number of questions) and error rate (number of wrong answers / total number of questions)

# Define local dataset path
dataset_local_path = "/netscratch/elmansoury/TreeVLM"

class AnswerFormat(BaseModel):
    Answer: str = Field(..., description="The answer to the question.", example="Circle")
    Justification: str = Field(..., description="The justification for the answer.", example="The 60 Kph Limit sign is a speed limit sign, which tells drivers to slow down to 60 kilometers per hour. The sign is a circle with a red border and a white background, with the number 60 in black in the center. The circle shape is a common shape for traffic signs, and the red border indicates that it is a regulatory sign.")


"""Class TEXT,
        LLM_name TEXT,
        Question TEXT,
        LLM_answer TEXT,
        Justification TEXT,
        history_questions_answers TEXT,"""
def insert_answer(conn_labeled, cursor_labeled, class_, LLM_name, Question, LLM_answer, Justification, history_questions_answers):
    image_path = str(image_path)
    # cursor_labeled.execute('SELECT * FROM answers WHERE Image_path = ?', (image_path,))
    # result = cursor.fetchone()
    # if result is not None:
    #     return
    # print the columns of the table
    # Insert the answer into the 'answers' table
    sql = '''
    INSERT INTO answers (Class, LLM_name, Question, LLM_answer, Justification, history_questions_answers)
    VALUES (:Class, :LLM_name, :Question, :LLM_answer, :Justification, :history_questions_answers)'''

    props = {
        'Class': class_,
        'LLM_name': LLM_name,
        'Question': Question,
        'LLM_answer': LLM_answer,
        'Justification': Justification,
        'history_questions_answers': history_questions_answers
    }

    cursor_labeled.execute(sql, props)
    # Commit the changes and close the connection
    conn_labeled.commit()

def tree_inference(client, model, class_, class_name, ground_truth_question_answer, questions_to_answers, conn, cursor):
    whole_tree_path = []
    accuracy = 0
    questions_to_correctness = {}
    for question, ground_truth_answer in ground_truth_question_answer.items():
        question = question.split("Q:", 1)[1].strip() # remove the level number e.g. L[1]
        history_questions_str = [f"{k}: {v}" for k, v in whole_tree_path.items()].join("\n")
        possible_answers = questions_to_answers[question]
        print(f"Question: {question}, Possible Answers: {possible_answers}")
        print(f"Ground Truth Answer: {ground_truth_answer}")

        prompt = prompt.format(
            label_name=class_name,
            past_questions_and_answers=history_questions_str,
            question=question,
            answers=json.dumps(possible_answers)
        )
        # Print the prompt
        print("Prompt: ", prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ]
        # Sort the answers first, as some answers are substrings of others
        treeanswers_sorted = sorted(possible_answers, key=len, reverse=True)

        # Regex here matches if one of the predefined answers is in the response. (takes the first one)
        regex_pattern = "(?i)("+"|".join(["\\b"+ans+"\\b" for ans in treeanswers_sorted])+")"
        
        completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=AnswerFormat,
    )

        response_json = completion.choices[0].message.content
        print(f"Response JSON: {response_json}")
        LLM_answer = response_json.Answer
        justification = response_json.Justification
        res = f"Answer: {LLM_answer}"

        whole_tree_path.append({"question": question, "LLM_answer": res})
        print("############################################")
        print("Response: ", res, "Justification: ", justification)
        match = re.search(regex_pattern.lower(), res.lower())
        insert_answer(conn, cursor, class_, model, question, LLM_answer, justification, json.dumps(whole_tree_path))
        if match:
            LLM_answer = match.group(1)
            print(f"Class: {class_name}, Question: {question}, Answer: {LLM_answer}")
            if LLM_answer.lower() == ground_truth_answer.lower():
                accuracy += 1
                questions_to_correctness[question] = 1
            else:
                questions_to_correctness[question] = 0
                print(f"Incorrect Answer for Class {class_name}, Question: {question}, Answer: {LLM_answer}, Ground Truth: {ground_truth_answer}")
        print("############################################")
        time.sleep(1)
    
    accuracy = accuracy / len(ground_truth_question_answer)
    print(f"Accuracy for class {class_}: {accuracy:.2f}")

    return whole_tree_path, accuracy, questions_to_correctness

def get_class_names():
    # Load class names
    with open(f'{dataset_local_path}/class_names_short_names.json', 'r') as file:
        class_names = json.load(file)
    return class_names

def handle_client():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='The model to use for the inference.', choices=['gpt-4o', 'gpt-4.1-2025-04-14', 'qwen-vl-max', 'llama-3.2-11b-vision-preview', 'llama-3.2-90b-vision-preview', 'llava:34b', 'gemma3:27b'], required=True)
    parser.add_argument('--hostip', type=str, help='The host ip address (see output of host_ollama.sh).', required=False)
    args = parser.parse_args()
    load_dotenv()
    model = args.model
    hostip = args.hostip
    model_to_api_key = {
        'gpt-4o': os.getenv("OPENAI_API_KEY"),
        'gpt-4.1-2025-04-14': os.getenv("OPENAI_API_KEY"),
        'llama-3.2-11b-vision-preview': os.getenv("GROQ_API_BARGH"),
        'llama-3.2-90b-vision-preview': os.getenv("GROQ_API_BARGH"),
        'qwen-vl-max': os.getenv("ALIBABA"),
        "llava:34b": "ollama",
        "gemma3:27b": "ollama",

    }
    model_to_base_url = {
        'gpt-4o': None,
        'gpt-4.1-2025-04-14': None,
        'llama-3.2-11b-vision-preview': "https://api.groq.com/openai/v1",
        'llama-3.2-90b-vision-preview': "https://api.groq.com/openai/v1",
        'qwen-vl-max': "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "llava:34b": f"http://{hostip}:11434/v1",
        "gemma3:27b": f"http://{hostip}:11434/v1",
    }
    print(f"Using model: {model}")
    print(f"Using base url: {model_to_base_url[model]}")
    return model, OpenAI(api_key=model_to_api_key[model], base_url=model_to_base_url[model])

def main():
    model, client = handle_client()
    class_names = get_class_names()

    # Connect to the SQLite database
    conn = sqlite3.connect("Results/db/likelihood.db")
    cursor = conn.cursor()

    # get class questions and their ground truth answers
    with open(f'{dataset_local_path}/ground_truth_branches.json', 'r') as file:
        ground_truth_question_answer = json.load(file)
    
    # get the questions to answers mapping
    with open(f'{dataset_local_path}/questions_to_all_answers.json', 'r') as file:
        questions_to_answers = json.load(file)
    
    for class_, class_name in tqdm(class_names.items(), desc="Processing Classes"):
        print(f"Processing class: {class_} - {class_name}")
        # Build the traffic sign tree for the class
        
        # Perform inference with the tree structure
        whole_tree_path, accuracy, questions_to_correctness = tree_inference(client, model, class_, class_name, ground_truth_question_answer[class_], questions_to_answers, conn, cursor)

        # Print the accuracy and questions to correctness mapping
        print("############################################")
        print(f"Class: {class_} - {class_name}")
        print("The whole tree questions answers: ", whole_tree_path)
        print(f"Accuracy for class {class_}: {accuracy:.2f}")
        print(f"Questions to correctness mapping: {questions_to_correctness}")
    

if __name__ == "__main__":
    main()

# I will give it the label without the image, and the history of the answered questions 

