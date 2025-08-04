from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os
import json
import regex as re

# Load environment variables from the .env file
load_dotenv()

# Access environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

#llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
llm = ChatOpenAI(
      # model="gpt-4o-mini",     # Price per 1M tokens: Input: $0.15, Cached input:$0.075, Output: $0.60
      model="gpt-4o",        # Price per 1M tokens: Input: $2.50, Cached input:$1.25, Output: $10.00
      # model="o1-mini",       # Price per 1M tokens: Input: $1.10, Cached input:$0.55, Output: $4.40
      )

# Load class names
with open('Data/BIRDS_200_2011/class_names.json', 'r') as file:
	lst = json.load(file)

previous_questions = []

# Basic example with a single variable
template = """
Generate a simple, clear question to visually categorize a given list of classes' names, ensuring classes fit into one of the generated question's possible answers.  

**Guidelines:**  
- Use your knowledge of the classes' appearances to create an effective question and possible answers.  
- Try to think of how each class looks like before creating the question and possible answers.
- The question should maximize the splits among the given classes.  
- Every class must be assigned to exactly one answer.
- Duplicate class assignments will result in the rejection of the whole output.
- Output must follow the specified object format shown in the example, or it will be rejected.  
- Avoid questions similar in meaning to:  
  - {previous_questions}  

**Example:**  
**List:**  
{{  
    '1': 'red and white circle red car and black car no passing',
    '2': 'white and yellow diamond priority road',
    '3': 'red circle with white horizontal stripe no entry',
    '4': 'red and white upside-down triangle yield right-of-way',
    '5': 'red and white circle 120 kph speed limit',
    '6': 'red and white triangle with exclamation mark warning'
}}  

**Output:**  
{{  
    "Question": "What is the shape of the sign?",  
    "Answers": {{  
        "Circle": [1, 3, 5],  
        "Triangle": [4, 6],  
        "Diamond": [2]  
    }}  
}}  


**Task:**  
Do not forget (Very Important):
- Every class must be assigned to exactly one answer.
- Duplicate class assignments will result in the rejection of the whole output.

**Given List:**  
{lst}  

**Output:**

"""

prompt_template  = PromptTemplate(
    input_variables=["lst", "previous_questions"],
    template=template,
)

def has_duplicates(lst):
    return len(lst) != len(set(lst))

total_cost = 0
def get_question(lst, previous_questions=[], depth=0):
    if depth > 1:
        return None
    
    print("Depth: ", depth)
    print("Previous Questions: ", previous_questions)

    # Generate the prompt
    prompt = prompt_template.format(lst=lst, previous_questions="\n  - ".join(previous_questions))
    
    in_list_vals = [int(cls) for cls in lst.keys() ]
    
    # get the model output
    result = llm.invoke(prompt)
    
    # get the json object from the content
    match = re.search(r"\{(?:[^{}]|(?R))*\}", result.content)

    if match:
        try:
            extract_text = match.group()
            json_output = json.loads(extract_text.replace("'", "\""))
            print(json_output)
            
            # Check for duplicates
            generated_cls_list = sorted([cls for classes in json_output['Answers'].values() for cls in classes])
            if has_duplicates(generated_cls_list):
                raise ValueError("Duplicate class assignments found")
            
            # Check for missing classes
            if set(generated_cls_list) != set(in_list_vals):
                difference = list(set(in_list_vals)-set(generated_cls_list))
                json_output["Answers"]["Other"] = difference
            
            
            # recursively call the function to build the tree
            for key in json_output["Answers"]:
                print("Key: ", key)
                key_classes = sorted(json_output["Answers"][key])
                class_list = {str(cls): lst[str(cls)] for cls in key_classes}
                if len(key_classes) > 1:
                    question = get_question(class_list, previous_questions + [json_output["Question"]], depth+1)
                    if question:
                        json_output["Answers"][key] = question

            return json_output
                   
        except Exception as e:
            print(e)
            return e
    else:
        print("No match")
        return "No match"

with get_openai_callback() as cb:  
    print(get_question(lst))
    print(cb)


# one depth answer from the Gpt-4o model for flower dataset
# {
#     'Question': 'What is the predominant color of the flower?', 
#     'Answers': {
#         'Red/Pink': {
#             'Question': 'What is the dominant color of the flower?', 
#             'Answers': {
#                 'Pink': [0, 59, 87], 
#                 'Yellow': [14], 
#                 'Red': [23, 84], 
#                 'Orange': [58], 
#                 'Other': [20, 25, 38]
#             }
#         }, 
#         'Yellow/Orange': {
#             'Question': 'What type of flowers are they?', 
#             'Answers': {
#                 'Wildflowers': [21, 47, 64, 70], 
#                 'Garden Flowers': [41, 53, 80, 95], 
#                 'Vines': [74, 75]
#             }
#         }, 
#         'White': {
#             'Question': 'What type of flower is it?', 
#             'Answers': {
#                 'Orchid': [6], 
#                 'Daisy': [11, 49], 
#                 'Lily': [19], 
#                 'Fern': [26], 
#                 'Aquatic': [77], 
#                 'Amaryllis': [90], 
#                 'Petunia': [97]
#             }
#         }, 
#         'Purple/Blue': {
#             'Question': 'Does the flower have a prominent central disc or spike?', 
#             'Answers': {
#                 'Yes, it has a central disc or spike': [1, 13], 
#                 'No, it does not have a central disc or spike': [2, 3, 8, 18, 51, 60, 66, 68]
#             }
#         }, 'Multicolor': {'Question': 'What is the general structure of the flower?', 'Answers': {'Lily-like': [5, 17], 'Round bulb-like': [12, 15, 22], 'Daisy-like': [16, 31, 40], 'Cluster-like': [44], 'Classic floral head': [73]}}, 'Other': {'Question': 'What type of plant is this?', 'Answers': {'Flower': [30, 32, 35, 37, 39, 42, 45, 46, 48, 50, 52, 54, 55, 57, 61, 62, 63, 65, 67, 69, 71, 72, 76, 78, 79, 81, 83, 85, 86, 89, 91, 93, 94, 96, 99, 100, 101], 'Shrub': [43], 'Aquatic': [88], 'Epiphyte': [92, 98], 'Other': [56, 82, 36]}}}
# }
# Tokens Used: 5025
#         Prompt Tokens: 4191
#         Completion Tokens: 834
# Successful Requests: 7
# Total Cost (USD): $0.0

# one depth answer from the Gpt-4o model for traffic signs dataset
# {
#     'Question': 'What is the color scheme of the sign?', 
#     'Answers': {
#         'Red and White': {
#             'Question': 'What is the shape of the sign?', 
#             'Answers': {
#                 'Circle': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 16, 17], 
#                 'Triangle': [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], 
#                 'Upside-Down Triangle': [13]
#             }
#         }, 
#         'White and Yellow': [12], 
#         'Blue and White': {
#             'Question': 'What is the main arrow orientation on the sign?', 
#             'Answers': {
#                 'Right': [33, 36, 38, 40], 
#                 'Left': [34, 37, 39], 
#                 'Forward': [35]
#             }
#         }, 
#         'White with Gray Strike Bar': {
#             'Question': 'What type of road regulation is indicated by the sign?', 
#             'Answers': {
#                 'End of speed limit': [6, 32], 
#                 'End of passing restriction': [41, 42], 
#                 'Empty circle': [15]
#             }
#             }, 
#         'Stop (Red)': [14]
#     }
# }
# Tokens Used: 3131
#         Prompt Tokens: 2686
#         Completion Tokens: 445
# Successful Requests: 4
# Total Cost (USD): $0.0



# one depth answer from the Gpt-4o model for birds dataset
# { 
#      'Question': 'What is the primary color associated with the bird?', 
#      'Answers': {
#           'Black': [0, 1, 2, 3, 8, 9, 10, 11, 25, 26, 28, 29, 48, 50, 51, 52, 56, 60, 61, 62, 63, 64, 65, 71, 140, 141, 142, 143], 
#           'White': [17, 18, 21, 83, 101, 92, 93, 100, 130, 131, 132], 
#           'Blue': [13, 14, 38, 40, 57, 66, 67, 94, 97, 47, 46, 96, 153], 
#           'Red': [9, 11, 86, 99, 109, 190, 44, 65, 87], 
#           'Brown': [7, 17, 58, 102, 79, 104, 108, 130, 132], 
#           'Yellow': [19, 44, 68, 69, 95, 96, 139, 142, 144, 156, 180, 186]
#     }
# }
# Duplicate class assignments found

# Tokens Used: 3014
#         Prompt Tokens: 2715
#         Completion Tokens: 299
# Successful Requests: 1
# Total Cost (USD): $0.0