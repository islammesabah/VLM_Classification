from openai import OpenAI
from PIL import Image
from tqdm import tqdm
import random
import json
import base64
from io import BytesIO
import re
import time

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access environment variables
api_key = os.getenv("API_KEY")

# random_numbers = random.sample(range(1, 12629), 100)

# Load class names
with open('Results/random_list.json', 'r') as file:
	random_numbers = json.load(file)

# Define local dataset path
dataset_local_path = "Data/GTSRB"

client = OpenAI(
	base_url="https://api-inference.huggingface.co/v1/",
	api_key=api_key
)

def encode_image(image):
    return base64.b64encode(image).decode('utf-8') 

# Load class names
with open(f'{dataset_local_path}/class_names.json', 'r') as file:
		class_names = json.load(file)

class_names_str = ", ".join([ i +" : "+ class_names[i] for i in class_names.keys()])

prompt = f"""Please classify the traffic sign in the given image. It should be only one of these classes list: {{{class_names_str}}}. Please generate only the ID for the class."""

res_obj = {}
for random_image_number in tqdm(random_numbers):                                                   
	# Path to your image
	image_path = f'{dataset_local_path}/test/images/{random_image_number}.png'
	img = Image.open(image_path)
	buffer = BytesIO()
	img.save(buffer, format="JPEG")
	img_bytes = buffer.getvalue()
	base64_image = encode_image(img_bytes)
	
	messages = [
		{
			"role": "user",
			"content": [
				{
					"type": "text",
					"text": prompt
				},
				{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{base64_image}"
					}
				}
			]
		}
	]

	completion = client.chat.completions.create(
		model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
		messages=messages, 
		max_tokens=100
	)


	res = completion.choices[0].message.content

	# Regex to find the first number
	match = re.search(r'^\D*(\d+)', res)

	if match:
		class_ = int(match.group(1))
		if str(class_) not in class_names.keys():
			class_ = -1
	else:
		class_ = -1
	
	if class_ == -1:
		print(f"Error with image {random_image_number}")
		print("Full Model Results:", res)
		
	res_obj[random_image_number] = class_

	time.sleep(3)


with open(f"Results/random_zero_shot_llama.json", "w") as f:
        json.dump(res_obj, f, indent=4)