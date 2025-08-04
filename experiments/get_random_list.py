import random
import json

random_numbers = random.sample(range(1, 12629), 50)

with open(f"Results/random_list.json", "w") as f:
    json.dump(random_numbers, f, indent=4)