import json

with open('Data/GTSRB/test/metadata.json', 'r') as file:
	gt = json.load(file)
	
with open('Results/random_zero_shot_llama_50_random.json', 'r') as file:
	pred = json.load(file)
	
with open('Results/random_tree_llama_50_random.json', 'r') as file:
	pred_t = json.load(file)
	
acc = 0
for image in pred.keys():
	gt_label = list(filter(lambda i: i["image"] == f"images/{image}.png", gt))[0]["label"]
	if pred[image] == gt_label:
		acc += 1
		
print(f"Zero-Shot Accuracy: {(acc/len(pred))*100}%")

acc = 0
for image in pred_t.keys():
	gt_label = list(filter(lambda i: i["image"] == f"images/{image}.png", gt))[0]["label"]
	# print(image, gt_label, pred_t[image])
	if pred_t[image] == gt_label:
		acc += 1

print(f"Tree Accuracy: {(acc/len(pred_t))*100}%")