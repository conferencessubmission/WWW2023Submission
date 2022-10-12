import json

filename = "taxonomy.en-US.txt"

dict_ = {}

classes_dict = {} # classes at diferent depths

with open(filename, 'r', encoding='utf-8') as f:
	for line_idx, line in enumerate(f):
		if line_idx == 0:
			continue
		line = line.strip()
		if line == "":
			continue
		list_ = line.split(" > ")
		depth = len(list_)
		category = list_[-1]
		dict_[category] = {'depth': depth, 'hierarchy': list_}
		if depth not in classes_dict:
			classes_dict[depth] = []
		if category not in classes_dict[depth]:
			classes_dict[depth].append(category)


print("Depth, Number of classes")
for key, value in classes_dict.items():
	print(key, len(value))


with open("taxonomy_category_hierarchy.json", 'w') as f:
	json.dump(dict_, f, indent = 4)

with open("taxonomy_classes_hierarchy.json", 'w') as f:
	json.dump(classes_dict, f, indent = 4)