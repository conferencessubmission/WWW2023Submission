## TODO: replace plural objects with singular

import json
import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_web_md')

# from sklearn.metrics import jaccard_score

def jaccard(a, b):
	# a and b are lists of words
	a = set(a)
	b = set(b)
	c = a.intersection(b)
	return float(len(c)) / (len(a) + len(b) - len(c))

def get_vec_sim(a, b):
	# a and b are sentences
	doc1 = nlp(a)
	doc2 = nlp(b)
	return doc1.similarity(doc2)

def get_total_sim(a, b, a_2):
	# a and b are sentences
	jacc_sim = jaccard(a.split(' '), b.split(' '))
	if jacc_sim == 1: # exact match
		return 10000
	b_compact = b.replace(" ", '')
	a_compact = a.replace(" ", '')
	if (jacc_sim == 0) and (b_compact not in a_compact) and (a_compact not in b_compact):
		return 0
	sim_1 = get_vec_sim(a, b)
	if a_2 == "":
		return sim_1
	return sim_1 + get_vec_sim(a_2, b)

with open("taxonomy_category_hierarchy.json", 'r') as f:
	dict_ = json.load(f)

with open("links_2.json", 'r') as f:
	manuals_info = json.load(f)

manuals_info = manuals_info['All_Manuals']

# tax_cat_list = list(dict_.keys())

category_hierarchy_dict = {}

for idx_, iter_dict in tqdm(enumerate(manuals_info)):
	category = iter_dict['category'].lower().replace(' manuals', '').replace(' equipment', '').replace(' equipments', '')

	if category in category_hierarchy_dict: # already covered
		continue

	max_sim = 0
	max_depth = 0
	hierarchy = []
	
	for dict_cat in dict_:
		dict_cat_l = dict_cat.lower()
		
		temp_depth = int(dict_[dict_cat]['depth'])
		temp_hierarchy = dict_[dict_cat]['hierarchy']
		if len(temp_hierarchy) <= 1:
			last_but_1 = ""
		else:
			last_but_1 = temp_hierarchy[-2].lower()
		temp_sim = get_total_sim(dict_cat_l, category, last_but_1)
		if temp_sim > max_sim and temp_depth > max_depth:
			hierarchy = temp_hierarchy
			max_depth = temp_depth
			max_sim = temp_sim

	category_hierarchy_dict[category] = hierarchy
	if idx_ < 20:
		print(category, hierarchy)
	# if idx_ == 20:
	# 	break

with open("category_hierarchy_dict.json", 'w') as f:
	json.dump(category_hierarchy_dict, f)