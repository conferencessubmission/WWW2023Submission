- Files/Folders

1. `taxonomy.en-US.txt` - contains Google Product Taxonomy
2. `parse_taxonomy.py` - parses taxonomy to generate the json files `taxonomy_category_hierarchy.json` and `taxonomy_classes_hierarchy.json`.
3. `get_silver_labels.py` - This takes in `links_2.json` (after unzipping `links_2.zip`)and `taxonomy_category_hierarchy.json` as input files, and generates `category_hierarchy_dict.json`.
4. `taxonomy_category_hierarchy.json` - dictionary, where keys are categories from the Google Product Taxonomy, and values are the depth in the hierarchy, and the parent categories of the category in the hierarchy.
5. `taxonomy_classes_hierarchy.json` - This is a dictionary where the keys are the depths, and the values are list of categories corresponding to that depth.
6. `links_2.zip` - When unzipped, we get `links_2.json`. This has a dictionary that contains information of the metadat in the E-Manuals in the E-Manuals Corpus
7. `category_hierarchy_dict.json` - This is a dictionary, where key is a category present in the EManuals Corpus, and the value is the list of hierarchical categories obtained from the Google Product Taxonomy.
8. `taxonomy_hier_classes_labels.json` - The key is the depth in the hierarchy, and the value is a dictionary, where the key is the hierarchical category from Google Product Taxonomy, and the value is the class label.
9. `filename_to_hier_class_labels.json` - This is a dictionary, where the key is the filename in the EManuals Corpus, and the value is a dictionary containing the hierarchical class labels corresponding to that E-Manual.
