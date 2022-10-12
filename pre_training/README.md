# Pre-training

- Files/Folders

1. `taxonomy_hierarchy` - contains files related to hierarchical classification labels
2. `hier_utils.py` - contains a set of loss function definitions used during pre-training by different models.
3. `only_classfn_rule_based_shards.py` - code for pre-training the `hier.` model
4. `triplet_net_rule_based_shards.py` - code for pre-training the `triplet` model
5. `heirarchical_classes_triplet_net_rule_based_shards.py` - code for pre-training the `triplet + hier.` model
6. `only_classfn_two_stage_rule_based_shards.py` - code for pre-training the `hier.` model using 'Paragraph Encoder + 2-layer transformer' as document encoder
7. `two_stage_triplet_rule_based_shards.py` - code for pre-training the `triplet` model using 'Paragraph Encoder + 2-layer transformer' as document encoder
8. `heirarchical_classes_two_stage_triplet_rule_based_shards.py` - code for pre-training the `triplet + hier.` model using 'Paragraph Encoder + 2-layer transformer' as document encoder
9. `convert_to_HF_model.py` - converting `.pt` model generated during pre-training to a model contemporary with HuggingFace.
10. `models_.py` - contains classes for the pre-training architectures used.

> In order to pre-train RoBERTa-based variants, assign `model_type = 'roberta'` in the code, and for BERT-based variants, assign `model_type = 'bert'`.

> Also, for pre-training, run 

```
python3 <PRE-TRAINING FILENAME> 1
```

> Link to the sentence embeddings used in the codes (in the form of `.pickle` files) - https://drive.google.com/drive/folders/1phYFoYcheU7Kzs-kfXuy_RRNAd4qaryi?usp=sharing

